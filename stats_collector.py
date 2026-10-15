"""
stats_collector.py  —  Ryu 앱: 통계 수집 + 특징 추출 → CSV
실행:
  source ~/ryu-env/bin/activate
  ryu-manager stats_collector.py

- POLL_INTERVAL 초마다 각 스위치에 포트/플로우 통계를 요청한다.
- 직전 폴링과의 '차분(delta)'으로 시간 윈도우 단위 rate 특징을 만든다.
- 라벨은 환경변수 TRAFFIC_LABEL 로 준다 (트래픽 스크립트에서 지정).
    export TRAFFIC_LABEL=syn_flood   # normal / syn_flood / udp_flood / flash_crowd
- 결과: dataset.csv (없으면 헤더 포함 새로 생성, 있으면 이어붙임)

주의: 이 앱은 simple_switch_13 의 학습 스위치 기능도 포함하므로
      별도의 스위치 앱을 같이 띄울 필요가 없다.
"""
import csv
import os
import time
from collections import defaultdict

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.lib import hub
from ryu.lib.packet import ethernet, ether_types, packet
from ryu.ofproto import ofproto_v1_3

POLL_INTERVAL = 2.0          # 통계 폴링 주기(초) = 특징 윈도우 길이
CSV_PATH = "dataset.csv"
FIELDS = [
    "ts", "dpid", "label",
    "pps", "bps", "new_flow_rate",     # rate 특징 (윈도우당 delta / 시간)
    "syn_rate", "ack_rate", "ack_to_syn_ratio",
    "active_flows", "avg_flow_pkts", "avg_flow_dur_s",
    "rx_pps", "tx_pps",
]


class StatsCollector(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label = os.environ.get("TRAFFIC_LABEL", "unknown")
        self.datapaths = {}
        self.mac_to_port = {}                    # 학습 스위치용
        self.prev = defaultdict(dict)            # dpid -> 직전 카운터
        self._init_csv()
        self.monitor_thread = hub.spawn(self._monitor)
        self.logger.info("StatsCollector 시작. label=%s, 주기=%ss", self.label, POLL_INTERVAL)

    def _init_csv(self):
        if not os.path.exists(CSV_PATH):
            with open(CSV_PATH, "w", newline="") as f:
                csv.writer(f).writerow(FIELDS)

    # ---------------- 학습 스위치 (연결 유지를 위해 필요) ----------------
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        dp = ev.msg.datapath
        p = dp.ofproto_parser
        # table-miss: 컨트롤러로 전송
        self._add_flow(dp, 0, p.OFPMatch(),
                       [p.OFPActionOutput(dp.ofproto.OFPP_CONTROLLER,
                                          dp.ofproto.OFPCML_NO_BUFFER)])

    def _add_flow(self, dp, priority, match, actions):
        p = dp.ofproto_parser
        inst = [p.OFPInstructionActions(dp.ofproto.OFPIT_APPLY_ACTIONS, actions)]
        dp.send_msg(p.OFPFlowMod(datapath=dp, priority=priority,
                                 match=match, instructions=inst))

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        msg = ev.msg
        dp = msg.datapath
        p = dp.ofproto_parser
        in_port = msg.match["in_port"]
        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)
        if eth is None or eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return
        self.mac_to_port.setdefault(dp.id, {})
        self.mac_to_port[dp.id][eth.src] = in_port
        out_port = self.mac_to_port[dp.id].get(eth.dst, dp.ofproto.OFPP_FLOOD)
        actions = [p.OFPActionOutput(out_port)]
        if out_port != dp.ofproto.OFPP_FLOOD:
            self._add_flow(dp, 1, p.OFPMatch(in_port=in_port, eth_dst=eth.dst), actions)
        dp.send_msg(p.OFPPacketOut(datapath=dp, buffer_id=msg.buffer_id,
                                   in_port=in_port, actions=actions,
                                   data=msg.data if msg.buffer_id == dp.ofproto.OFP_NO_BUFFER else None))

    # ---------------- 통계 폴링 ----------------
    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER])
    def _state_change(self, ev):
        dp = ev.datapath
        if dp.id:
            self.datapaths[dp.id] = dp

    def _monitor(self):
        while True:
            for dp in list(self.datapaths.values()):
                p = dp.ofproto_parser
                dp.send_msg(p.OFPPortStatsRequest(dp, 0, dp.ofproto.OFPP_ANY))
                dp.send_msg(p.OFPFlowStatsRequest(dp))
            hub.sleep(POLL_INTERVAL)

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def _port_stats(self, ev):
        dpid = ev.msg.datapath.id
        rx = sum(s.rx_packets for s in ev.msg.body)
        tx = sum(s.tx_packets for s in ev.msg.body)
        rx_b = sum(s.rx_bytes for s in ev.msg.body)
        self.prev[dpid].update({"_rx": rx, "_tx": tx, "_rxb": rx_b, "_pt": time.time()})

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats(self, ev):
        dpid = ev.msg.datapath.id
        now = time.time()
        flows = [f for f in ev.msg.body if f.priority > 0]
        n_flows = len(flows)
        tot_pkts = sum(f.packet_count for f in flows)
        tot_dur = sum(f.duration_sec + f.duration_nsec / 1e9 for f in flows)
        # TCP 플래그별 근사: match 에 tcp_flags 를 넣지 않으므로,
        # SYN 은 'tcp 목적지로 향하는 신규 단방향 플로우' 수로 근사한다.
        syn = sum(1 for f in flows if f.match.get("ip_proto") == 6)
        ack = sum(1 for f in flows if f.match.get("ip_proto") == 6 and f.packet_count > 2)

        pv = self.prev[dpid]
        dt = now - pv.get("_ft", now) or POLL_INTERVAL
        d_flows = max(0, n_flows - pv.get("_nflows", 0))
        d_pkts = max(0, tot_pkts - pv.get("_pkts", 0))
        d_rx = max(0, pv.get("_rx", 0) - pv.get("_rx_prev", 0))
        d_tx = max(0, pv.get("_tx", 0) - pv.get("_tx_prev", 0))
        d_rxb = max(0, pv.get("_rxb", 0) - pv.get("_rxb_prev", 0))

        row = {
            "ts": round(now, 2), "dpid": dpid, "label": self.label,
            "pps": round(d_pkts / dt, 2),
            "bps": round(d_rxb / dt, 2),
            "new_flow_rate": round(d_flows / dt, 2),
            "syn_rate": round(syn / dt, 2),
            "ack_rate": round(ack / dt, 2),
            "ack_to_syn_ratio": round(ack / syn, 3) if syn else 1.0,
            "active_flows": n_flows,
            "avg_flow_pkts": round(tot_pkts / n_flows, 2) if n_flows else 0,
            "avg_flow_dur_s": round(tot_dur / n_flows, 2) if n_flows else 0,
            "rx_pps": round(d_rx / dt, 2),
            "tx_pps": round(d_tx / dt, 2),
        }
        with open(CSV_PATH, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
        self.logger.info("[%s] pps=%.0f bps=%.0f newflow/s=%.1f ack/syn=%.2f flows=%d",
                         self.label, row["pps"], row["bps"], row["new_flow_rate"],
                         row["ack_to_syn_ratio"], n_flows)

        # 다음 윈도우용 저장
        pv.update({"_nflows": n_flows, "_pkts": tot_pkts, "_ft": now,
                   "_rx_prev": pv.get("_rx", 0), "_tx_prev": pv.get("_tx", 0),
                   "_rxb_prev": pv.get("_rxb", 0)})
