"""
moe_closed_loop.py  —  폐루프 SDN-IDS: 통계 수집 → MoE 탐지 → OpenFlow 룰 설치
실행:
  source ~/ryu-env/bin/activate
  # 먼저 시스템 python 으로 train_rt_model.py 를 돌려 moe_rt.joblib 을 만들어 둔다.
  # (Ryu venv 에도 scikit-learn 이 깔려 있어야 joblib 로드가 됨: setup 스크립트에서 설치함)
  ryu-manager moe_closed_loop.py

동작:
  - POLL_INTERVAL 마다 각 스위치의 플로우 통계를 수집한다.
  - 각 플로우의 특징을 만들어 MoE(top-k)로 공격 여부/유형을 판정한다.
  - 판정 결과를 표2 정책으로 매핑해 OpenFlow 룰을 설치한다:
       공격 확신(y>=THETA2)  -> drop 룰
       의심(THETA1<=y<THETA2) -> meter(rate-limit), meter 미지원 시 임시 drop
       정상                    -> 조치 없음
  - 모든 판정과 룰 설치를 events.csv 에 기록한다(플래핑/지연 분석용).

주의:
  - 이 앱은 학습 스위치 기능도 포함한다.
  - 룰 설치는 '출발지 IP' 기준으로 하며, 하드타임아웃으로 자동 만료시켜
    플로우 플래핑을 억제한다.
"""
import csv
import os
import time
from collections import defaultdict

import joblib
import numpy as np

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.lib import hub
from ryu.lib.packet import ethernet, ether_types, packet, ipv4, tcp
from ryu.ofproto import ofproto_v1_3

POLL_INTERVAL = 3.0
MODEL_PATH = "moe_rt.joblib"
EVENTS_CSV = "events.csv"
THETA1, THETA2 = 0.5, 0.8          # 표2 임계치
BLOCK_HARD_TIMEOUT = 20            # drop 룰 자동 만료(초): 플래핑 억제
USE_METER = True                   # meter 미지원 환경이면 False 로


class MoEClosedLoop(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.datapaths = {}
        self.mac_to_port = {}
        self.blocked = {}                       # (dpid, src_ip) -> 만료시각
        self.model = joblib.load(MODEL_PATH)
        self.classes = self.model['classes']
        self.norm_idx = self.classes.index('Normal') if 'Normal' in self.classes else -1
        self._init_csv()
        self.monitor = hub.spawn(self._monitor)
        self.logger.info("MoE 폐루프 IDS 시작. 클래스=%s", self.classes)

    def _init_csv(self):
        if not os.path.exists(EVENTS_CSV):
            with open(EVENTS_CSV, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["ts", "dpid", "src_ip", "pred_class", "attack_score",
                     "action", "install_latency_ms"])

    # ---------------- 학습 스위치 ----------------
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def _features(self, ev):
        dp = ev.msg.datapath; p = dp.ofproto_parser
        self._add_flow(dp, 0, p.OFPMatch(),
                       [p.OFPActionOutput(dp.ofproto.OFPP_CONTROLLER,
                                          dp.ofproto.OFPCML_NO_BUFFER)])
        self.datapaths[dp.id] = dp            # 통계 폴링 대상으로 직접 등록
        self.logger.info(">>> 스위치 연결됨 dpid=%s, table-miss 설치 완료 (폴링 시작)", dp.id)

    def _add_flow(self, dp, priority, match, actions, hard=0):
        p = dp.ofproto_parser
        inst = [p.OFPInstructionActions(dp.ofproto.OFPIT_APPLY_ACTIONS, actions)]
        dp.send_msg(p.OFPFlowMod(datapath=dp, priority=priority, match=match,
                                 instructions=inst, hard_timeout=hard))

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in(self, ev):
        msg = ev.msg; dp = msg.datapath; ofp = dp.ofproto; p = dp.ofproto_parser
        in_port = msg.match["in_port"]
        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)
        if eth is None or eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return
        dst = eth.dst; src = eth.src
        self.mac_to_port.setdefault(dp.id, {})
        self.mac_to_port[dp.id][src] = in_port
        out_port = self.mac_to_port[dp.id].get(dst, ofp.OFPP_FLOOD)
        actions = [p.OFPActionOutput(out_port)]
        # 목적지를 알면 플로우 설치. IP 패킷이면 출발지 IP를 매치에 넣어
        # 플로우 통계에 ipv4_src 가 잡히도록 한다(탐지 대상 확보).
        if out_port != ofp.OFPP_FLOOD:
            ip = pkt.get_protocol(ipv4.ipv4)
            if ip is not None:
                match = p.OFPMatch(in_port=in_port, eth_type=0x0800,
                                   ipv4_src=ip.src)
            else:
                match = p.OFPMatch(in_port=in_port, eth_dst=dst, eth_src=src)
            if msg.buffer_id != ofp.OFP_NO_BUFFER:
                self._add_flow(dp, 1, match, actions); return
            self._add_flow(dp, 1, match, actions)
        data = msg.data if msg.buffer_id == ofp.OFP_NO_BUFFER else None
        dp.send_msg(p.OFPPacketOut(datapath=dp, buffer_id=msg.buffer_id,
                                   in_port=in_port, actions=actions, data=data))

    # ---------------- 통계 수집 ----------------
    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER])
    def _state(self, ev):
        if ev.datapath.id:
            self.datapaths[ev.datapath.id] = ev.datapath

    def _monitor(self):
        while True:
            n = len(self.datapaths)
            for dp in list(self.datapaths.values()):
                dp.send_msg(dp.ofproto_parser.OFPFlowStatsRequest(dp))
            if n == 0:
                self.logger.info("[폴링] 등록된 스위치 없음 (대기 중)")
            hub.sleep(POLL_INTERVAL)

    def _features_from_flow(self, st):
        """플로우 통계 -> 모델 입력 벡터 (train_rt_model.py 의 특징 순서와 맞춤)
        수집 불가한 특징은 0 또는 근사값으로 채운다."""
        dur = st.duration_sec + st.duration_nsec / 1e9 or 1e-3
        fwd = st.packet_count; byts = st.byte_count
        return {
            'Flow Duration': dur * 1e6,
            'Tot Fwd Pkts': fwd, 'Tot Bwd Pkts': 0,
            'Flow Byts/s': byts / dur, 'Flow Pkts/s': fwd / dur,
            'Fwd Pkts/s': fwd / dur, 'Bwd Pkts/s': 0,
            'SYN Flag Cnt': 0, 'ACK Flag Cnt': 0,
            'Pkt Len Mean': byts / fwd if fwd else 0,
            'Pkt Size Avg': byts / fwd if fwd else 0, 'Down/Up Ratio': 0,
        }

    def _moe_predict(self, xdict):
        m = self.model
        x = np.array([[xdict.get(f, 0) for f in m['features']]])
        x = m['scaler'].transform(x)
        g = m['gate'].predict_proba(x)[0]
        ep = np.array([e.predict_proba(x)[0, 1] for e in m['experts']])
        k = m.get('k', 2)
        idx = np.argsort(g)[::-1][:k]
        gk = g[idx] / g[idx].sum()
        score = np.zeros(len(g)); score[idx] = gk * ep[idx]
        cls = int(score.argmax())
        attack_score = float(1 - score[self.norm_idx]) if self.norm_idx >= 0 else float(score.max())
        return self.classes[cls], attack_score

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats(self, ev):
        dp = ev.msg.datapath; p = dp.ofproto_parser; now = time.time()
        n_total = n_ip = 0
        for st in ev.msg.body:
            if st.priority == 0:
                continue
            n_total += 1
            src_ip = st.match.get('ipv4_src')
            if not src_ip:
                continue
            n_ip += 1
            if (dp.id, src_ip) in self.blocked and self.blocked[(dp.id, src_ip)] > now:
                continue                                    # 이미 차단 중
            pred, score = self._moe_predict(self._features_from_flow(st))
            action = self._apply_policy(dp, src_ip, score, now)
            if action != "none":
                self.logger.info("[탐지] src=%s pred=%s score=%.2f -> %s",
                                 src_ip, pred, score, action)
        # 진단: 통계를 받고는 있는지, IP 매치 플로우가 잡히는지 확인
        self.logger.info("[통계] 플로우 %d개 (IP매치 %d개)", n_total, n_ip)

    # ---------------- 완화 (표2 정책) ----------------
    def _apply_policy(self, dp, src_ip, y, now):
        p = dp.ofproto_parser; t0 = time.time()
        if y >= THETA2:
            match = p.OFPMatch(eth_type=0x0800, ipv4_src=src_ip)
            self._add_flow(dp, 100, match, [], hard=BLOCK_HARD_TIMEOUT)   # drop
            self.blocked[(dp.id, src_ip)] = now + BLOCK_HARD_TIMEOUT
            action = "drop"
        elif y >= THETA1:
            action = self._rate_limit(dp, src_ip)                        # meter/redirect
        else:
            action = "none"
        if action != "none":
            self._log(dp.id, src_ip, y, action, (time.time() - t0) * 1000)
        return action

    def _rate_limit(self, dp, src_ip):
        p = dp.ofproto_parser
        if USE_METER:
            try:
                bands = [p.OFPMeterBandDrop(rate=1000, burst_size=100)]
                dp.send_msg(p.OFPMeterMod(dp, dp.ofproto.OFPMC_ADD,
                                          dp.ofproto.OFPMF_KBPS, meter_id=1, bands=bands))
                match = p.OFPMatch(eth_type=0x0800, ipv4_src=src_ip)
                inst = [p.OFPInstructionMeter(1),
                        p.OFPInstructionActions(dp.ofproto.OFPIT_APPLY_ACTIONS,
                                                [p.OFPActionOutput(dp.ofproto.OFPP_NORMAL)])]
                dp.send_msg(p.OFPFlowMod(datapath=dp, priority=90, match=match,
                                         instructions=inst, hard_timeout=BLOCK_HARD_TIMEOUT))
                return "meter"
            except Exception:
                pass
        # meter 미지원: 약한 조치로 임시 drop
        self._add_flow(dp, 90, p.OFPMatch(eth_type=0x0800, ipv4_src=src_ip),
                       [], hard=BLOCK_HARD_TIMEOUT)
        return "weak_drop"

    def _log(self, dpid, src_ip, y, action, latency_ms):
        with open(EVENTS_CSV, "a", newline="") as f:
            csv.writer(f).writerow([round(time.time(), 2), dpid, src_ip,
                                    "", round(y, 3), action, round(latency_ms, 2)])
