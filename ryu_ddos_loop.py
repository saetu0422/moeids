#!/usr/bin/env python3
"""
ryu_ddos_loop.py  --  SDN DDoS closed-loop controller (detect -> OpenFlow drop)
  논문 프로토타입의 폐루프 컨트롤러. 포트/플로우 통계를 주기적으로 폴링해
  소스별 pps를 추정하고, 임계치(1차 필터)를 넘는 소스에 drop 룰을 설치한다.

  * 탐지기: 기본은 pps 임계치(논문 식(1) 1차 필터). 학습된 MoE를 끼우려면
    _detect() 안의 주석 지점에서 gating+experts를 호출하도록 교체하면 된다.
  * 로그: events.csv (완화 이벤트 + 룰 설치 지연). 전달 트래픽 측정은
    피해 호스트의 victim_monitor.py가 담당(의미가 더 명확).

실행 (터미널 1):
  ryu-manager ryu_ddos_loop.py
  # OpenFlow 1.3, 포트 6633 기본. Mininet은 run_mininet_demo.py 로.

설정: 아래 HOSTS / THRESH_PPS / POLL 를 환경에 맞게.
"""
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import (CONFIG_DISPATCHER, MAIN_DISPATCHER,
                                     DEAD_DISPATCHER, set_ev_cls)
from ryu.ofproto import ofproto_v1_3
from ryu.lib import hub
import time, csv, os

# --- 실험 상수 (환경에 맞게 수정) ---
VICTIM   = '10.0.0.1'                       # 피해/서버 (drop 대상에서 제외)
HOSTS    = ['10.0.0.1', '10.0.0.2', '10.0.0.6']  # 모니터링할 소스 IP (victim, legit, attacker)
POLL     = 2.0                              # 통계 폴링 주기(초)
THRESH_PPS = 300.0                          # 1차 필터 임계 pps -> 초과 시 공격 플래그
EVENTS   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'events.csv')


class DDoSLoop(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.datapaths = {}
        self.prev = {}          # (dpid, ip) -> (packet_count, timestamp)
        self.dropped = set()    # 이미 완화한 소스
        self.monitor_thread = hub.spawn(self._monitor)
        if not os.path.exists(EVENTS):
            with open(EVENTS, 'w', newline='') as f:
                csv.writer(f).writerow(
                    ['ts', 'dpid', 'src_ip', 'pred_class', 'attack_score',
                     'action', 'install_latency_ms'])

    # ---- datapath 등록 ----
    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change(self, ev):
        dp = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            self.datapaths[dp.id] = dp
        elif ev.state == DEAD_DISPATCHER:
            self.datapaths.pop(dp.id, None)

    # ---- 초기 플로우 설치 ----
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def _features(self, ev):
        dp = ev.msg.datapath
        ofp, psr = dp.ofproto, dp.ofproto_parser
        # 1) catch-all: ARP 등 일반 트래픽은 스위치가 NORMAL 포워딩
        self._add_flow(dp, 1, psr.OFPMatch(), [psr.OFPActionOutput(ofp.OFPP_NORMAL)])
        # 2) 소스별 모니터링 플로우(카운트 목적) — 역시 NORMAL 포워딩
        for ip in HOSTS:
            m = psr.OFPMatch(eth_type=0x0800, ipv4_src=ip)
            self._add_flow(dp, 10, m, [psr.OFPActionOutput(ofp.OFPP_NORMAL)])
        self.logger.info('init flows installed on dpid=%s', dp.id)

    def _add_flow(self, dp, prio, match, actions):
        ofp, psr = dp.ofproto, dp.ofproto_parser
        inst = [psr.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]
        dp.send_msg(psr.OFPFlowMod(datapath=dp, priority=prio,
                                   match=match, instructions=inst))

    def _install_drop(self, dp, ip):
        """해당 소스 IP를 drop (빈 action = drop), 모니터 플로우보다 높은 우선순위."""
        psr = dp.ofproto_parser
        m = psr.OFPMatch(eth_type=0x0800, ipv4_src=ip)
        inst = [psr.OFPInstructionActions(dp.ofproto.OFPIT_APPLY_ACTIONS, [])]
        dp.send_msg(psr.OFPFlowMod(datapath=dp, priority=100,
                                   match=m, instructions=inst))

    # ---- 주기적 통계 폴링 ----
    def _monitor(self):
        while True:
            for dp in list(self.datapaths.values()):
                dp.send_msg(dp.ofproto_parser.OFPFlowStatsRequest(dp))
            hub.sleep(POLL)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats(self, ev):
        dp = ev.msg.datapath
        now = time.time()
        for st in ev.msg.body:
            ip = st.match.get('ipv4_src')
            if ip is None:
                continue
            key = (dp.id, ip)
            pkts = st.packet_count
            if key in self.prev:
                p0, t0 = self.prev[key]
                dt = now - t0
                pps = (pkts - p0) / dt if dt > 0 else 0.0
                self._detect(dp, ip, pps)
            self.prev[key] = (pkts, now)

    def _detect(self, dp, ip, pps):
        """1차 필터(pps 임계). MoE를 쓰려면 여기서 gating+experts로 교체."""
        if ip == VICTIM or ip in self.dropped:
            return
        # === MoE 교체 지점: score = moe_predict(features_of(ip)) ===
        is_attack = pps >= THRESH_PPS
        if is_attack:
            t_req = time.time()
            self._install_drop(dp, ip)
            lat_ms = (time.time() - t_req) * 1000.0
            self.dropped.add(ip)
            with open(EVENTS, 'a', newline='') as f:
                csv.writer(f).writerow(
                    [f'{t_req:.2f}', dp.id, ip, '', 1.0, 'drop', f'{lat_ms:.2f}'])
            self.logger.info('MITIGATE src=%s pps=%.0f latency=%.2fms', ip, pps, lat_ms)
