#!/usr/bin/env python3
"""
run_scenarios.py  —  라벨별 트래픽 생성 자동화 (Mininet API)
실행:
  # 터미널 1: 라벨은 이 스크립트가 컨트롤러를 재시작하며 넘겨줄 수 없으므로,
  # 각 시나리오 구간을 CSV 의 label 로 남기기 위해 아래 두 가지 중 하나를 쓴다.
  #  (A) 간단: 시나리오마다 TRAFFIC_LABEL 을 바꿔 ryu-manager 를 따로 실행하고,
  #            이 스크립트의 해당 함수만 각각 돌린다 (권장, 라벨이 깔끔함).
  #  (B) 한 번에: 아래 main 을 그대로 돌리고, 나중에 ts 구간으로 라벨을 보정한다.
  #
  # (A) 방식 예시 —— SYN flood 데이터 수집:
  #   [터미널1] export TRAFFIC_LABEL=syn_flood; ryu-manager stats_collector.py
  #   [터미널2] sudo python3 topo.py
  #   [터미널3] sudo python3 run_scenarios.py syn_flood
  #
  # 시나리오: normal / syn_flood / udp_flood / flash_crowd

주의:
  - 이 스크립트는 이미 떠 있는 Mininet 노드에 명령을 넣기 위해,
    topo.py 대신 자체적으로 net 을 만들어 실행한다(독립 실행형).
  - 서버 srv 에서 http.server 를 자동으로 띄운다.
  - hping3 는 root 권한이 필요하므로 sudo 로 실행한다.
"""
import sys
import time

from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.link import TCLink
from mininet.log import setLogLevel, info

DURATION = 60          # 시나리오당 트래픽 지속 시간(초)
SERVER_IP = "10.0.0.100"


def make_net():
    net = Mininet(controller=None, switch=OVSSwitch, link=TCLink)
    c0 = net.addController("c0", controller=RemoteController, ip="127.0.0.1", port=6653)
    s1 = net.addSwitch("s1", protocols="OpenFlow13")
    names = {"h1": "10.0.0.1", "h2": "10.0.0.2", "h3": "10.0.0.3",
             "a1": "10.0.0.11", "a2": "10.0.0.12", "a3": "10.0.0.13",
             "srv": SERVER_IP}
    nodes = {n: net.addHost(n, ip=f"{ip}/24") for n, ip in names.items()}
    for n in nodes.values():
        net.addLink(n, s1, bw=10)
    net.build()
    c0.start()
    s1.start([c0])
    nodes["srv"].cmd("python3 -m http.server 80 &")
    time.sleep(2)
    return net, nodes


def normal(nodes):
    """정상 사용자: 완만한 웹 요청 + 간간이 ping"""
    info("*** normal 트래픽\n")
    for h in ("h1", "h2", "h3"):
        nodes[h].cmd(f"( for i in $(seq 1 {DURATION}); do "
                     f"curl -s http://{SERVER_IP}/ >/dev/null; sleep 1; done ) &")
    time.sleep(DURATION)


def syn_flood(nodes):
    """SYN flood: 완료되지 않는 다량의 SYN (ack_to_syn 급락 유도)"""
    info("*** syn_flood 트래픽\n")
    for a in ("a1", "a2", "a3"):
        nodes[a].cmd(f"timeout {DURATION} hping3 -S --flood -p 80 {SERVER_IP} "
                     f"--rand-source >/dev/null 2>&1 &")
    time.sleep(DURATION + 2)


def udp_flood(nodes):
    """volumetric UDP flood: pps/bps 급증"""
    info("*** udp_flood 트래픽\n")
    for a in ("a1", "a2", "a3"):
        nodes[a].cmd(f"timeout {DURATION} hping3 --udp --flood -p 53 {SERVER_IP} "
                     f">/dev/null 2>&1 &")
    time.sleep(DURATION + 2)


def flash_crowd(nodes):
    """정상 급증: 많은 정상 사용자가 동시에 몰림 (핸드셰이크는 정상 완료)"""
    info("*** flash_crowd 트래픽\n")
    # 정상 3대가 동시에 빠른 반복 요청 → 볼륨은 커지되 ack_to_syn 은 정상
    for h in ("h1", "h2", "h3"):
        nodes[h].cmd(f"( for i in $(seq 1 {DURATION * 20}); do "
                     f"curl -s http://{SERVER_IP}/ >/dev/null; done ) &")
    time.sleep(DURATION)


SCENARIOS = {"normal": normal, "syn_flood": syn_flood,
             "udp_flood": udp_flood, "flash_crowd": flash_crowd}


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in SCENARIOS:
        print("사용법: sudo python3 run_scenarios.py [normal|syn_flood|udp_flood|flash_crowd]")
        sys.exit(1)
    setLogLevel("info")
    net, nodes = make_net()
    try:
        SCENARIOS[sys.argv[1]](nodes)
        info("*** 트래픽 종료. 통계 안정화 대기\n")
        time.sleep(4)
    finally:
        net.stop()


if __name__ == "__main__":
    main()
