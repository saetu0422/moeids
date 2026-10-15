#!/usr/bin/env python3
"""
demo_closed_loop.py  —  폐루프 시연 자동 실행 + 측정
실행 순서:
  1) 모델 준비 (시스템 python):   python3 train_rt_model.py      # moe_rt.joblib 생성
  2) 컨트롤러 (Ryu venv, 터미널1): source ~/ryu-env/bin/activate
                                    ryu-manager moe_closed_loop.py
  3) 시연 (sudo, 터미널2):        sudo python3 demo_closed_loop.py

측정 내용:
  - 공격 전/중/후 서버가 받는 정상 사용자 트래픽 처리량(응답 성공률)
  - 컨트롤러가 설치한 룰과 탐지 지연은 events.csv 에 기록됨
  - 결과: demo_timeline.csv (시간대별 정상 요청 성공률)

시나리오 타임라인(기본 90초):
  0~30s   정상만
  30~60s  정상 + SYN/UDP flood(공격)  -> 컨트롤러가 탐지·차단해야 함
  60~90s  공격 종료, 정상만 (회복 확인)
"""
import csv
import sys
import time

from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.link import TCLink
from mininet.log import setLogLevel, info

SERVER_IP = "10.0.0.100"
PHASE = 30          # 각 구간 길이(초)


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
    net.build(); c0.start(); s1.start([c0])
    nodes["srv"].cmd("python3 -m http.server 80 &")
    time.sleep(2)
    return net, nodes


def probe_success(h, n=10):
    """정상 사용자 h 가 서버에 n번 요청해서 성공 횟수 반환 (처리량 대리 지표)"""
    out = h.cmd(f"for i in $(seq 1 {n}); do "
                f"curl -s -o /dev/null -w '%{{http_code}}\\n' --max-time 1 "
                f"http://{SERVER_IP}/ ; done")
    return sum(1 for line in out.split() if line.strip() == "200")


def main():
    setLogLevel("info")
    net, nodes = make_net()
    timeline = []
    t_start = time.time()
    try:
        # ---- 구간1: 정상만 ----
        info("*** [0-%ds] 정상만\n" % PHASE)
        for _ in range(PHASE // 3):
            s = probe_success(nodes["h1"])
            timeline.append((round(time.time()-t_start), "normal", s))
            time.sleep(3)

        # ---- 구간2: 정상 + 공격 ----
        info("*** [%d-%ds] 공격 시작 (SYN + UDP flood)\n" % (PHASE, 2*PHASE))
        for a in ("a1", "a2", "a3"):
            nodes[a].cmd(f"timeout {PHASE} hping3 -S --flood -p 80 {SERVER_IP} "
                         f"--rand-source >/dev/null 2>&1 &")
        for _ in range(PHASE // 3):
            s = probe_success(nodes["h1"])
            timeline.append((round(time.time()-t_start), "attack", s))
            time.sleep(3)

        # ---- 구간3: 회복 ----
        info("*** [%d-%ds] 공격 종료, 회복 확인\n" % (2*PHASE, 3*PHASE))
        time.sleep(3)   # 공격 완전 종료 대기
        for _ in range(PHASE // 3):
            s = probe_success(nodes["h1"])
            timeline.append((round(time.time()-t_start), "recovery", s))
            time.sleep(3)

    finally:
        with open("demo_timeline.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["t_sec", "phase", "success_out_of_10"])
            w.writerows(timeline)
        info("*** demo_timeline.csv 저장 완료\n")
        for t, ph, s in timeline:
            info(f"  t={t:3d}s [{ph:8s}] 정상 요청 성공 {s}/10\n")
        net.stop()


if __name__ == "__main__":
    main()
