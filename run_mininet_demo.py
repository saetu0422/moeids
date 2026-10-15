#!/usr/bin/env python3
"""
run_mininet_demo.py  --  mitigation effectiveness 실험 (Ryu 불필요, OVS 단독).

  Ryu 컨트롤러 없이 Mininet + Open vSwitch 만으로 폐루프를 구성한다.
  탐지·완화는 ovs-ofctl 로 직접 수행:
    - 소스별 모니터링 플로우(priority 10, actions=NORMAL)를 설치해 패킷 수를 셈
    - 주기적으로 dump-flows 로 소스별 pps 추정
    - 임계치 초과 소스에 drop 플로우(priority 100) 설치 = 완화
  피해 호스트(h1)에서 victim_monitor.py 가 '실제 도달' 트래픽을 초단위로 기록.

  토폴로지: s1 + h1(10.0.0.1 피해) h2(10.0.0.2 정상) h6(10.0.0.6 공격자)
  타임라인: 0~15s 정상 → 15s~ 공격 flood → 탐지/완화 후 attack 도달 급감

  실행 (한 줄, 루트 권한):
    sudo python3 run_mininet_demo.py
  끝나면:
    python3 run_mitigation_effectiveness.py --traffic traffic_log.csv --events events.csv

  * '진짜' Ryu 컨트롤러 버전을 원하면 함께 드린 ryu_ddos_loop.py 를 쓰면 되지만,
    Ryu 설치가 필요하다(아래 설명 참고).
"""
import os, sys, re, csv, time, threading
from mininet.net import Mininet
from mininet.node import OVSSwitch
from mininet.link import TCLink
from mininet.log import setLogLevel

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable or 'python3'

VICTIM = '10.0.0.1'
HOSTS = ['10.0.0.1', '10.0.0.2', '10.0.0.6']
POLL = 2.0
THRESH_PPS = 300.0
NORMAL_SEC = 15
ATTACK_SEC = 30
TOTAL = NORMAL_SEC + ATTACK_SEC + 13
EVENTS = os.path.join(HERE, 'events.csv')


def ofctl(sw, cmd):
    return sw.cmd(f'ovs-ofctl -O OpenFlow13 {cmd}')


def detect_loop(sw, stop_at):
    """OVS 플로우 통계를 폴링해 소스별 pps 추정, 임계 초과 시 drop 설치."""
    prev = {}
    dropped = set()
    with open(EVENTS, 'w', newline='') as f:
        csv.writer(f).writerow(['ts', 'dpid', 'src_ip', 'pred_class',
                                'attack_score', 'action', 'install_latency_ms'])
    while time.time() < stop_at:
        for ip in HOSTS:
            out = ofctl(sw, f'dump-flows s1 "ip,nw_src={ip}"')
            pk = sum(int(m) for m in re.findall(r'n_packets=(\d+)', out))
            now = time.time()
            if ip in prev:
                p0, t0 = prev[ip]
                dt = now - t0
                pps = (pk - p0) / dt if dt > 0 else 0.0
                if ip not in dropped and ip != VICTIM and pps >= THRESH_PPS:
                    t_req = time.time()
                    ofctl(sw, f'add-flow s1 "priority=100,ip,nw_src={ip},actions=drop"')
                    lat = (time.time() - t_req) * 1000.0
                    dropped.add(ip)
                    with open(EVENTS, 'a', newline='') as f:
                        csv.writer(f).writerow(
                            [f'{t_req:.2f}', 1, ip, '', 1.0, 'drop', f'{lat:.2f}'])
                    print(f'  [MITIGATE] src={ip} pps={pps:.0f} latency={lat:.2f}ms')
            prev[ip] = (pk, now)
        time.sleep(POLL)


def main():
    net = Mininet(controller=None, switch=OVSSwitch, link=TCLink, autoSetMacs=True)
    s1 = net.addSwitch('s1', failMode='secure', protocols='OpenFlow13')
    h1 = net.addHost('h1', ip='10.0.0.1')
    h2 = net.addHost('h2', ip='10.0.0.2')
    h6 = net.addHost('h6', ip='10.0.0.6')
    for h in (h1, h2, h6):
        net.addLink(h, s1)
    net.build()
    s1.start([])                      # 컨트롤러 없이 시작

    # 기반 플로우: ARP/일반은 NORMAL, 소스별 모니터 플로우(카운트)
    ofctl(s1, 'add-flow s1 "priority=1,actions=NORMAL"')
    for ip in HOSTS:
        ofctl(s1, f'add-flow s1 "priority=10,ip,nw_src={ip},actions=NORMAL"')
    time.sleep(1)
    net.pingAll()                     # ARP 사전 해소

    stop_at = time.time() + TOTAL + 2
    th = threading.Thread(target=detect_loop, args=(s1, stop_at), daemon=True)
    th.start()

    # 피해 호스트 로깅 (매초 flush 하므로 중간 종료돼도 데이터 유지)
    exp_start = time.time()
    h1.cmd(f'{PY} {HERE}/victim_monitor.py --iface h1-eth0 '
           f'--attacker 10.0.0.6 --victim 10.0.0.1 --duration {TOTAL} '
           f'--out {HERE}/traffic_log.csv > {HERE}/victim_monitor.log 2>&1 &')
    # 정상 트래픽 내내
    h2.cmd(f'{PY} {HERE}/traffic_gen.py --dst 10.0.0.1 --pps 50 '
           f'--duration {TOTAL} > /dev/null 2>&1 &')

    print(f'[phase] NORMAL (0~{NORMAL_SEC}s)')
    time.sleep(NORMAL_SEC)
    print(f'[phase] ATTACK ({NORMAL_SEC}s~, h6 flood {ATTACK_SEC}s)')
    h6.cmd(f'{PY} {HERE}/traffic_gen.py --dst 10.0.0.1 --pps 0 '
           f'--duration {ATTACK_SEC} > /dev/null 2>&1 &')

    # victim_monitor(--duration TOTAL)가 완전히 끝나고 파일을 닫을 때까지 대기(+5s 여유)
    while time.time() - exp_start < TOTAL + 5:
        time.sleep(1)
    print('[phase] DONE')
    net.stop()
    print(f'\n산출물:\n  {HERE}/traffic_log.csv\n  {HERE}/events.csv')
    print('다음:\n  python3 run_mitigation_effectiveness.py '
          '--traffic traffic_log.csv --events events.csv')


if __name__ == '__main__':
    setLogLevel('warning')
    main()
