#!/usr/bin/env python3
"""
victim_monitor.py  --  피해 호스트에서 '실제 전달된' 트래픽을 초단위로 기록.
  raw AF_PACKET 소켓으로 피해 인터페이스에 도착한 IPv4 패킷을 소스별로 센다.
  완화(drop 룰) 설치 후에는 공격 패킷이 피해에 도달하지 못하므로 attack_pps가
  떨어진다 -> 이 로그가 곧 mitigation effectiveness 의 근거.

  * 매초 CSV에 즉시 append + flush 하므로, 프로세스가 중간에 종료돼도
    그때까지의 데이터가 남는다(이전 버전은 끝에서 한 번만 저장해 유실됐음).

  출력 traffic_log.csv 컬럼: t_sec, attack_pps, legit_pps
    attack_pps = 공격자 IP발 도착 pps,  legit_pps = 그 외(피해 자신 제외) 도착 pps
"""
import socket, time, csv, argparse

ap = argparse.ArgumentParser()
ap.add_argument('--iface', required=True)
ap.add_argument('--attacker', default='10.0.0.6')
ap.add_argument('--victim', default='10.0.0.1')
ap.add_argument('--duration', type=int, default=60)
ap.add_argument('--out', default='traffic_log.csv')
a = ap.parse_args()

ETH_P_ALL = 0x0003
s = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(ETH_P_ALL))
s.bind((a.iface, 0))
s.settimeout(0.5)


def ipv4_src(pkt):
    # Ethernet(14) + IPv4; ethertype at [12:14], IP src at [26:30]
    if len(pkt) < 34 or pkt[12:14] != b'\x08\x00':
        return None
    return socket.inet_ntoa(pkt[26:30])


f = open(a.out, 'w', newline='')
w = csv.writer(f)
w.writerow(['t_sec', 'attack_pps', 'legit_pps'])
f.flush()

t0 = time.time()
sec = 0
atk = leg = 0
print(f'[victim_monitor] iface={a.iface} attacker={a.attacker} dur={a.duration}s -> {a.out}')
while time.time() - t0 < a.duration:
    try:
        pkt = s.recv(2048)
    except socket.timeout:
        pkt = None
    if pkt:
        src = ipv4_src(pkt)
        if src == a.attacker:
            atk += 1
        elif src and src != a.victim:
            leg += 1
    cur = int(time.time() - t0)
    if cur > sec:
        w.writerow([sec, atk, leg]); f.flush()      # 매초 즉시 저장
        print(f't={sec:3d}s  attack_pps={atk:6d}  legit_pps={leg:5d}')
        sec = cur
        atk = leg = 0

w.writerow([sec, atk, leg]); f.flush()
f.close()
print(f'[victim_monitor] done -> {a.out}')
