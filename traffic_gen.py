#!/usr/bin/env python3
"""
traffic_gen.py  --  단순 UDP 송신기 (정상 또는 공격 트래픽).
  외부 도구(hping3/iperf) 없이 순수 Python으로 트래픽을 생성한다.
    --pps N : 초당 N개 패킷(정상 트래픽용, 예: 50)
    --pps 0 : 지연 없이 최대 속도로 전송(공격 flood용)

  사용 예:
    python3 traffic_gen.py --dst 10.0.0.1 --pps 50  --duration 60   # 정상
    python3 traffic_gen.py --dst 10.0.0.1 --pps 0   --duration 30   # flood
"""
import socket, time, argparse

ap = argparse.ArgumentParser()
ap.add_argument('--dst', required=True)
ap.add_argument('--port', type=int, default=9999)
ap.add_argument('--pps', type=int, default=50, help='0 = flood (no delay)')
ap.add_argument('--duration', type=int, default=30)
ap.add_argument('--size', type=int, default=64)
a = ap.parse_args()

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
msg = b'x' * a.size
interval = 1.0 / a.pps if a.pps > 0 else 0.0
t0 = time.time()
n = 0
while time.time() - t0 < a.duration:
    try:
        s.sendto(msg, (a.dst, a.port))
        n += 1
    except OSError:
        pass
    if interval:
        time.sleep(interval)
print(f'[traffic_gen] sent {n} packets to {a.dst}:{a.port} '
      f'({"flood" if a.pps == 0 else str(a.pps)+" pps"}, {a.duration}s)')
