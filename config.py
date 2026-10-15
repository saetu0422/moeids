#!/usr/bin/env python3
"""
topo.py  —  Mininet 토폴로지 (단일 스위치, 호스트 7대)
실행:
  sudo python3 topo.py            # 대화형 CLI 진입
  또는 sudo mn -c 로 정리 후 사용

구성:
  s1 에 연결:
    h1, h2, h3  : 정상 사용자 (10.0.0.1~3)
    a1, a2, a3  : 공격자      (10.0.0.11~13)
    srv         : 대상 서버   (10.0.0.100)
컨트롤러: 127.0.0.1:6653 (ryu-manager stats_collector.py)
"""
from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.cli import CLI
from mininet.link import TCLink
from mininet.log import setLogLevel


def build():
    net = Mininet(controller=None, switch=OVSSwitch, link=TCLink)
    c0 = net.addController("c0", controller=RemoteController,
                           ip="127.0.0.1", port=6653)
    s1 = net.addSwitch("s1", protocols="OpenFlow13")

    hosts = {
        "h1": "10.0.0.1", "h2": "10.0.0.2", "h3": "10.0.0.3",
        "a1": "10.0.0.11", "a2": "10.0.0.12", "a3": "10.0.0.13",
        "srv": "10.0.0.100",
    }
    for name, ip in hosts.items():
        h = net.addHost(name, ip=f"{ip}/24")
        # 링크 대역폭을 제한해 공격 시 포화 효과가 통계에 드러나게 함
        net.addLink(h, s1, bw=10)

    net.build()
    c0.start()
    s1.start([c0])
    print("토폴로지 준비 완료. 서버(srv)에서 웹/echo 서비스를 먼저 띄우세요.")
    print("예) mininet> srv python3 -m http.server 80 &")
    CLI(net)
    net.stop()


if __name__ == "__main__":
    setLogLevel("info")
    build()
