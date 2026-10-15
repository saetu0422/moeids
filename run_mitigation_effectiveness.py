#!/usr/bin/env python3
"""
run_mitigation_effectiveness.py
  리뷰어 지적 2 대응: closed-loop의 "실제 완화 효과(mitigation effectiveness)"를 수치로 제시.

  두 가지 입력 중 가진 것으로 측정한다(둘 다 주면 둘 다 보고):

  (A) --traffic traffic_log.csv   (가장 강한 근거)
      컬럼: t_sec, attack_pps, legit_pps
        - 스위치 포트/플로우 통계를 1초(또는 polling 주기)마다 기록.
        - attack_pps = 공격 소스(예: 10.0.0.6)발 pps,  legit_pps = 정상 트래픽 pps.
      → 완화(첫 drop rule 설치) 전/후 window의 평균 attack pps 비교 →
        '공격 트래픽 감소율', 그리고 정상 throughput 유지율을 계산.

  (B) --timeline demo_timeline.csv
      컬럼: t_sec, phase(normal|attack|recovery), success_out_of_10
        - 10회 정상 요청 중 성공 수(서비스 가용성 proxy).
      → 구간별 평균 성공률, attack 구간 대비 recovery 구간 회복률 계산.

  --events events.csv (ts, ..., action, ...) 가 있으면 첫 'drop' 설치 시각을 완화 시점으로 사용.
     (traffic_log의 t_sec가 epoch가 아니면 --mitig-t 로 완화 시점을 직접 지정)

실행 예:
  python3 run_mitigation_effectiveness.py --timeline demo_timeline.csv
  python3 run_mitigation_effectiveness.py --traffic traffic_log.csv --events events.csv
  python3 run_mitigation_effectiveness.py --traffic traffic_log.csv --mitig-t 31 --win 8

출력:
  표(stdout) + mitigation_effectiveness.csv

[주의] demo_timeline.csv처럼 세 구간 모두 success가 동일하면(저하가 없으면)
  완화 효과를 '보여줄' 수 없다. 반드시 attack 구간에서 서비스가 저하되고
  recovery 구간에서 회복되는 실측 로그를 사용할 것.
"""
import argparse, numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--traffic', default=None, help='t_sec, attack_pps, legit_pps')
ap.add_argument('--timeline', default=None, help='t_sec, phase, success_out_of_10')
ap.add_argument('--events', default=None, help='events.csv (ts,...,action,...)')
ap.add_argument('--mitig-t', type=float, default=None, help='완화 시점(초). 미지정 시 events의 첫 drop')
ap.add_argument('--win', type=float, default=8.0, help='완화 전/후 비교 window 길이(초)')
args = ap.parse_args()

rows_out = []

# ---------- (A) traffic-based: attack pps 감소율 + 정상 throughput 유지 ----------
if args.traffic:
    tr = pd.read_csv(args.traffic); tr.columns = tr.columns.str.strip()
    t = tr['t_sec'].values.astype(float)
    atk = tr['attack_pps'].values.astype(float)
    leg = tr['legit_pps'].values.astype(float) if 'legit_pps' in tr.columns else None

    # 완화 시점
    mt = args.mitig_t
    if mt is None and args.events:
        ev = pd.read_csv(args.events); ev.columns = ev.columns.str.strip()
        drops = ev[ev['action'].astype(str).str.strip() == 'drop']
        if len(drops):
            mt = float(drops['ts'].iloc[0])
            # events ts가 epoch이고 traffic t_sec가 상대초면 맞춰줌
            if mt > 1e6 and t.max() < 1e6:
                mt = mt - float(ev['ts'].iloc[0])   # 첫 이벤트를 0초로
    if mt is None:
        raise SystemExit('완화 시점을 모름: --mitig-t 또는 --events 로 지정하세요.')

    pre = (t >= mt - args.win) & (t < mt)
    post = (t > mt) & (t <= mt + args.win)
    atk_pre, atk_post = atk[pre].mean(), atk[post].mean()
    reduction = 100 * (atk_pre - atk_post) / atk_pre if atk_pre > 0 else float('nan')
    print(f'\n=== (A) Attack-traffic reduction (완화 전/후 {args.win:.0f}s window) ===')
    print(f'  완화 시점 t = {mt:.2f}s')
    print(f'  공격 pps  전: {atk_pre:10.1f}  → 후: {atk_post:10.1f}   감소율 = {reduction:5.1f}%')
    rows_out += [['attack_pps_before', f'{atk_pre:.1f}'],
                 ['attack_pps_after',  f'{atk_post:.1f}'],
                 ['attack_reduction_%', f'{reduction:.1f}']]
    if leg is not None:
        leg_pre, leg_post = leg[pre].mean(), leg[post].mean()
        retain = 100 * leg_post / leg_pre if leg_pre > 0 else float('nan')
        print(f'  정상 pps  전: {leg_pre:10.1f}  → 후: {leg_post:10.1f}   유지율 = {retain:5.1f}%')
        rows_out += [['legit_pps_before', f'{leg_pre:.1f}'],
                     ['legit_pps_after',  f'{leg_post:.1f}'],
                     ['legit_retention_%', f'{retain:.1f}']]

# ---------- (B) timeline-based: 구간별 서비스 가용성 + 회복률 ----------
if args.timeline:
    tl = pd.read_csv(args.timeline); tl.columns = tl.columns.str.strip()
    tl['phase'] = tl['phase'].astype(str).str.strip()
    g = tl.groupby('phase')['success_out_of_10'].mean()
    print(f'\n=== (B) Service availability by phase (success/10) ===')
    for ph in ['normal', 'attack', 'recovery']:
        if ph in g.index:
            print(f'  {ph:9s}: {g[ph]:.2f} / 10')
            rows_out.append([f'avail_{ph}', f'{g[ph]:.2f}'])
    if {'normal', 'attack', 'recovery'} <= set(g.index):
        drop = g['normal'] - g['attack']
        recov = g['recovery'] - g['attack']
        rr = 100 * recov / drop if drop > 0 else float('nan')
        print(f'  공격 중 저하 = {drop:.2f},  완화 후 회복 = {recov:.2f}'
              f'  → 회복률 = {rr:.0f}%' if drop > 0 else
              '  [경고] 공격 중 서비스 저하가 없어 완화 효과를 보일 수 없음. 실측 로그 필요.')
        if drop > 0:
            rows_out.append(['recovery_rate_%', f'{rr:.0f}'])

if not args.traffic and not args.timeline:
    raise SystemExit('--traffic 또는 --timeline 중 하나는 필요합니다.')

if rows_out:
    pd.DataFrame(rows_out, columns=['metric', 'value']).to_csv('mitigation_effectiveness.csv', index=False)
    print('\nsaved: mitigation_effectiveness.csv')
