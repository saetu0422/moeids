#!/usr/bin/env python3
"""
explore_cicddos.py  —  CIC-DDoS2019 데이터 구조 탐색 (1단계)
  실제 실험 전에 파일 목록, 컬럼, 라벨 종류를 파악한다.
  (CIC-DDoS2019는 InSDN과 특징 세트는 같지만 컬럼명·라벨이 다를 수 있음)

실행:
  python3 explore_cicddos.py /path/to/cicddos2019_folder

출력:
  - 폴더 내 CSV 파일 목록과 크기
  - 각 파일의 컬럼 수와 라벨 컬럼 후보
  - 라벨(공격 유형) 종류와 개수
"""
import sys, os, glob
import pandas as pd

if len(sys.argv) < 2:
    print('사용법: python3 explore_cicddos.py <데이터폴더>')
    sys.exit(1)
folder = sys.argv[1]

files = sorted(glob.glob(os.path.join(folder, '*.csv')) +
               glob.glob(os.path.join(folder, '*.parquet')))
if not files:
    print(f'{folder} 에 csv/parquet 파일이 없습니다.')
    sys.exit(1)

print(f'=== {len(files)}개 파일 발견 ===')
for f in files:
    size_mb = os.path.getsize(f) / 1e6
    print(f'  {os.path.basename(f):40s} {size_mb:8.1f} MB')

# 첫 파일로 구조 파악
f0 = files[0]
print(f'\n=== 구조 확인: {os.path.basename(f0)} ===')
if f0.endswith('.parquet'):
    df = pd.read_parquet(f0)
else:
    df = pd.read_csv(f0, nrows=5000)
df.columns = df.columns.str.strip()   # 컬럼명 공백 제거
print(f'컬럼 수: {len(df.columns)}')

# 라벨 컬럼 후보 찾기
label_candidates = [c for c in df.columns if c.lower() in ('label','labels','class','attack')]
print(f'라벨 컬럼 후보: {label_candidates}')

if label_candidates:
    lc = label_candidates[0]
    print(f'\n=== 전체 파일의 라벨 분포 ({lc}) ===')
    all_labels = {}
    for f in files:
        try:
            if f.endswith('.parquet'):
                d = pd.read_parquet(f, columns=[lc] if lc in pd.read_parquet(f).columns else None)
            else:
                d = pd.read_csv(f, usecols=lambda c: c.strip() == lc)
            d.columns = d.columns.str.strip()
            for label, cnt in d[lc].astype(str).str.strip().value_counts().items():
                all_labels[label] = all_labels.get(label, 0) + cnt
        except Exception as e:
            print(f'  {os.path.basename(f)}: 읽기 실패 ({e})')
    for label, cnt in sorted(all_labels.items(), key=lambda x: -x[1]):
        print(f'  {label:20s}: {cnt:,}')

print('\n=== 컬럼 목록 (처음 20개) ===')
for i, c in enumerate(df.columns[:20]):
    print(f'  {i+1:2d}. {c}')
if len(df.columns) > 20:
    print(f'  ... 외 {len(df.columns)-20}개')

print('\n이 결과를 알려주시면 실험 스크립트를 맞춰 드립니다.')
