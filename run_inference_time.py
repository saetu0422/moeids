#!/usr/bin/env python3
"""
run_inference_time.py  —  3단계 추가 실험: 추론 시간 측정 (직접 실행)
  - MoE의 top-k별 실제 추론 시간을 측정하여 "조건부 계산 = 효율" 주장을 실측으로 뒷받침
  - 랜덤 포레스트, 결정 트리, 단일 MLP와 비교
  - 샘플당 추론 시간(마이크로초)과 처리량(초당 샘플 수)을 보고

실행:
  cd ~/Downloads/moeids
  python3 run_inference_time.py

결과물:
  table_inference.csv          (모델별 추론 시간·처리량)
  fig_inference_time.png       (top-k별 추론 시간 그래프)
"""
import warnings, time, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

DROP = ['Flow ID','Src IP','Dst IP','Timestamp','Src Port','Dst Port']
N_REPEAT = 5          # 시간 측정 반복 (평균)
RANDOM_STATE = 42

def load(fn):
    d = pd.read_csv(fn); d['Label'] = d['Label'].astype(str).str.strip(); return d

print('데이터 로딩...')
df = pd.concat([load('Normal_data.csv'), load('OVS.csv'),
                load('metasploitable-2.csv')], ignore_index=True)
df['cls'] = df['Label'].apply(lambda l: l if l in ('Normal','DDoS','DoS','Probe') else 'Other-Attack')
X = df.drop(columns=[c for c in DROP if c in df]+['Label','cls']).select_dtypes(
    include=[np.number]).replace([np.inf,-np.inf], np.nan).fillna(0).values.astype('float32')
le = LabelEncoder(); y = le.fit_transform(df['cls']); N = len(le.classes_)

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE)
sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
print(f'테스트 샘플: {Xte.shape[0]:,}개')

# ---- 모델 학습 ----
print('모델 학습...')
dt = DecisionTreeClassifier(max_depth=20, random_state=0).fit(Xtr, ytr)
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0).fit(Xtr, ytr)
mlp = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=100, random_state=0).fit(Xtr, ytr)
gate = LogisticRegression(max_iter=300).fit(Xtr, ytr)
experts = [MLPClassifier(hidden_layer_sizes=(32,), max_iter=100, random_state=0)
           .fit(Xtr, (ytr == c).astype(int)) for c in range(N)]

# ---- MoE 추론 (top-k: 선택된 전문가만 실제로 실행) ----
def moe_infer(Xb, k):
    g = gate.predict_proba(Xb)                          # 게이팅
    idx = np.argsort(g, axis=1)[:, ::-1][:, :k]         # top-k 선택
    # 선택된 전문가만 실행 (조건부 계산의 핵심)
    scores = np.zeros((len(Xb), N))
    for c in range(N):
        rows = np.where((idx == c).any(axis=1))[0]      # 이 전문가가 선택된 샘플만
        if len(rows) > 0:
            scores[rows, c] = experts[c].predict_proba(Xb[rows])[:, 1]
    mask = np.zeros_like(g); np.put_along_axis(mask, idx, 1, axis=1)
    gk = g*mask; gk = gk/gk.sum(1, keepdims=True)
    return (gk*scores).argmax(1)

# ---- 시간 측정 헬퍼 ----
def timeit(fn, Xb):
    ts = []
    for _ in range(N_REPEAT):
        t0 = time.perf_counter(); fn(Xb); ts.append(time.perf_counter()-t0)
    return np.median(ts)

n = Xte.shape[0]
print('\n추론 시간 측정...')
results = {}
results['Decision Tree'] = timeit(lambda X: dt.predict(X), Xte)
results['Random Forest'] = timeit(lambda X: rf.predict(X), Xte)
results['Single MLP'] = timeit(lambda X: mlp.predict(X), Xte)
for k in range(1, N+1):
    results[f'MoE (top-{k})'] = timeit(lambda X, kk=k: moe_infer(X, kk), Xte)

# ---- 표 ----
print('\n=== 추론 시간 (테스트셋 %s개 기준) ===' % f'{n:,}')
rows = []
for name, t in results.items():
    per_sample_us = t / n * 1e6
    throughput = n / t
    print(f'{name:16s}: {t*1000:7.2f} ms  |  {per_sample_us:6.2f} us/샘플  |  {throughput:>12,.0f} 샘플/초')
    rows.append([name, f'{t*1000:.2f}', f'{per_sample_us:.2f}', f'{throughput:,.0f}'])
pd.DataFrame(rows, columns=['Model','Total (ms)','Per-sample (us)','Throughput (/s)']).to_csv('table_inference.csv', index=False)

# ---- 그래프: top-k별 추론 시간 ----
try:
    import matplotlib.font_manager as fm, glob
    for fp in glob.glob('/usr/share/fonts/**/NanumGothic*.ttf', recursive=True):
        prop = fm.FontProperties(fname=fp); plt.rcParams['font.family'] = prop.get_name()
        fm.fontManager.addfont(fp); break
except Exception: pass
plt.rcParams['axes.unicode_minus'] = False

ks = list(range(1, N+1))
moe_times = [results[f'MoE (top-{k})']*1000 for k in ks]
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(ks, moe_times, 'o-', color='#2c6fbb', linewidth=2, markersize=7, label='MoE (top-k)')
ax.axhline(results['Random Forest']*1000, ls='--', color='#e07b39', label='Random Forest')
ax.axhline(results['Decision Tree']*1000, ls=':', color='#3a9679', label='Decision Tree')
ax.set_xlabel('k (선택 전문가 수)'); ax.set_ylabel('추론 시간 (ms)')
ax.set_xticks(ks); ax.set_title('Top-k별 추론 시간 비교'); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('fig_inference_time.png', dpi=200, bbox_inches='tight')
print('\n저장 완료: table_inference.csv, fig_inference_time.png')
