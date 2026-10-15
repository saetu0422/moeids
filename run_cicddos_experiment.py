#!/usr/bin/env python3
"""
run_cicddos_experiment.py  —  CIC-DDoS2019 다양한 데이터셋 검증 실험
  InSDN에서 확인한 top-k 포화 현상이 다른 데이터셋에서도 재현되는지 검증한다.
  - seed 5회 반복으로 F1-macro 평균±표준편차
  - top-k ablation (조건부 계산 효율)
  - 라우팅 정확도

실행:
  cd ~/Downloads/moeids
  python3 run_cicddos_experiment.py ~/Downloads/cicddos2019

결과물:
  table_cicddos_seed.csv       (모델별 평균±표준편차)
  table_cicddos_topk.csv       (top-k별 F1)
  fig_cicddos_topk.png         (top-k 곡선)
  table_cicddos_routing.csv    (라우팅 정확도)
"""
import sys, os, glob, warnings, numpy as np, pandas as pd
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
from sklearn.metrics import f1_score, confusion_matrix

folder = sys.argv[1] if len(sys.argv) > 1 else '.'
SEEDS = [0, 1, 2, 3, 4]

# ---- 공격 유형 -> 전문가 매핑 ----
def to_expert(label):
    l = label.strip()
    if l == 'Benign': return 'Normal'
    if l == 'Syn': return 'SYN'
    # 반사/증폭 공격 (DrDoS_ 접두사 + TFTP)
    if l.startswith('DrDoS_') or l == 'TFTP': return 'Reflection'
    # 나머지 직접 볼류메트릭
    return 'Volumetric'

# ---- 데이터 로딩 ----
print('데이터 로딩...')
files = sorted(glob.glob(os.path.join(folder, '*.parquet')))
dfs = []
for f in files:
    d = pd.read_parquet(f)
    d.columns = d.columns.str.strip()
    dfs.append(d)
df = pd.concat(dfs, ignore_index=True)
df['Label'] = df['Label'].astype(str).str.strip()
df['cls'] = df['Label'].apply(to_expert)
print('전체:', df.shape)
print(df['cls'].value_counts())

# ---- 특징 준비 (식별자·비수치 제외) ----
DROP = ['Flow ID','Source IP','Src IP','Destination IP','Dst IP','Timestamp',
        'Source Port','Src Port','Destination Port','Dst Port','SimillarHTTP','Unnamed: 0']
X = df.drop(columns=[c for c in DROP if c in df.columns] + ['Label','cls'], errors='ignore')
X = X.select_dtypes(include=[np.number]).replace([np.inf,-np.inf], np.nan).fillna(0).values.astype('float32')
le = LabelEncoder(); y = le.fit_transform(df['cls']); CLASSES = list(le.classes_); N = len(CLASSES)
print(f'특징 수: {X.shape[1]}, 클래스: {CLASSES}')

def train_moe(Xtr, ytr):
    gate = LogisticRegression(max_iter=300).fit(Xtr, ytr)
    experts = [MLPClassifier(hidden_layer_sizes=(32,), max_iter=100, random_state=0)
               .fit(Xtr, (ytr == c).astype(int)) for c in range(N)]
    return gate, experts

def moe_predict(gate, experts, Xte, k):
    g = gate.predict_proba(Xte)
    ep = np.stack([e.predict_proba(Xte)[:,1] for e in experts], axis=1)
    idx = np.argsort(g, axis=1)[:, ::-1][:, :k]
    mask = np.zeros_like(g); np.put_along_axis(mask, idx, 1, axis=1)
    gk = g*mask; gk = gk/gk.sum(1, keepdims=True)
    return (gk*ep).argmax(1), g.argmax(1)

# ---- seed 반복 + top-k + 라우팅 ----
metrics = {'Decision Tree':[], 'Random Forest':[]}
for k in range(1, N+1): metrics[f'MoE (top-{k})'] = []
route_cms = []
for s in SEEDS:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=s)
    sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
    metrics['Decision Tree'].append(f1_score(yte,
        DecisionTreeClassifier(max_depth=20, random_state=s).fit(Xtr,ytr).predict(Xte), average='macro'))
    metrics['Random Forest'].append(f1_score(yte,
        RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=s).fit(Xtr,ytr).predict(Xte), average='macro'))
    gate, experts = train_moe(Xtr, ytr)
    top1_saved = None
    for k in range(1, N+1):
        pred, top1 = moe_predict(gate, experts, Xte, k)
        metrics[f'MoE (top-{k})'].append(f1_score(yte, pred, average='macro'))
        if k == 2: top1_saved = top1
    route_cms.append(confusion_matrix(yte, top1_saved, labels=range(N)))
    print(f'  seed {s} 완료')

# ---- 결과 출력 ----
print('\n=== seed 5회 F1-macro (평균 ± 표준편차) ===')
srows = []
for name, v in metrics.items():
    print(f'{name:18s}: {np.mean(v):.4f} ± {np.std(v):.4f}')
    srows.append([name, f'{np.mean(v):.4f}', f'{np.std(v):.4f}'])
pd.DataFrame(srows, columns=['Model','Mean','Std']).to_csv('table_cicddos_seed.csv', index=False)

print('\n=== top-k별 F1-macro (연산량) ===')
ks = list(range(1, N+1)); trows = []
for k in ks:
    m = np.mean(metrics[f'MoE (top-{k})'])
    frac = int(k/N*100)
    print(f'  top-{k}: {m:.4f}  (전문가 {k}/{N}, 연산 {frac}%)')
    trows.append([f'top-{k}', f'{k}/{N}', f'{frac}%', f'{m:.4f}'])
pd.DataFrame(trows, columns=['Routing','Experts','Compute','F1-macro']).to_csv('table_cicddos_topk.csv', index=False)

# ---- 라우팅 정확도 ----
route_cm = np.sum(route_cms, axis=0)
route_acc = np.diag(route_cm).sum() / route_cm.sum()
print(f'\n=== 라우팅 정확도: {route_acc:.4f} ===')
per_class = np.diag(route_cm) / route_cm.sum(1)
rrows = []
for c, acc in zip(CLASSES, per_class):
    print(f'  {c:12s}: {acc:.4f}')
    rrows.append([c, f'{acc:.4f}'])
rrows.append(['전체', f'{route_acc:.4f}'])
pd.DataFrame(rrows, columns=['Class','Routing accuracy']).to_csv('table_cicddos_routing.csv', index=False)

# ---- top-k 그래프 ----
try:
    import matplotlib.font_manager as fm
    fp = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
    if os.path.exists(fp):
        prop = fm.FontProperties(fname=fp); fm.fontManager.addfont(fp)
        plt.rcParams['font.family'] = prop.get_name()
except Exception: pass
plt.rcParams['axes.unicode_minus'] = False
moe_f1 = [np.mean(metrics[f'MoE (top-{k})']) for k in ks]
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(ks, moe_f1, 'o-', color='#2c6fbb', linewidth=2, markersize=8)
for k, v in zip(ks, moe_f1):
    ax.annotate(f'{v:.3f}', (k, v), textcoords='offset points', xytext=(0,9), fontsize=9, ha='center')
ax.set_xlabel('k (선택 전문가 수)'); ax.set_ylabel('F1-macro')
ax.set_xticks(ks); ax.set_title('CIC-DDoS2019: Top-k 라우팅에 따른 F1'); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('fig_cicddos_topk.png', dpi=200, bbox_inches='tight')
print('\n저장 완료: table_cicddos_seed.csv, table_cicddos_topk.csv, table_cicddos_routing.csv, fig_cicddos_topk.png')
