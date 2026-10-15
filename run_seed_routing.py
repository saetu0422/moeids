#!/usr/bin/env python3
"""
run_seed_routing.py  —  1단계 추가 실험 (직접 실행)
  - seed 5개 반복으로 F1-macro 평균±표준편차 산출 (통계적 유의성)
  - 라우팅 정확도 측정 (게이팅이 실제 공격 유형으로 보내는가)
  - 논문용 그림/표 자동 생성

실행:
  cd ~/Downloads/moeids
  python3 run_seed_routing.py           # 전체 데이터, 몇 분 소요
  python3 run_seed_routing.py --fast    # 40% 샘플, 빠른 확인

결과물:
  table_seed.csv                  (모델별 평균±표준편차)
  fig_routing_confusion.png       (라우팅 정확도 혼동행렬)
  table_routing.csv               (클래스별 라우팅 정확도)
"""
import argparse, warnings, numpy as np, pandas as pd
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

ap = argparse.ArgumentParser()
ap.add_argument('--fast', action='store_true', help='40%% 샘플로 빠르게')
ap.add_argument('--data', default='.', help='CSV 폴더')
args = ap.parse_args()

SEEDS = [0, 1, 2, 3, 4]
DROP = ['Flow ID','Src IP','Dst IP','Timestamp','Src Port','Dst Port']

def load(fn):
    d = pd.read_csv(f'{args.data}/{fn}'); d['Label'] = d['Label'].astype(str).str.strip(); return d

print('데이터 로딩...')
df = pd.concat([load('Normal_data.csv'), load('OVS.csv'),
                load('metasploitable-2.csv')], ignore_index=True)
df['cls'] = df['Label'].apply(lambda l: l if l in ('Normal','DDoS','DoS','Probe') else 'Other-Attack')
X = df.drop(columns=[c for c in DROP if c in df]+['Label','cls']).select_dtypes(
    include=[np.number]).replace([np.inf,-np.inf], np.nan).fillna(0).values.astype('float32')
le = LabelEncoder(); y = le.fit_transform(df['cls']); CLASSES = list(le.classes_); N = len(CLASSES)

if args.fast:
    X, _, y, _ = train_test_split(X, y, train_size=0.4, stratify=y, random_state=99)
print(f'데이터: {X.shape}, 클래스: {CLASSES}')

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
    return (gk*ep).argmax(1), g.argmax(1)   # 예측, 게이팅 top-1(라우팅 대상)

# ---- seed 반복 ----
metrics = {'Decision Tree':[], 'Random Forest':[], 'MoE-IDS (top-2)':[], 'MoE-IDS (top-3)':[]}
route_cms = []
for s in SEEDS:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=s)
    sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
    metrics['Decision Tree'].append(f1_score(yte,
        DecisionTreeClassifier(max_depth=20, random_state=s).fit(Xtr,ytr).predict(Xte), average='macro'))
    metrics['Random Forest'].append(f1_score(yte,
        RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=s).fit(Xtr,ytr).predict(Xte), average='macro'))
    gate, experts = train_moe(Xtr, ytr)
    p2,_ = moe_predict(gate, experts, Xte, 2)
    p3, top1 = moe_predict(gate, experts, Xte, 3)
    metrics['MoE-IDS (top-2)'].append(f1_score(yte, p2, average='macro'))
    metrics['MoE-IDS (top-3)'].append(f1_score(yte, p3, average='macro'))
    route_cms.append(confusion_matrix(yte, top1, labels=range(N)))
    print(f'  seed {s} 완료')

# ---- 표: 평균±표준편차 ----
print('\n=== seed 5회 F1-macro (평균 ± 표준편차) ===')
rows = []
for k, v in metrics.items():
    m, sd = np.mean(v), np.std(v)
    print(f'{k:20s}: {m:.4f} ± {sd:.4f}')
    rows.append([k, f'{m:.4f}', f'{sd:.4f}', f'{m:.4f} ± {sd:.4f}'])
pd.DataFrame(rows, columns=['Model','Mean','Std','F1-macro (mean±std)']).to_csv('table_seed.csv', index=False)

# ---- 라우팅 정확도 ----
route_cm = np.sum(route_cms, axis=0)
route_acc = np.diag(route_cm).sum() / route_cm.sum()
print(f'\n=== 라우팅 정확도 (게이팅 top-1 = 실제 클래스): {route_acc:.4f} ===')
per_class = np.diag(route_cm) / route_cm.sum(1)
for c, acc in zip(CLASSES, per_class):
    print(f'  {c:14s}: {acc:.4f}')
pd.DataFrame({'Class': CLASSES, 'Routing accuracy': per_class.round(4)}).to_csv('table_routing.csv', index=False)

# ---- 라우팅 혼동행렬 그림 ----
try:
    import matplotlib.font_manager as fm, glob
    for fp in glob.glob('/usr/share/fonts/**/NanumGothic*.ttf', recursive=True):
        fm.fontManager.addfont(fp); plt.rcParams['font.family']='NanumGothic'; break
except Exception: pass
plt.rcParams['axes.unicode_minus'] = False
cmn = route_cm.astype(float) / route_cm.sum(1, keepdims=True) * 100
fig, ax = plt.subplots(figsize=(6.2, 5.2))
im = ax.imshow(cmn, cmap='Greens', vmin=0, vmax=100)
ax.set_xticks(range(N)); ax.set_yticks(range(N))
ax.set_xticklabels(CLASSES, rotation=40, ha='right'); ax.set_yticklabels(CLASSES)
for i in range(N):
    for j in range(N):
        ax.text(j, i, f'{cmn[i,j]:.1f}', ha='center', va='center', fontsize=9,
                color='white' if cmn[i,j] > 50 else 'black')
ax.set_xlabel('선택된 전문가 (게이팅 top-1)'); ax.set_ylabel('실제 클래스')
ax.set_title(f'라우팅 정확도 혼동행렬 (전체 {route_acc*100:.1f}%)')
plt.colorbar(im, fraction=0.046, pad=0.04, label='%')
plt.tight_layout(); plt.savefig('fig_routing_confusion.png', dpi=200, bbox_inches='tight')
print('\n저장 완료: table_seed.csv, table_routing.csv, fig_routing_confusion.png')
