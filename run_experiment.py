#!/usr/bin/env python3
"""
run_experiment.py  —  InSDN 기반 MoE-IDS 전체 실험 (노트북과 동일)
실행:
  cd ~/Downloads/moeids
  pip install pandas numpy scikit-learn matplotlib   # 최초 1회
  python3 run_experiment.py

- DATA_DIR 아래에 Normal_data.csv, OVS.csv, metasploitable-2.csv 필요
- 결과: table_multiclass.csv, table_binary.csv, fig_*.png 가 현재 폴더에 생성
- 빠른 테스트는 아래 SAMPLE_FRAC 을 0.2 등으로. 최종 결과는 1.0
"""
import matplotlib
matplotlib.use("Agg")   # 창 없이 그림 파일로 저장 (서버/헤드리스 안전)

# ============ 0. 설정 ============
DATA_DIR = '/home/user/Downloads/moeids'          # <-- CSV 3개가 있는 폴더 (예: '/home/user/Downloads/moeids')
SAMPLE_FRAC = 1.0       # 빠른 테스트는 0.2 등으로. 최종 결과는 1.0
RANDOM_STATE = 42

import warnings, time, os
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             average_precision_score, confusion_matrix,
                             classification_report)
from sklearn.preprocessing import label_binarize
np.random.seed(RANDOM_STATE)
print('설정 완료')

DROP = ['Flow ID','Src IP','Dst IP','Timestamp','Src Port','Dst Port']
def load(fn):
    df = pd.read_csv(os.path.join(DATA_DIR, fn))
    df['Label'] = df['Label'].astype(str).str.strip()   # 'DDoS ' 같은 공백 제거
    return df

df = pd.concat([load('Normal_data.csv'), load('OVS.csv'),
                load('metasploitable-2.csv')], ignore_index=True)

def to_expert(l):
    return l if l in ('Normal','DDoS','DoS','Probe') else 'Other-Attack'
df['cls'] = df['Label'].apply(to_expert)

if SAMPLE_FRAC < 1.0:
    df = df.groupby('cls', group_keys=False)[df.columns].sample(
        frac=SAMPLE_FRAC, random_state=RANDOM_STATE).reset_index(drop=True)

print('전체:', df.shape)
print(df['cls'].value_counts())
print('\n이진(정상 vs 공격):')
print(df['cls'].eq('Normal').map({True:'Normal',False:'Attack'}).value_counts())

# 특징 행렬 X, 라벨 준비
X = df.drop(columns=[c for c in DROP if c in df.columns] + ['Label','cls'])
X = X.select_dtypes(include=[np.number]).replace([np.inf,-np.inf], np.nan).fillna(0)
FEATURES = X.columns.tolist()
print('특징 수:', len(FEATURES))

# 다중 분류 라벨
le = LabelEncoder(); y_multi = le.fit_transform(df['cls'])
CLASSES = list(le.classes_); N_CLS = len(CLASSES)
# 이진 라벨 (1=공격)
y_bin = (df['cls'] != 'Normal').astype(int).values
print('클래스:', CLASSES)

# 분할 + 표준화
Xtr, Xte, ytr_m, yte_m, ytr_b, yte_b = train_test_split(
    X.values, y_multi, y_bin, test_size=0.3, stratify=y_multi, random_state=RANDOM_STATE)
scaler = StandardScaler(); Xtr = scaler.fit_transform(Xtr); Xte = scaler.transform(Xte)
print('train', Xtr.shape, ' test', Xte.shape)

results = {}   # 이름 -> dict(지표)

def evaluate(name, y_true, y_pred, store=results):
    store[name] = {
        'Accuracy': (y_pred == y_true).mean(),
        'F1-macro': f1_score(y_true, y_pred, average='macro'),
        'Precision-macro': precision_score(y_true, y_pred, average='macro'),
        'Recall-macro': recall_score(y_true, y_pred, average='macro'),
    }
    print(f'{name:22s} acc={store[name]["Accuracy"]:.4f}  F1={store[name]["F1-macro"]:.4f}')
    return store[name]

# --- 규칙 기반: Flow Pkts/s 가 정상 분위수를 넘으면 공격(가장 흔한 공격 클래스로) ---
pps_idx = FEATURES.index('Flow Pkts/s') if 'Flow Pkts/s' in FEATURES else 0
# train 의 정상 트래픽 95 분위수를 임계치로
normal_pps = Xtr[ytr_m == CLASSES.index('Normal'), pps_idx]
thr = np.quantile(normal_pps, 0.95)
major_attack = np.bincount(ytr_m[ytr_m != CLASSES.index('Normal')]).argmax()
rule_pred = np.where(Xte[:, pps_idx] > thr, major_attack, CLASSES.index('Normal'))
evaluate('Rule (pps threshold)', yte_m, rule_pred)

# --- DT / RF / MLP ---
t=time.time(); dt=DecisionTreeClassifier(max_depth=20,random_state=0).fit(Xtr,ytr_m)
evaluate('Decision Tree', yte_m, dt.predict(Xte)); print('  (%.1fs)'%(time.time()-t))
t=time.time(); rf=RandomForestClassifier(n_estimators=100,n_jobs=-1,random_state=0).fit(Xtr,ytr_m)
evaluate('Random Forest', yte_m, rf.predict(Xte)); print('  (%.1fs)'%(time.time()-t))
t=time.time(); mlp=MLPClassifier(hidden_layer_sizes=(64,32),max_iter=100,random_state=0).fit(Xtr,ytr_m)
evaluate('Single MLP (DL)', yte_m, mlp.predict(Xte)); print('  (%.1fs)'%(time.time()-t))

class MoEIDS:
    """게이팅 + one-vs-rest 전문가 + top-k 라우팅"""
    def __init__(self, n_cls, k=2):
        self.n_cls, self.k = n_cls, k
    def fit(self, X, y):
        self.gate = LogisticRegression(max_iter=300).fit(X, y)     # 라우터 g
        self.experts = []
        for c in range(self.n_cls):                                 # 전문가 f_i
            e = MLPClassifier(hidden_layer_sizes=(32,), max_iter=100, random_state=0)
            e.fit(X, (y == c).astype(int)); self.experts.append(e)
        return self
    def _combine(self, X, k):
        g = self.gate.predict_proba(X)                              # (n, N)
        ep = np.stack([e.predict_proba(X)[:,1] for e in self.experts], axis=1)
        idx = np.argsort(g, axis=1)[:, ::-1][:, :k]                 # top-k 전문가
        mask = np.zeros_like(g); np.put_along_axis(mask, idx, 1, axis=1)
        gk = g*mask; gk = gk/gk.sum(1, keepdims=True)               # 재정규화
        return gk*ep, g                                            # y = sum g_i f_i
    def predict(self, X, k=None):
        score,_ = self._combine(X, k or self.k); return score.argmax(1)
    def routing_dist(self, X, k=None):
        _,g = self._combine(X, k or self.k)
        idx = np.argsort(g,axis=1)[:,::-1][:,:(k or self.k)]
        return np.bincount(idx.flatten(), minlength=self.n_cls)

t=time.time(); moe = MoEIDS(N_CLS, k=2).fit(Xtr, ytr_m)
print('MoE 학습 %.1fs'%(time.time()-t))
evaluate('MoE-IDS (top-2)', yte_m, moe.predict(Xte, k=2))

print('=== top-k 변화 ===')
abl = {}
for k in range(1, N_CLS+1):
    p = moe.predict(Xte, k=k)
    abl[k] = f1_score(yte_m, p, average='macro')
    print(f'  top-{k}: F1-macro {abl[k]:.4f}, acc {(p==yte_m).mean():.4f}')

print('\n=== 전문가 선택 분포 (expert collapse 확인) ===')
dist = moe.routing_dist(Xte, k=2)
for c, n in zip(CLASSES, dist):
    print(f'  {c:14s}: {n:6d}  ({100*n/dist.sum():.1f}%)')
print('한 전문가에 심하게 쏠리면 collapse. 고르게 퍼지면 정상 분업.')

# top-k F1 그래프
fig, ax = plt.subplots(1,2, figsize=(12,4))
ks=list(abl.keys()); ax[0].plot(ks,[abl[k] for k in ks],'o-')
ax[0].set_xlabel('k (선택 전문가 수)'); ax[0].set_ylabel('F1-macro')
ax[0].set_title('Ablation: top-k 라우팅'); ax[0].grid(alpha=.3)
ax[1].bar(CLASSES, dist); ax[1].set_title('전문가 선택 분포 (top-2)')
ax[1].set_ylabel('선택 횟수'); plt.setp(ax[1].get_xticklabels(),rotation=30,ha='right')
plt.tight_layout(); plt.savefig('fig_ablation.png',dpi=150,bbox_inches='tight'); plt.show()
print('저장: fig_ablation.png')

bin_res = {}
def eval_bin(name, model):
    model.fit(Xtr, ytr_b)
    proba = model.predict_proba(Xte)[:,1]; pred = (proba>0.5).astype(int)
    bin_res[name] = {'F1': f1_score(yte_b,pred),
                     'PR-AUC': average_precision_score(yte_b, proba),
                     'Recall': recall_score(yte_b,pred),
                     'Precision': precision_score(yte_b,pred)}
    print(f'{name:16s} F1={bin_res[name]["F1"]:.4f}  PR-AUC={bin_res[name]["PR-AUC"]:.4f}')

eval_bin('DT', DecisionTreeClassifier(max_depth=20,random_state=0))
eval_bin('RF', RandomForestClassifier(n_estimators=100,n_jobs=-1,random_state=0))
eval_bin('MLP', MLPClassifier(hidden_layer_sizes=(64,32),max_iter=100,random_state=0))

# 다중 분류 결과 표
res_df = pd.DataFrame(results).T.round(4)
print('=== 표: 다중 분류 성능 ==='); print(res_df)
res_df.to_csv('table_multiclass.csv')

# 이진 결과 표
bin_df = pd.DataFrame(bin_res).T.round(4)
print('\n=== 표: 이진 분류 성능 ==='); print(bin_df)
bin_df.to_csv('table_binary.csv')

# MoE 혼동행렬
cm = confusion_matrix(yte_m, moe.predict(Xte,k=2))
fig,ax=plt.subplots(figsize=(6,5)); im=ax.imshow(cm,cmap='Blues')
ax.set_xticks(range(N_CLS)); ax.set_yticks(range(N_CLS))
ax.set_xticklabels(CLASSES,rotation=45,ha='right'); ax.set_yticklabels(CLASSES)
for i in range(N_CLS):
    for j in range(N_CLS):
        ax.text(j,i,cm[i,j],ha='center',va='center',
                color='white' if cm[i,j]>cm.max()/2 else 'black',fontsize=8)
ax.set_xlabel('예측'); ax.set_ylabel('실제'); ax.set_title('MoE-IDS 혼동행렬 (top-2)')
plt.colorbar(im); plt.tight_layout()
plt.savefig('fig_confusion.png',dpi=150,bbox_inches='tight'); plt.show()

# 비교 막대그래프
fig,ax=plt.subplots(figsize=(9,4))
res_df['F1-macro'].plot(kind='bar',ax=ax,color='steelblue')
ax.set_ylabel('F1-macro'); ax.set_title('모델별 다중 분류 F1 비교')
ax.set_ylim(0,1); plt.setp(ax.get_xticklabels(),rotation=30,ha='right')
plt.tight_layout(); plt.savefig('fig_compare.png',dpi=150,bbox_inches='tight'); plt.show()
print('\n저장 완료: table_multiclass.csv, table_binary.csv, fig_*.png')

for k in range(1, N_CLS+1):
    frac = k / N_CLS
    print(f'top-{k}: 전문가 {k}/{N_CLS} 실행 = 전체 대비 {frac*100:.0f}% 연산, F1 {abl[k]:.4f}')
print('\n-> 적은 k로도 F1 유지되면, 정확도 손실 없이 연산 절감 = 실시간성 이점')
