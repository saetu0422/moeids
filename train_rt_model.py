#!/usr/bin/env python3
"""
train_rt_model.py  —  폐루프 시연용 MoE 모델 학습·저장
실행:
  cd ~/Downloads/moeids
  python3 train_rt_model.py
- Ryu 가 실시간으로 수집 가능한 흐름 특징만 사용해 학습한다.
  (전체 77개 특징이 아닌, 포트/플로우 통계로 재현 가능한 12개)
- 결과: moe_rt.joblib  (Ryu 앱이 이 파일을 불러 추론)

주의: 이 모델은 InSDN 으로 학습되지만, 시연에서는 Mininet 실시간 트래픽에
      적용된다. 두 환경의 특징 분포가 달라 완벽히 맞지는 않으므로,
      시연의 목적은 '최고 정확도'가 아니라 '탐지→룰 설치 폐루프의 동작'임을
      논문에 명시할 것.
"""
import warnings, os, numpy as np, pandas as pd, joblib
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
np.random.seed(42)

DATA_DIR = '.'
RT_FEATURES = ['Flow Duration','Tot Fwd Pkts','Tot Bwd Pkts','Flow Byts/s','Flow Pkts/s',
               'Fwd Pkts/s','Bwd Pkts/s','SYN Flag Cnt','ACK Flag Cnt','Pkt Len Mean',
               'Pkt Size Avg','Down/Up Ratio']

def load(fn):
    d = pd.read_csv(os.path.join(DATA_DIR, fn))
    d['Label'] = d['Label'].astype(str).str.strip(); return d

df = pd.concat([load('Normal_data.csv'), load('OVS.csv'),
                load('metasploitable-2.csv')], ignore_index=True)
df['cls'] = df['Label'].apply(lambda l: l if l in ('Normal','DDoS','DoS','Probe') else 'Other-Attack')

feats = [c for c in RT_FEATURES if c in df.columns]
X = df[feats].replace([np.inf,-np.inf], np.nan).fillna(0)
le = LabelEncoder(); y = le.fit_transform(df['cls']); CLASSES = list(le.classes_); N = len(CLASSES)

Xtr, Xte, ytr, yte = train_test_split(X.values, y, test_size=0.3, stratify=y, random_state=42)
sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)

gate = LogisticRegression(max_iter=300).fit(Xtr, ytr)
experts = [MLPClassifier(hidden_layer_sizes=(32,), max_iter=100, random_state=0)
           .fit(Xtr, (ytr == c).astype(int)) for c in range(N)]

def combine(Xb, k=2):
    g = gate.predict_proba(Xb)
    ep = np.stack([e.predict_proba(Xb)[:,1] for e in experts], axis=1)
    idx = np.argsort(g, axis=1)[:, ::-1][:, :k]
    mask = np.zeros_like(g); np.put_along_axis(mask, idx, 1, axis=1)
    gk = g*mask; gk = gk/gk.sum(1, keepdims=True); return gk*ep

pred = combine(Xte, 2).argmax(1)
print('실시간 특징 MoE top-2 F1-macro: %.4f, acc %.4f'
      % (f1_score(yte, pred, average='macro'), (pred == yte).mean()))
print('클래스:', CLASSES)

joblib.dump({'gate': gate, 'experts': experts, 'scaler': sc,
             'classes': CLASSES, 'features': feats, 'k': 2}, 'moe_rt.joblib')
print('저장 완료: moe_rt.joblib')
