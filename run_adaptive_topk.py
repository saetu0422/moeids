#!/usr/bin/env python3
"""
run_adaptive_topk.py  --  Load-Adaptive Top-k MoE for SDN DDoS Detection
  논문용 정량 평가 (trace-driven, 재현 가능).

  핵심 기여: 네트워크 부하(pps)를 관측해 탐지기의 top-k를 동적 조절한다.
    - 부하 높음 -> k 낮춤 (탐지 연산 절감, 여유를 완화에 배분)
    - 부하 낮음 -> k 높임 (정밀 탐지)
  비교: 고정 k=1..N  vs  부하 적응형 동적 k
  지표: F1-macro(전체 + 부하 구간별), 평균 실행 전문가 수(연산량), 연산 절감률.

실행:
  python3 run_adaptive_topk.py                          # InSDN
  python3 run_adaptive_topk.py --cic ~/Downloads/cicddos2019

결과물:
  table_adaptive.csv        (고정 k vs 동적 k: F1, 평균 k, 절감률)
  table_adaptive_regime.csv (부하 저/고 구간별 F1)
  fig_adaptive_tradeoff.png (평균 연산량 대 F1 산점도)
"""
import argparse, glob, os, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

ap = argparse.ArgumentParser()
ap.add_argument('--cic', default=None)
ap.add_argument('--data', default='.')
ap.add_argument('--seeds', type=int, default=5)
args = ap.parse_args()
SEEDS = list(range(args.seeds))

DROP = ['Flow ID','Source IP','Src IP','Destination IP','Dst IP','Timestamp',
        'Source Port','Src Port','Destination Port','Dst Port','SimillarHTTP','Unnamed: 0']

# ---------- load ----------
if args.cic:
    print('Loading CIC-DDoS2019...')
    dfs=[pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(args.cic,'*.parquet')))]
    for d in dfs: d.columns=d.columns.str.strip()
    df=pd.concat(dfs,ignore_index=True); df['Label']=df['Label'].astype(str).str.strip()
    df['cls']=df['Label'].apply(lambda l:'Normal' if l=='Benign' else
        ('SYN' if l=='Syn' else ('Reflection' if (l.startswith('DrDoS_') or l=='TFTP') else 'Volumetric')))
    LOADCOL='Flow Packets/s'
else:
    print('Loading InSDN...')
    def load(f):
        d=pd.read_csv(os.path.join(args.data,f)); d['Label']=d['Label'].astype(str).str.strip(); return d
    df=pd.concat([load('Normal_data.csv'),load('OVS.csv'),load('metasploitable-2.csv')],ignore_index=True)
    df['cls']=df['Label'].apply(lambda l: l if l in('Normal','DDoS','DoS','Probe') else 'Other-Attack')
    LOADCOL='Flow Pkts/s'
df.columns=df.columns.str.strip()

# load signal (pps proxy); fall back to any Pkts/s col
if LOADCOL not in df.columns:
    cand=[c for c in df.columns if 'Pkts/s' in c or 'Packets/s' in c]
    LOADCOL=cand[0] if cand else None
load_all = df[LOADCOL].replace([np.inf,-np.inf],np.nan).fillna(0).values.astype('float32') if LOADCOL else np.zeros(len(df))

X = df.drop(columns=[c for c in DROP if c in df.columns]+['Label','cls'],errors='ignore')
X = X.select_dtypes(include=[np.number]).replace([np.inf,-np.inf],np.nan).fillna(0).values.astype('float32')
le=LabelEncoder(); y=le.fit_transform(df['cls']); CLASSES=list(le.classes_); N=len(CLASSES)
print(f'data={X.shape}, classes={CLASSES}, load_col={LOADCOL}')

def train(Xtr,ytr):
    g=LogisticRegression(max_iter=300).fit(Xtr,ytr)
    ex=[MLPClassifier(hidden_layer_sizes=(32,),max_iter=100,random_state=0).fit(Xtr,(ytr==c).astype(int)) for c in range(N)]
    return g,ex

def gate_expert_probs(g,ex,Xte):
    gp=g.predict_proba(Xte)
    ep=np.stack([e.predict_proba(Xte)[:,1] for e in ex],axis=1)
    return gp,ep

def predict_with_k(gp,ep,kvec):
    """kvec: per-sample k (int array). returns preds, and experts-executed per sample (=k)."""
    n=len(gp); preds=np.zeros(n,dtype=int)
    order=np.argsort(gp,axis=1)[:,::-1]
    for k in np.unique(kvec):
        rows=np.where(kvec==k)[0]
        if len(rows)==0: continue
        idx=order[rows,:k]
        mask=np.zeros((len(rows),N)); np.put_along_axis(mask,idx,1,axis=1)
        gk=gp[rows]*mask; gk=gk/gk.sum(1,keepdims=True)
        preds[rows]=(gk*ep[rows]).argmax(1)
    return preds, kvec.astype(float)

# adaptive policy: high load -> low k, low load -> high k
def adaptive_k(load_te, k_low=3, k_high=2, q=0.5):
    thr=np.quantile(load_te,q)
    return np.where(load_te>=thr,k_high,k_low)  # high load => k_high(=fewer)

rows_fixed={k:[] for k in range(1,N+1)}
rows_adpt=[]; rows_regime={'low':[], 'high':[]}
for s in SEEDS:
    Xtr,Xte,ytr,yte,_,lte=train_test_split(X,y,load_all,test_size=0.3,stratify=y,random_state=s)
    sc=StandardScaler(); Xtr=sc.fit_transform(Xtr); Xte=sc.transform(Xte)
    g,ex=train(Xtr,ytr); gp,ep=gate_expert_probs(g,ex,Xte)
    # fixed k baselines
    for k in range(1,N+1):
        pr,_=predict_with_k(gp,ep,np.full(len(Xte),k))
        rows_fixed[k].append((f1_score(yte,pr,average='macro'), float(k)))
    # adaptive
    kvec=adaptive_k(lte, k_low=min(3,N), k_high=max(1,min(2,N-1)), q=0.5)
    pr,kcost=predict_with_k(gp,ep,kvec)
    rows_adpt.append((f1_score(yte,pr,average='macro'), kcost.mean()))
    # per-regime F1 (low vs high load), using adaptive preds
    thr=np.quantile(lte,0.5)
    lo=np.where(lte<thr)[0]; hi=np.where(lte>=thr)[0]
    rows_regime['low'].append(f1_score(yte[lo],pr[lo],average='macro'))
    rows_regime['high'].append(f1_score(yte[hi],pr[hi],average='macro'))
    print(f'  seed {s} done')

def ms(a): return np.mean(a),np.std(a)

print('\n=== Fixed k vs Adaptive k (F1-macro, avg experts executed) ===')
out=[]
for k in range(1,N+1):
    f1s=[x[0] for x in rows_fixed[k]]; kc=[x[1] for x in rows_fixed[k]]
    m,sd=ms(f1s)
    print(f'fixed k={k:<2d}: F1={m:.4f}±{sd:.4f}, experts={np.mean(kc):.2f}')
    out.append([f'fixed k={k}', f'{m:.4f}', f'{sd:.4f}', f'{np.mean(kc):.2f}'])
f1s=[x[0] for x in rows_adpt]; kc=[x[1] for x in rows_adpt]
m,sd=ms(f1s); avgk=np.mean(kc)
full_k=N
save=100*(full_k-avgk)/full_k
print(f'adaptive : F1={m:.4f}±{sd:.4f}, experts={avgk:.2f} ({save:.0f}% fewer than full k={N})')
out.append(['adaptive (load-aware)', f'{m:.4f}', f'{sd:.4f}', f'{avgk:.2f}'])
pd.DataFrame(out,columns=['Method','F1_mean','F1_std','AvgExperts']).to_csv('table_adaptive.csv',index=False)

print('\n=== Per-load-regime F1 (adaptive) ===')
for r in ('low','high'):
    mm,ss=ms(rows_regime[r]); print(f'  {r:4s}-load: F1={mm:.4f}±{ss:.4f}')
pd.DataFrame([['low-load',f'{ms(rows_regime["low"])[0]:.4f}'],
              ['high-load',f'{ms(rows_regime["high"])[0]:.4f}']],
             columns=['Regime','F1_macro']).to_csv('table_adaptive_regime.csv',index=False)

# figure
try:
    import matplotlib.font_manager as fm
    fp='/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
    if os.path.exists(fp):
        pr_=fm.FontProperties(fname=fp); fm.fontManager.addfont(fp); plt.rcParams['font.family']=pr_.get_name()
except Exception: pass
plt.rcParams['axes.unicode_minus']=False
fig,ax=plt.subplots(figsize=(7,4.5))
for k in range(1,N+1):
    f1s=[x[0] for x in rows_fixed[k]]; ax.scatter(k,np.mean(f1s),c='#8a8d91',s=90,zorder=3)
ax.plot(range(1,N+1),[np.mean([x[0] for x in rows_fixed[k]]) for k in range(1,N+1)],'--',color='#8a8d91',label='fixed k',zorder=2)
ax.scatter(avgk,np.mean([x[0] for x in rows_adpt]),c='#e07b39',marker='*',s=340,edgecolor='black',zorder=4,label='adaptive (load-aware)')
ax.set_xlabel('Average experts executed (compute)'); ax.set_ylabel('F1-macro')
ax.set_title('Load-Adaptive Top-k: compute vs detection'); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('fig_adaptive_tradeoff.png',dpi=200,bbox_inches='tight')
print('\nsaved: table_adaptive.csv, table_adaptive_regime.csv, fig_adaptive_tradeoff.png')
