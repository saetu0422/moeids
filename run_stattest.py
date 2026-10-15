#!/usr/bin/env python3
"""
run_stattest.py  --  Paired significance test: Adaptive vs Full-execution MoE.
  리뷰어 대응: "statistically indistinguishable" 주장을 뒷받침하는 통계 검정.
  H0: F1_adaptive == F1_full  (seed별 paired)
  - paired t-test + Wilcoxon signed-rank (비모수)
  - 두 지표를 함께 보고하고, 효과크기(평균차)도 제시.

실행:
  python3 run_stattest.py                          # InSDN (uploads 폴더)
  python3 run_stattest.py --cic ~/Downloads/cicddos2019
  python3 run_stattest.py --data /mnt/user-data/uploads --seeds 5
"""
import argparse, glob, os, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument('--cic', default=None)
ap.add_argument('--data', default='/mnt/user-data/uploads')
ap.add_argument('--seeds', type=int, default=5)
args = ap.parse_args()
SEEDS = list(range(args.seeds))

DROP = ['Flow ID','Source IP','Src IP','Destination IP','Dst IP','Timestamp',
        'Source Port','Src Port','Destination Port','Dst Port','SimillarHTTP','Unnamed: 0']

if args.cic:
    print('Loading CIC-DDoS2019...')
    dfs=[pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(args.cic,'*.parquet')))]
    for d in dfs: d.columns=d.columns.str.strip()
    df=pd.concat(dfs,ignore_index=True); df['Label']=df['Label'].astype(str).str.strip()
    df['cls']=df['Label'].apply(lambda l:'Normal' if l=='Benign' else
        ('SYN' if l=='Syn' else ('Reflection' if (l.startswith('DrDoS_') or l=='TFTP') else 'Volumetric')))
    LOADCOL='Flow Packets/s'; NAME='CIC-DDoS2019'
else:
    print('Loading InSDN...')
    def load(f):
        d=pd.read_csv(os.path.join(args.data,f)); d['Label']=d['Label'].astype(str).str.strip(); return d
    df=pd.concat([load('Normal_data.csv'),load('OVS.csv'),load('metasploitable-2.csv')],ignore_index=True)
    df['cls']=df['Label'].apply(lambda l: l if l in('Normal','DDoS','DoS','Probe') else 'Other-Attack')
    LOADCOL='Flow Pkts/s'; NAME='InSDN'
df.columns=df.columns.str.strip()

if LOADCOL not in df.columns:
    cand=[c for c in df.columns if 'Pkts/s' in c or 'Packets/s' in c]
    LOADCOL=cand[0] if cand else None
load_all = df[LOADCOL].replace([np.inf,-np.inf],np.nan).fillna(0).values.astype('float32') if LOADCOL else np.zeros(len(df))

X = df.drop(columns=[c for c in DROP if c in df.columns]+['Label','cls'],errors='ignore')
X = X.select_dtypes(include=[np.number]).replace([np.inf,-np.inf],np.nan).fillna(0).values.astype('float32')
le=LabelEncoder(); y=le.fit_transform(df['cls']); N=len(le.classes_)
print(f'data={X.shape}, classes={list(le.classes_)}, load_col={LOADCOL}')

def train(Xtr,ytr):
    g=LogisticRegression(max_iter=300).fit(Xtr,ytr)
    ex=[MLPClassifier(hidden_layer_sizes=(32,),max_iter=100,random_state=0).fit(Xtr,(ytr==c).astype(int)) for c in range(N)]
    return g,ex

def predict_with_k(gp,ep,kvec):
    n=len(gp); preds=np.zeros(n,dtype=int); order=np.argsort(gp,axis=1)[:,::-1]
    for k in np.unique(kvec):
        rows=np.where(kvec==k)[0]
        if len(rows)==0: continue
        idx=order[rows,:int(k)]
        mask=np.zeros((len(rows),N)); np.put_along_axis(mask,idx,1,axis=1)
        gk=gp[rows]*mask; gk=gk/gk.sum(1,keepdims=True)
        preds[rows]=(gk*ep[rows]).argmax(1)
    return preds

f1_adpt=[]; f1_full=[]
for s in SEEDS:
    Xtr,Xte,ytr,yte,_,lte=train_test_split(X,y,load_all,test_size=0.3,stratify=y,random_state=s)
    sc=StandardScaler(); Xtr=sc.fit_transform(Xtr); Xte=sc.transform(Xte)
    g,ex=train(Xtr,ytr)
    gp=g.predict_proba(Xte); ep=np.stack([e.predict_proba(Xte)[:,1] for e in ex],axis=1)
    # full
    pf=predict_with_k(gp,ep,np.full(len(Xte),N))
    f1_full.append(f1_score(yte,pf,average='macro'))
    # adaptive: high load -> k=2, low load -> k=3
    thr=np.quantile(lte,0.5); kvec=np.where(lte>=thr,min(2,N-1),min(3,N))
    pa=predict_with_k(gp,ep,kvec)
    f1_adpt.append(f1_score(yte,pa,average='macro'))
    print(f'  seed {s}: adaptive={f1_adpt[-1]:.5f}  full={f1_full[-1]:.5f}  diff={f1_adpt[-1]-f1_full[-1]:+.5f}')

a=np.array(f1_adpt); b=np.array(f1_full); d=a-b
print(f'\n=== {NAME} :  Adaptive vs Full  (paired, n={len(SEEDS)} seeds) ===')
print(f'  Adaptive F1 : {a.mean():.5f} ± {a.std(ddof=1):.5f}')
print(f'  Full     F1 : {b.mean():.5f} ± {b.std(ddof=1):.5f}')
print(f'  mean diff   : {d.mean():+.5f}  (adaptive - full)')
t,pt=stats.ttest_rel(a,b)
print(f'  paired t-test      : t={t:.3f},  p={pt:.4f}')
try:
    w,pw=stats.wilcoxon(a,b)
    print(f'  Wilcoxon signed-rank: W={w:.3f}, p={pw:.4f}')
except Exception as e:
    print(f'  Wilcoxon: n/a ({e})')
print(f'\n  => {"FAIL to reject H0 (indistinguishable)" if pt>0.05 else "reject H0 (different)"} at alpha=0.05')
