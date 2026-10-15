#!/usr/bin/env python3
"""
run_regime_analysis.py  --  Load-subset x Fixed-k 분리 실험 (적응형 정책 검증)
  목적: high-load F1 저하가 (A) k 축소 때문인지 (B) 원래 어려운 트래픽인지 분리.
  방법: 테스트셋을 부하 중앙값으로 low/high로 나누고,
        각 subset에 고정 k=2,k=3,full 을 적용해 macro-F1 비교.
        클래스별 F1과 overall 정합성(sanity)도 출력.

실행:
  python3 run_regime_analysis.py                          # InSDN
  python3 run_regime_analysis.py --cic ~/Downloads/cicddos2019
결과물:
  table_regime_fixedk.csv   (low/high x k=2/k=3/full/adaptive, macro-F1)
  table_regime_perclass.csv (low/high 구간별 클래스별 F1)
"""
import argparse, glob, os, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

ap=argparse.ArgumentParser(); ap.add_argument('--cic',default=None); ap.add_argument('--data',default='.')
ap.add_argument('--seeds',type=int,default=5); args=ap.parse_args()
SEEDS=list(range(args.seeds))
DROP=['Flow ID','Source IP','Src IP','Destination IP','Dst IP','Timestamp',
      'Source Port','Src Port','Destination Port','Dst Port','SimillarHTTP','Unnamed: 0']

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
if LOADCOL not in df.columns:
    c=[x for x in df.columns if 'Pkts/s' in x or 'Packets/s' in x]; LOADCOL=c[0] if c else None
load_all=df[LOADCOL].replace([np.inf,-np.inf],np.nan).fillna(0).values.astype('float32') if LOADCOL else np.zeros(len(df))
X=df.drop(columns=[c for c in DROP if c in df.columns]+['Label','cls'],errors='ignore')
X=X.select_dtypes(include=[np.number]).replace([np.inf,-np.inf],np.nan).fillna(0).values.astype('float32')
le=LabelEncoder(); y=le.fit_transform(df['cls']); CLASSES=list(le.classes_); N=len(CLASSES)
print(f'data={X.shape}, N={N}, classes={CLASSES}')

def topk_pred(gp,ep,kvec):
    order=np.argsort(gp,axis=1)[:,::-1]; preds=np.zeros(len(gp),dtype=int)
    for k in np.unique(kvec):
        r=np.where(kvec==k)[0]; idx=order[r,:k]; mask=np.zeros((len(r),N)); np.put_along_axis(mask,idx,1,axis=1)
        gk=gp[r]*mask; gk=gk/gk.sum(1,keepdims=True); preds[r]=(gk*ep[r]).argmax(1)
    return preds

# 누적: [regime][policy] -> list of macro-F1 ;  perclass[regime][k] -> list of per-class F1 arrays
import collections
regime_f1=collections.defaultdict(lambda: collections.defaultdict(list))
perclass=collections.defaultdict(lambda: collections.defaultdict(list))
overall_adpt=[]
for s in SEEDS:
    Xtr,Xte,ytr,yte,_,lte=train_test_split(X,y,load_all,test_size=0.3,stratify=y,random_state=s)
    sc=StandardScaler(); Xtr=sc.fit_transform(Xtr); Xte=sc.transform(Xte)
    g=LogisticRegression(max_iter=300).fit(Xtr,ytr)
    ex=[MLPClassifier(hidden_layer_sizes=(32,),max_iter=100,random_state=0).fit(Xtr,(ytr==c).astype(int)) for c in range(N)]
    gp=g.predict_proba(Xte); ep=np.stack([e.predict_proba(Xte)[:,1] for e in ex],axis=1)
    thr=np.quantile(lte,0.5); lo=np.where(lte<thr)[0]; hi=np.where(lte>=thr)[0]
    for name,sub in (('low',lo),('high',hi)):
        for k in (2,3,N):
            pr=topk_pred(gp[sub],ep[sub],np.full(len(sub),k))
            regime_f1[name][f'k={k}' if k!=N else 'full'].append(f1_score(yte[sub],pr,average='macro'))
            if k in (2,3):  # per-class at the k the adaptive policy uses
                pass
        # adaptive uses: low->k=3, high->k=2
        kad=3 if name=='low' else 2
        pr=topk_pred(gp[sub],ep[sub],np.full(len(sub),kad))
        regime_f1[name]['adaptive'].append(f1_score(yte[sub],pr,average='macro'))
        pc=f1_score(yte[sub],pr,average=None,labels=range(N))
        perclass[name][kad].append(pc)
    # overall adaptive (sanity)
    kvec=np.where(lte>=thr,2,3); pr=topk_pred(gp,ep,kvec); overall_adpt.append(f1_score(yte,pr,average='macro'))
    print(f'  seed {s} done')

def ms(a): return float(np.mean(a)),float(np.std(a))
print('\n=== Load-subset x Fixed-k (macro-F1, 5 seeds) ===')
print(f'{"regime":6s} {"k=2":>14s} {"k=3":>14s} {"full":>14s} {"adaptive":>14s}')
rows=[]
for r in ('low','high'):
    cells={}
    for p in ('k=2','k=3','full','adaptive'):
        m,sd=ms(regime_f1[r][p]); cells[p]=f'{m:.4f}±{sd:.4f}'
    print(f'{r:6s} {cells["k=2"]:>14s} {cells["k=3"]:>14s} {cells["full"]:>14s} {cells["adaptive"]:>14s}')
    rows.append([r,cells['k=2'],cells['k=3'],cells['full'],cells['adaptive']])
pd.DataFrame(rows,columns=['Regime','k=2','k=3','full','adaptive']).to_csv('table_regime_fixedk.csv',index=False)

# 핵심 해석: high-load에서 full vs k=2 차이 = 'k 축소로 인한 추가 손실'
hi_full=ms(regime_f1['high']['full'])[0]; hi_k2=ms(regime_f1['high']['k=2'])[0]; hi_k3=ms(regime_f1['high']['k=3'])[0]
print(f'\n[해석] high-load: full={hi_full:.4f}, k=3={hi_k3:.4f}, k=2={hi_k2:.4f}')
print(f'       k 축소(full->k=2)로 인한 추가 손실 = {hi_full-hi_k2:.4f}')
print(f'       (작으면 A효과 작음=정책 안전 / 크면 고부하에서 k 축소가 해로움)')

print('\n=== 구간별 클래스별 F1 (adaptive k 적용) ===')
prows=[]
for r in ('low','high'):
    kad=3 if r=='low' else 2
    arr=np.mean(np.stack(perclass[r][kad]),axis=0)
    print(f'[{r}-load, k={kad}] ' + ', '.join(f'{c}={v:.3f}' for c,v in zip(CLASSES,arr)))
    prows.append([r]+[f'{v:.3f}' for v in arr])
pd.DataFrame(prows,columns=['Regime']+CLASSES).to_csv('table_regime_perclass.csv',index=False)

print(f'\n=== Sanity: overall adaptive macro-F1 = {ms(overall_adpt)[0]:.4f} (이전 0.940과 비교) ===')
