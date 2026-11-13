"""Reproduce M4 descriptive comparisons from DP4 and final Orange RQ1 labels.
Run: python scripts/m4_fixed.py (paths resolve relative to this package).
"""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'; OUT.mkdir(exist_ok=True)
FOCAL='Shinjuku Ku'
COMPS={'Shibuya':'Shibuya Ku','Toshima':'Toshima Ku','Taito':'Taito Ku'}
COLORS={'Shinjuku':'#B44450','Shibuya':'#4477AA','Toshima':'#228877','Taito':'#7B8490'}
NAMES={1:'Extended stay',2:'Higher availability',3:'Lower availability',4:'Large host listing count',5:'Higher price and capacity'}
# 1. Validate baseline, follow-up, exact string IDs and final cluster membership.
df=pd.read_csv(ROOT/'inputs/T3_compare.csv',dtype={'listing_id':str,'host_id':str})
cl=pd.read_csv(ROOT/'inputs/Q1_id_cluster.csv',dtype={'listing_id':str})
assert len(df)==14935 and df.listing_id.is_unique
assert set(df.snapshot)=={'2026_Q2'} and set(df.followup_period)=={'2026_Q3'}
assert df.present_next_observed.isin([0,1]).all()
assert len(cl)==5328 and cl.listing_id.is_unique and set(cl.cluster)=={1,2,3,4,5}
sjmask=df.neighbourhood_cleansed.eq(FOCAL)
assert set(cl.listing_id)==set(df.loc[sjmask & df.price_numeric.notna(),'listing_id'])
df=df.drop(columns=['rq1_cluster'],errors='ignore').merge(cl[['listing_id','cluster']],on='listing_id',how='left',validate='one_to_one')
assert len(df)==14935 and df.loc[~sjmask,'cluster'].isna().all()
assert df.loc[df.cluster.notna()].groupby('cluster').size().to_dict()=={1:647,2:1185,3:1440,4:1030,5:1026}
S=df[df.neighbourhood_cleansed.eq(FOCAL)].copy()
assert len(S)==5689 and S.price_numeric.isna().sum()==361
assert (df.price_numeric.dropna()>0).all()
TABLES={}
def save(name,t):
 TABLES[name]=t
 t.to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig')
def rate(d):return 100*d.present_next_observed.mean() if len(d) else np.nan
def cells(a,b):
 rows=[]
 for group in sorted(set(df.comparison_group)):
  x=a[a.comparison_group.eq(group)];y=b[b.comparison_group.eq(group)]
  rows.append(dict(stratum=group,n_shinjuku=len(x),n_comparator=len(y),n_price_shinjuku=x.price_numeric.notna().sum(),n_price_comparator=y.price_numeric.notna().sum(),median_shinjuku_jpy=x.price_numeric.median(),median_comparator_jpy=y.price_numeric.median(),presence_shinjuku_pct=rate(x),presence_comparator_pct=rate(y)))
 t=pd.DataFrame(rows)
 t['price_gap_pct']=100*(t.median_shinjuku_jpy/t.median_comparator_jpy-1)
 t['presence_gap_pp']=t.presence_shinjuku_pct-t.presence_comparator_pct
 return t
def supported(t,n=30):
 return t[['n_shinjuku','n_comparator','n_price_shinjuku','n_price_comparator']].min(axis=1).ge(n)
def aggregate(t):
 if t.empty:return np.nan,np.nan
 w=t.n_shinjuku/t.n_shinjuku.sum()
 return 100*np.expm1(np.sum(w*np.log(t.median_shinjuku_jpy/t.median_comparator_jpy))),np.sum(w*t.presence_gap_pp)
# 2. Full-ward summaries and explicit priced / price-missing denominators.
ward=[]; audit=[];struct=[]
for name,wardname in {'Shinjuku':FOCAL,**COMPS}.items():
 d=df[df.neighbourhood_cleansed.eq(wardname)]
 ward.append(dict(ward=name,n_listings=len(d),n_hosts=d.host_id.nunique(),n_price=d.price_numeric.notna().sum(),median_price_jpy=d.price_numeric.median(),n_present=d.present_next_observed.sum(),presence_pct=rate(d)))
 for cohort,mask in [('Priced',d.price_numeric.notna()),('Price missing',d.price_numeric.isna())]:
  z=d[mask]; audit.append(dict(ward=name,cohort=cohort,n_listings=len(z),n_present=z.present_next_observed.sum(),presence_pct=rate(z)))
 for dim in ['room_type','capacity_group','host_portfolio_group']:
  for cat,n in d[dim].value_counts(dropna=False).items():struct.append(dict(ward=name,dimension=dim,category=cat,n_listings=n,denominator=len(d),share_pct=100*n/len(d)))
save('M4_ward_overview',pd.DataFrame(ward));save('M4_price_availability_audit',pd.DataFrame(audit));save('M4_structure_shares',pd.DataFrame(struct))
# 3. Ward-level standardisation preserves the original, documented estimand.
wg=[];summary=[];sens=[]
for name,wardname in COMPS.items():
 C=df[df.neighbourhood_cleansed.eq(wardname)];t=cells(S,C);t['supported']=supported(t);t['comparator']=name;wg.append(t)
 u=t[t.supported];pg,rg=aggregate(u)
 summary.append(dict(comparator=name,role='Sensitivity' if name=='Taito' else 'Primary',raw_price_gap_pct=100*(S.price_numeric.median()/C.price_numeric.median()-1),weighted_geometric_price_gap_pct=pg,raw_presence_gap_pp=rate(S)-rate(C),weighted_presence_gap_pp=rg,n_supported_strata=len(u),shinjuku_covered_n=u.n_shinjuku.sum(),shinjuku_coverage_pct=100*u.n_shinjuku.sum()/len(S)))
 for minimum in [1,20,30,50]:
  u=t[supported(t,minimum)];pg,rg=aggregate(u)
  sens.append(dict(comparator=name,scenario=f'Cell minimum {minimum}',weighted_geometric_price_gap_pct=pg,weighted_presence_gap_pp=rg,shinjuku_coverage_pct=100*u.n_shinjuku.sum()/len(S),note='Full baseline; weights use supported Shinjuku listing counts'))
 # Priced-only sensitivity changes the cohort and weights explicitly.
 sp=S[S.price_numeric.notna()];cp=C[C.price_numeric.notna()];u=cells(sp,cp);u=u[supported(u)];pg,rg=aggregate(u)
 sens.append(dict(comparator=name,scenario='Priced only',weighted_geometric_price_gap_pct=pg,weighted_presence_gap_pp=rg,shinjuku_coverage_pct=100*u.n_shinjuku.sum()/len(sp),note='Coverage denominator is 5,328 priced Shinjuku listings'))
 # Price trimming is a price-only check: keep full-cohort presence unchanged.
 for rule in ['Cap JPY 200000','Ward price percentile 99']:
  a=S.copy();b=C.copy()
  for z in [a,b]:
   cut=200000 if rule=='Cap JPY 200000' else z.price_numeric.quantile(.99)
   z.loc[z.price_numeric.gt(cut),'price_numeric']=np.nan
  u=cells(a,b);u=u[supported(u)];pg,rg=aggregate(u)
  sens.append(dict(comparator=name,scenario=rule,weighted_geometric_price_gap_pct=pg,weighted_presence_gap_pp=rg,shinjuku_coverage_pct=100*u.n_shinjuku.sum()/len(S),note='High prices omitted from price medians only; listing rows retained'))
save('M4_within_stratum_gaps',pd.concat(wg,ignore_index=True));save('M4_ward_comparisons',pd.DataFrame(summary));save('M4_sensitivity_checks',pd.DataFrame(sens))
# 4. Segment benchmarks compare both sides over the SAME supported strata.
# Require at least 30 priced listings in the segment AND comparator stratum.
# Segment price is a weighted geometric mean of within-stratum median ratios.
# Segment presence is the weighted mean of within-stratum rate differences.
seg=[];segcells=[];full=[];segment_sensitivity=[]
for k in range(1,6):
 A=S[S.cluster.eq(k)]
 full.append(dict(cluster=k,name=NAMES[k],n_listings=len(A),share_of_priced_pct=100*len(A)/5328,share_of_all_shinjuku_pct=100*len(A)/5689,n_present=A.present_next_observed.sum(),full_presence_pct=rate(A),full_median_price_jpy=A.price_numeric.median()))
 for name,wardname in COMPS.items():
  B=df[df.neighbourhood_cleansed.eq(wardname)&df.price_numeric.notna()]
  t=cells(A,B);t['supported']=supported(t);t['cluster']=k;t['comparator']=name;segcells.append(t)
  for minimum in [20,30,50]:
   v=t[supported(t,minimum)];gprice,gpresence=aggregate(v)
   segment_sensitivity.append(dict(cluster=k,comparator=name,minimum_per_side=minimum,n_supported_strata=len(v),covered_segment_n=int(v.n_shinjuku.sum()),coverage_pct=100*v.n_shinjuku.sum()/len(A),weighted_geometric_price_gap_pct=gprice,presence_gap_pp=gpresence))
  u=t[t.supported];pg,rg=aggregate(u)
  if u.empty:ash=bsh=np.nan;covered=0
  else:
   w=u.n_shinjuku/u.n_shinjuku.sum();ash=np.sum(w*u.presence_shinjuku_pct);bsh=np.sum(w*u.presence_comparator_pct);covered=int(u.n_shinjuku.sum())
  seg.append(dict(cluster=k,name=NAMES[k],comparator=name,full_segment_n=len(A),covered_segment_n=covered,excluded_segment_n=len(A)-covered,segment_coverage_pct=100*covered/len(A),n_supported_strata=len(u),covered_segment_presence_pct=ash,weighted_comparator_presence_pct=bsh,presence_gap_pp=rg,weighted_geometric_price_gap_pct=pg))
save('M4_segment_sensitivity',pd.DataFrame(segment_sensitivity));save('M4_segment_profiles',pd.DataFrame(full));save('M4_segment_cell_audit',pd.concat(segcells,ignore_index=True));save('M4_segment_benchmarks',pd.DataFrame(seg))
# 5. Price-missing listings are an unstratified supplementary cohort, not a cluster.
uS=S[S.price_numeric.isna()];unpriced=[]
for name,wn in COMPS.items():
 c=df[df.neighbourhood_cleansed.eq(wn)&df.price_numeric.isna()]
 unpriced.append(dict(comparator=name,n_shinjuku=len(uS),n_comparator=len(c),shinjuku_presence_pct=rate(uS),comparator_presence_pct=rate(c),unadjusted_presence_gap_pp=rate(uS)-rate(c)))
save('M4_unpriced_comparison',pd.DataFrame(unpriced))
# 6. Record which published segment values change and why.
old=pd.read_csv(ROOT/'inputs/previous_segment_benchmarks.csv').iloc[:5].copy();old['cluster']=range(1,6)
changes=[]
for r in seg:
 k=r['cluster']; c=r['comparator'].lower();o=old[old.cluster.eq(k)].iloc[0]
 changes.append(dict(cluster=k,comparator=r['comparator'],previous_presence_gap_pp=o[f'presence_gap_vs_{c}_pp'],revised_presence_gap_pp=r['presence_gap_pp'],previous_price_ratio_pct=o[f'price_gap_vs_{c}_mix_pct'],revised_weighted_geometric_price_gap_pct=r['weighted_geometric_price_gap_pct'],revised_coverage_pct=r['segment_coverage_pct'],reason='Use identical supported strata on both sides; require segment-cell n>=30; price metric now aggregates within-stratum ratios'))
save('M4_revision_audit',pd.DataFrame(changes))
# 7. Numerical consistency checks and input fingerprints.
summary=TABLES['M4_ward_comparisons'].set_index('comparator')
assert np.allclose(summary.loc[['Shibuya','Toshima'],'weighted_geometric_price_gap_pct'],[-11.613375670725345,21.782567227362403])
assert np.allclose(summary.shinjuku_coverage_pct,98.99806644401477)
for r in seg:
 assert np.isclose(r['presence_gap_pp'],r['covered_segment_presence_pct']-r['weighted_comparator_presence_pct'])
 assert 0<r['covered_segment_n']<=r['full_segment_n']
 assert r['covered_segment_n']+r['excluded_segment_n']==r['full_segment_n']
qa={'status':'PASS','baseline_rows':len(df),'clustered_rows':len(cl),'unclustered_price_missing':361,'final_cluster_counts':{str(k):int((S.cluster==k).sum()) for k in range(1,6)},'segment_minimum_per_side':30,'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'inputs').glob('*.csv')},'uncertainty':'Descriptive point estimates only; no significance stars or inferential intervals','rq2_predictions_used':False,'scope':'Checks cover implemented joins, cohort counts, formulas and reproduction of ward headline values.'}
(OUT/'M4_QA.json').write_text(json.dumps(qa,indent=2))
# 8. Figures use the same tables as the report; all units and cohorts are explicit.
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(12,5.6));summary=summary.loc[list(COMPS)]
for ax,cols,title,xlabel in [(axes[0],['raw_price_gap_pct','weighted_geometric_price_gap_pct'],'Price comparison','Gap (%)'),(axes[1],['raw_presence_gap_pp','weighted_presence_gap_pp'],'Next-snapshot presence','Gap (percentage points)')]:
 for j,(col,label,mk) in enumerate(zip(cols,['Raw','Stratum-weighted'],['o','D'])):
  for i,(name,r) in enumerate(summary.iterrows()):
   val=r[col];y=2-i+(j-.5)*.22;ax.scatter(val,y,color=COLORS[name],marker=mk,s=65,label=label if i==0 else None);ax.annotate(f'{val:+.1f}',(val,y),xytext=(7,0),textcoords='offset points',va='center',fontsize=10)
 ax.axvline(0,color='#333333',lw=.8);ax.set_yticks([2,1,0],['Shibuya','Toshima','Taito (sensitivity)']);ax.set_ylim(-.5,2.5);ax.margins(x=.22);ax.set_title(title);ax.set_xlabel(xlabel);ax.grid(axis='x',alpha=.15);ax.legend(loc='lower right',fontsize=9)
fig.suptitle('Shinjuku versus comparator wards',fontsize=17,fontweight='bold')
fig.text(.05,.03,'Q2 prices; Q2 to Q3 observed presence. Weighted results cover 5,632 / 5,689 Shinjuku listings (99.0%).\nWeighted price = geometric mean of stratum median ratios; weighted presence = mean stratum rate difference.',fontsize=9)
fig.tight_layout(rect=[0,.12,1,.94]);fig.savefig(OUT/'M4_01_ward_comparisons.png',dpi=180);plt.close(fig)
# A readable structural figure with a stable ward palette.
fig,axes=plt.subplots(1,3,figsize=(14,5.7));st=TABLES['M4_structure_shares']
for ax,dim,cats,title in zip(axes,['room_type','capacity_group','host_portfolio_group'],[['Entire home/apt','Private room','Shared room','Hotel room'],['1-2','3-4','5+'],['1','2-5','6+']],['Room type','Guest capacity','Tokyo host listing count']):
 for j,name in enumerate(['Shinjuku',*COMPS]):
  z=st[(st.ward==name)&(st.dimension==dim)].set_index('category');vals=z.share_pct.reindex(cats,fill_value=0)
  ax.bar(np.arange(len(cats))+(j-1.5)*.19,vals,width=.18,color=COLORS[name],label=name)
 ax.set_xticks(np.arange(len(cats)),['Entire home\nor apartment','Private\nroom','Shared\nroom','Hotel\nroom'] if dim=='room_type' else cats);ax.set_title(title);ax.set_ylim(0,100);ax.grid(axis='y',alpha=.15)
axes[0].set_ylabel('Share of listings (%)');fig.suptitle('Market structure in Q2',fontsize=17,fontweight='bold');handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.91),ncol=4,frameon=False)
fig.text(.05,.025,'Denominators: Shinjuku 5,689; Shibuya 2,433; Toshima 3,232; Taito 3,581.\nHost scale describes listings associated with a host across Tokyo; shares count listings, not hosts.',fontsize=9)
fig.tight_layout(rect=[0,.13,1,.84]);fig.savefig(OUT/'M4_02_market_structure.png',dpi=180);plt.close(fig)
# Separate gap and coverage panels avoid comparing full-cluster outcomes with restricted benchmarks.
sg=TABLES['M4_segment_benchmarks'];fig,axes=plt.subplots(1,2,figsize=(12,6),gridspec_kw={'width_ratios':[2,1]},sharey=True)
for j,name in enumerate(COMPS):
 z=sg[sg.comparator.eq(name)].sort_values('cluster');ys=np.arange(5)[::-1]+(j-1)*.18
 axes[0].scatter(z.presence_gap_pp,ys,color=COLORS[name],label=name,s=50)
 for x,y in zip(z.presence_gap_pp,ys):axes[0].annotate(f'{x:+.1f}',(x,y),xytext=(6,0),textcoords='offset points',fontsize=9,va='center')
 axes[1].scatter(z.segment_coverage_pct,ys,color=COLORS[name],s=50)
 for x,y in zip(z.segment_coverage_pct,ys):axes[1].annotate(f'{x:.1f}%',(x,y),xytext=(6,0),textcoords='offset points',fontsize=9,va='center')
axes[0].axvline(0,color='#333333',lw=.8);axes[0].set_yticks(range(5),[f'C{k}  {NAMES[k]}' for k in range(5,0,-1)]);axes[0].set_xlabel('Covered segment minus weighted comparator (pp)');axes[0].set_xlim(sg.presence_gap_pp.min()-1,sg.presence_gap_pp.max()+1.2);axes[0].grid(axis='x',alpha=.15)
axes[1].set_xlabel('Share of the full Shinjuku segment (%)');axes[1].set_xlim(max(0,sg.segment_coverage_pct.min()-5),106);axes[1].grid(axis='x',alpha=.15)
fig.suptitle('Segment presence gaps on matched strata',fontsize=17,fontweight='bold');fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.55,.91),ncol=3,frameon=False)
fig.text(.04,.025,'Q2 priced listings; observed presence in Q3. Each retained room type × capacity stratum has at least 30 listings\nin the Shinjuku segment and comparator. Both sides use the same retained strata and segment weights.\nPoint estimates are descriptive; they do not establish statistical significance or policy effects.',fontsize=9)
fig.tight_layout(rect=[0,.14,1,.85]);fig.savefig(OUT/'M4_03_segment_presence.png',dpi=180);plt.close(fig)
# Show the full range, including Taito's price-missing cohort.
a=TABLES['M4_price_availability_audit'];fig,ax=plt.subplots(figsize=(10,5.8))
for j,cohort in enumerate(['Priced','Price missing']):
 z=a[a.cohort.eq(cohort)].set_index('ward').loc[['Shinjuku',*COMPS]];x=np.arange(4)+(j-.5)*.34
 ax.bar(x,z.presence_pct,width=.32,color='#5C4B8B' if j==0 else '#D9A441',label=cohort)
 for xx,(_,r) in zip(x,z.iterrows()):ax.text(xx,r.presence_pct+1,f'{r.presence_pct:.1f}%\nn={int(r.n_listings):,}',ha='center',fontsize=9)
ax.set_xticks(range(4),['Shinjuku','Shibuya','Toshima','Taito']);ax.set_ylim(0,110);ax.set_yticks(range(0,101,20));ax.set_ylabel('Observed presence in Q3 (%)');ax.grid(axis='y',alpha=.15);ax.legend(loc='lower left');ax.set_title('Price availability changes the comparison cohort',fontsize=16,fontweight='bold')
fig.text(.07,.025,'Unadjusted ward rates for Q2 cohorts. Price-missing listings are not a sixth RQ1 cluster.\nPresence means the listing ID appears in the full Tokyo Q3 snapshot.',fontsize=9)
fig.tight_layout(rect=[0,.1,1,1]);fig.savefig(OUT/'M4_04_price_missing_audit.png',dpi=180);plt.close(fig)
print('M4 fixed: checks passed; '+str(len(TABLES))+' CSV tables and 4 figures written.')
print(TABLES['M4_segment_benchmarks'].to_string(index=False))
