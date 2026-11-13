"""Check regenerated results against the reviewed analytical contract."""
from pathlib import Path
import json,hashlib
import pandas as pd
R=Path(__file__).resolve().parent
checks={}
def read(p):return pd.read_csv(R/p,dtype={'listing_id':'string','host_id':'string'})
def check(n,v):
 checks[n]=bool(v)
 if not v:raise AssertionError(n)
def main():
 manifest=read('Shared_Preparation_and_Audits/RAW_DATA_MANIFEST.csv')
 for x in manifest.itertuples():
  p=R/'Shared_Preparation_and_Audits/raw_data'/x.file_name
  check('raw_hash_'+x.file_name,hashlib.sha256(p.read_bytes()).hexdigest()==x.sha256)
 d='Shared_Preparation_and_Audits/outputs/DP1_prepared/'
 for n,k in [('01_T1_RQ1_Q2_Shinjuku.csv',5689),('02_T2_RQ2_train_Q1_to_Q2.csv',4166),('03_T2_RQ2_test_Q2_to_Q3.csv',5689),('04_T3_RQ3_Q2_four_wards.csv',14935)]:
  f=read(d+n);check(n+'_rows_and_ids',len(f)==k and f.listing_id.is_unique)
 test=read(d+'03_T2_RQ2_test_Q2_to_Q3.csv');check('test_absent_317',test.present_next.eq(0).sum()==317)
 a=read('RQ1_Segmentation/outputs/M1_Output/Q1_id_cluster.csv')
 check('final_cluster_counts',a.groupby('cluster').size().to_dict()=={1:647,2:1185,3:1440,4:1030,5:1026})
 o=read('RQ1_Segmentation/outputs/DP2_Output/T1_RQ1_clean_observed.csv')
 check('rq1_exact_priced_cohort',set(o.listing_id)==set(test.loc[test.price_numeric.notna(),'listing_id']))
 m=read('RQ1_Segmentation/outputs/M2_Output/M2_cluster_profile.csv')
 joined=o.merge(a,on='listing_id',validate='one_to_one')
 for c in ['price_numeric','accommodates_clean','minimum_nights_clean','availability_365_clean','reviews_per_month_clean','calculated_host_listings_count_clean']:
  actual=joined.groupby('cluster')[c].median();expected=m.set_index('cluster')[c+'_median'];check('M2_observed_median_'+c,(actual-expected).abs().max()<1e-8)
 pred=read('RQ2_Persistence/outputs/M3/M3_test_predictions.csv')
 cm=read('RQ2_Persistence/outputs/M3/M3_confusion_matrix_test_random_forest.csv')
 check('RF_confusion_matrix_reproduced',cm[['predicted_0','predicted_1']].values.tolist()==[[13,304],[70,5302]])
 check('M3_prediction_rows',len(pred)==5689 and pred.listing_id.is_unique)
 for folder,name in [('RQ3_Ward_Comparison/DP4/outputs','DP4_QA.json'),('RQ3_Ward_Comparison/M4/outputs','M4_QA.json')]:
  p=R/folder/name
  check(name+'_PASS',p.exists() and json.loads(p.read_text()).get('status')=='PASS')
 (R/'VALIDATION.json').write_text(json.dumps({'status':'PASS','checks':checks},indent=2))
 print('PASS:',len(checks),'checks')
if __name__=='__main__':main()
