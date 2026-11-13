"""Rebuild the submitted analysis in dependency order; never tunes on Q3."""
from pathlib import Path
import argparse,subprocess,sys,shutil,json,hashlib,os
ROOT=Path(__file__).resolve().parent

def run(rel):
 print('Running',rel,flush=True)
 subprocess.run([sys.executable,str(ROOT/rel)],cwd=ROOT,check=True,env={**os.environ,'MPLBACKEND':'Agg'})
def copy(src,dst):
 dst=ROOT/dst;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/src,dst)
def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['all','shared','rq1','rq2','rq3'],default='all');a=p.parse_args()
 shared='Shared_Preparation_and_Audits';prepared=shared+'/outputs/DP1_prepared/'
 if a.stage in ['all','shared']:
  for s in ['du1_audit','dp1_prepare','du4_comparators']:run(shared+'/scripts/'+s+'.py')
 if a.stage in ['all','rq1']:
  for n in ['01_T1_RQ1_Q2_Shinjuku.csv','05_variable_roles.csv']:copy(prepared+n,'RQ1_Segmentation/inputs/'+n)
  for s in ['dp2_prepare','m1_evaluate','m2_profile']:run('RQ1_Segmentation/scripts/'+s+'.py')
 if a.stage in ['all','rq2']:
  for n in ['02_T2_RQ2_train_Q1_to_Q2.csv','03_T2_RQ2_test_Q2_to_Q3.csv','15_RQ2_label_audit_NOT_predictors.csv']:copy(prepared+n,'RQ2_Persistence/inputs/'+n)
  for s in ['dp3_prepare','m3_run_orange']:run('RQ2_Persistence/scripts/'+s+'.py')
 if a.stage in ['all','rq3']:
  import pandas as pd
  base='RQ3_Ward_Comparison/DP4/inputs/'
  for n in ['04_T3_RQ3_Q2_four_wards.csv','03_T2_RQ2_test_Q2_to_Q3.csv']:copy(prepared+n,base+n)
  source=ROOT/shared/'raw_data/Tokyo_listings_2026_Q3.csv'
  ids=pd.read_csv(source,usecols=['id'],dtype='string').rename(columns={'id':'listing_id'}).assign(snapshot='2026_Q3')
  ids.to_csv(ROOT/base/'Tokyo_Q3_listing_ids.csv',index=False)
  provenance={'purpose':'Generated from current DP1 and original full Tokyo Q3 snapshot.','packaged_inputs':{}}
  for n in ['04_T3_RQ3_Q2_four_wards.csv','03_T2_RQ2_test_Q2_to_Q3.csv','Tokyo_Q3_listing_ids.csv']:
   b=(ROOT/base/n).read_bytes();provenance['packaged_inputs'][n]={'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
  (ROOT/base/'source_provenance.json').write_text(json.dumps(provenance,indent=2))
  run('RQ3_Ward_Comparison/DP4/scripts/dp4_prepare.py')
  copy('RQ3_Ward_Comparison/DP4/outputs/T3_compare.csv','RQ3_Ward_Comparison/M4/inputs/T3_compare.csv')
  copy('RQ1_Segmentation/outputs/M1_Output/Q1_id_cluster.csv','RQ3_Ward_Comparison/M4/inputs/Q1_id_cluster.csv')
  run('RQ3_Ward_Comparison/M4/scripts/m4_fixed.py')
if __name__=='__main__':main()
