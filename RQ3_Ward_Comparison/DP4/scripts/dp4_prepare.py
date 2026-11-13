"""DP4: reproducible Q2 comparison data and QA. Run from the package root.
Comments explain the purpose in Vietnamese; outputs use English field names.
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WARD = 'neighbourhood_cleansed'
WARDS = ['Shinjuku Ku', 'Shibuya Ku', 'Toshima Ku', 'Taito Ku']
ROOMS = ['Entire home/apt', 'Private room', 'Shared room', 'Hotel room']
CAPS = ['1-2', '3-4', '5+']
PORTS = ['1', '2-5', '6+']
EXPECTED = {'Shinjuku Ku': 5689, 'Shibuya Ku': 2433, 'Toshima Ku': 3232, 'Taito Ku': 3581}
NUMERIC = ['price_numeric','accommodates_clean','bedrooms_clean','beds_clean',
           'bathrooms_clean','minimum_nights_clean','availability_365_clean',
           'number_of_reviews_clean','reviews_per_month_clean','review_scores_rating_clean',
           'calculated_host_listings_count_clean']
SOURCE_COLUMNS = ['listing_id','host_id','snapshot',WARD,'room_type','superhost_status',
                  *NUMERIC,'capacity_group','host_portfolio_group']


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_ids(path):
    # ID luôn là chuỗi để không mất chữ số khi đọc listing ID dài.
    return pd.read_csv(path, dtype={'listing_id':'string','host_id':'string'})


def load_sources(input_dir):
    # 1. Kiểm tra dữ liệu chính thức; dừng nếu có sai cohort, ID hoặc nhóm.
    d = read_ids(input_dir / '04_T3_RQ3_Q2_four_wards.csv')
    require(d.columns.tolist() == SOURCE_COLUMNS, 'Unexpected DP1 schema.')
    require(not d[['listing_id','host_id','snapshot',WARD,'room_type','capacity_group','host_portfolio_group']].isna().any().any(), 'Missing key/group.')
    require(d.listing_id.str.fullmatch(r'\d+').all(), 'Invalid listing ID.')
    require(d.host_id.str.fullmatch(r'\d+').all(), 'Invalid host ID.')
    require(not d.duplicated(['listing_id','snapshot']).any(), 'Duplicate baseline key.')
    require(set(d.snapshot) == {'2026_Q2'}, 'Expected Q2 baseline only.')
    require(d.groupby(WARD).size().to_dict() == EXPECTED, 'Official DP1 cohort changed; review before updating expectations.')
    require(set(d.room_type).issubset(ROOMS), 'Unknown room type.')
    require(set(d.capacity_group).issubset(CAPS), 'Unknown capacity group.')
    require(set(d.host_portfolio_group).issubset(PORTS), 'Unknown host portfolio group.')
    for col in NUMERIC:
        require(pd.api.types.is_numeric_dtype(d[col]), f'Non-numeric {col}.')
        require(np.isfinite(d[col].dropna()).all(), f'Infinite value in {col}.')
    require((d.price_numeric.dropna()>0).all(), 'Price must be positive when observed.')
    require(d.accommodates_clean.notna().all() and (d.accommodates_clean>=1).all(), 'Invalid capacity.')
    cap = pd.cut(d.accommodates_clean, [0,2,4,np.inf], labels=CAPS).astype('string')
    host = pd.cut(d.calculated_host_listings_count_clean, [0,1,5,np.inf], labels=PORTS).astype('string')
    require(cap.eq(d.capacity_group).all(), 'Capacity group disagrees with DP1 values.')
    require(host.eq(d.host_portfolio_group).all(), 'Host group disagrees with Tokyo-scoped count.')
    followup = read_ids(input_dir / 'Tokyo_Q3_listing_ids.csv')
    require(followup.columns.tolist()==['listing_id','snapshot'], 'Invalid Q3 ID index schema.')
    require(len(followup)==35970 and not followup.listing_id.isna().any(), 'Incomplete Q3 ID index.')
    require(followup.listing_id.str.fullmatch(r'\d+').all() and followup.listing_id.is_unique, 'Invalid/duplicate Q3 ID.')
    require(set(followup.snapshot)=={'2026_Q3'}, 'Wrong follow-up snapshot.')
    provenance = json.loads((input_dir/'source_provenance.json').read_text())
    for name, info in provenance['packaged_inputs'].items():
        require(digest(input_dir/name)==info['sha256'], f'Input checksum changed: {name}.')
    return d, followup


def add_metadata(source, followup):
    # 2. Thêm metadata, không điền thiếu hay biến đổi giá quan sát.
    d = source.copy()
    d['area_name'] = d[WARD].str.replace(' Ku','',regex=False)
    d['area_role'] = d[WARD].map({'Shinjuku Ku':'focal','Shibuya Ku':'primary_comparator',
                                'Toshima Ku':'primary_comparator','Taito Ku':'sensitivity_comparator'})
    d['comparison_group'] = d.room_type + ' | ' + d.capacity_group
    d['comparison_group_definition'] = 'room_type x capacity_group'
    for col in NUMERIC:
        d['missing_'+col] = d[col].isna().astype(int)
    d['followup_period'] = '2026_Q3'
    # Nhãn quan sát: dò vào TOÀN BỘ Tokyo Q3, không lọc theo quận tại Q3.
    d['present_next_observed'] = d.listing_id.isin(set(followup.listing_id)).astype(int)
    d['presence_label_source'] = 'Full Tokyo Q3 listing ID index'
    # Nhãn cụm chỉ thuộc Shinjuku; không gán cụm cho comparator.
    eligible = d[WARD].eq('Shinjuku Ku') & d.price_numeric.notna()
    d['rq1_model_eligible'] = eligible.astype(int)
    d['rq1_cluster'] = pd.Series(pd.NA, index=d.index, dtype='string')
    d['rq1_cluster_status'] = np.select(
        [~d[WARD].eq('Shinjuku Ku'), d.price_numeric.isna()],
        ['outside_rq1_scope','excluded_missing_price'], default='awaiting_final_labels')
    d['rq2_pred_present_probability'] = np.nan
    d['rq2_prediction_status'] = np.where(d[WARD].eq('Shinjuku Ku'),
                                         'awaiting_final_predictions','outside_rq2_scope')
    return d


def attach_results(d, path, kind):
    # 3. Ghép kết quả tùy chọn: đúng kỳ, đúng ID, quan hệ 1-1, không tăng dòng.
    r = read_ids(path)
    key = ['listing_id','snapshot'] if kind=='cluster' else ['listing_id','baseline_period','followup_period']
    value = 'rq1_cluster' if kind=='cluster' else 'rq2_pred_present_probability'
    require(set(r.columns)==set(key+[value]), f'{kind}: use the documented schema only.')
    require(len(r)>0 and not r[key+[value]].isna().any().any(), f'{kind}: empty/missing key or result.')
    require(not r.duplicated(key).any(), f'{kind}: duplicate join key.')
    require(set(r[key[1]])=={'2026_Q2'}, f'{kind}: wrong baseline period.')
    if kind!='cluster':
        require(set(r.followup_period)=={'2026_Q3'}, 'prediction: wrong follow-up period.')
        r[value] = pd.to_numeric(r[value],errors='raise')
        require(r[value].between(0,1).all(), 'Probabilities must be in [0,1].')
        r = r.rename(columns={'baseline_period':'snapshot'})
    else:
        r[value] = r[value].astype('string')
        require(r[value].str.strip().ne('').all(), 'Empty cluster label.')
    eligible = d.rq1_model_eligible.eq(1) if kind=='cluster' else d[WARD].eq('Shinjuku Ku')
    require(set(r.listing_id).issubset(set(d.loc[eligible,'listing_id'])), f'{kind}: IDs outside eligible Shinjuku cohort.')
    keys = ['listing_id','snapshot'] + ([] if kind=='cluster' else ['followup_period'])
    out = d.drop(columns=value).merge(r,on=keys,how='left',validate='one_to_one',sort=False)
    require(len(out)==len(d), 'Join inflated rows.')
    status = 'rq1_cluster_status' if kind=='cluster' else 'rq2_prediction_status'
    out.loc[eligible.to_numpy(),status] = 'not_in_supplied_results'
    out.loc[out[value].notna(),status] = 'matched'
    return out


def summarize(g):
    # 4. Mẫu số minh bạch: giá dùng n_price; cơ cấu/presence dùng toàn bộ listing.
    n = len(g); price = g.price_numeric.dropna()
    row = {'n_listings':n,'n_unique_hosts':g.host_id.nunique(), 'n_price':len(price),
           'n_price_missing':n-len(price),'price_coverage_pct':100*len(price)/n if n else np.nan,
           'price_median_jpy':price.median() if len(price) else np.nan,
           'price_q25_jpy':price.quantile(.25) if len(price) else np.nan,
           'price_q75_jpy':price.quantile(.75) if len(price) else np.nan,
           'n_present_next':int(g.present_next_observed.sum()),'n_presence_denominator':n,
           'presence_pct':100*g.present_next_observed.mean() if n else np.nan}
    return row


def build_tables(d, min_n):
    # 5. Bảng 48 ô bao gồm ô rỗng; cờ nhóm nhỏ chỉ là quy tắc mô tả, không phải kiểm định.
    ward = pd.DataFrame([{WARD:w,**summarize(d[d[WARD].eq(w)])} for w in WARDS])
    cells=[]
    for w in WARDS:
        for room in ROOMS:
            for cap in CAPS:
                g=d[d[WARD].eq(w)&d.room_type.eq(room)&d.capacity_group.eq(cap)]
                cells.append({WARD:w,'room_type':room,'capacity_group':cap,
                              'comparison_group':room+' | '+cap,**summarize(g)})
    cells=pd.DataFrame(cells)
    cells['small_listing_cell']=cells.n_listings.lt(min_n).astype(int)
    cells['small_price_cell']=cells.n_price.lt(min_n).astype(int)
    # So sánh được chuẩn bị riêng theo từng comparator, không gộp tùy tiện.
    focal=cells[cells[WARD].eq(WARDS[0])].set_index('comparison_group')
    support=[]
    for w in WARDS[1:]:
        peer=cells[cells[WARD].eq(w)].set_index('comparison_group')
        for group in focal.index:
            a,b=focal.loc[group],peer.loc[group]
            support.append({'comparator':w,'comparator_role':'sensitivity' if w=='Taito Ku' else 'primary',
              'comparison_group':group,'room_type':a.room_type,'capacity_group':a.capacity_group,
              'n_shinjuku':int(a.n_listings),'n_comparator':int(b.n_listings),
              'n_price_shinjuku':int(a.n_price),'n_price_comparator':int(b.n_price),
              'both_have_listings':int(min(a.n_listings,b.n_listings)>0),
              'both_listing_n_ge_threshold':int(min(a.n_listings,b.n_listings)>=min_n),
              'both_price_n_ge_threshold':int(min(a.n_price,b.n_price)>=min_n),
              'descriptive_min_n':min_n})
    missing=[]
    for w,g in d.groupby(WARD,sort=False):
        for col in NUMERIC:
            missing.append({WARD:w,'variable':col,'n_total':len(g),'n_observed':int(g[col].notna().sum()),
                            'n_missing':int(g[col].isna().sum()),'missing_pct':100*g[col].isna().mean()})
    structure=[]
    for w in WARDS:
        g=d[d[WARD].eq(w)]
        for variable,levels in [('room_type',ROOMS),('capacity_group',CAPS),('host_portfolio_group',PORTS)]:
            for level in levels:
                count=int(g[variable].eq(level).sum())
                structure.append({WARD:w,'dimension':variable,'category':level,'n_listings':count,
                                  'denominator_listings':len(g),'share_pct':100*count/len(g)})
    join=d.groupby([WARD,'rq1_cluster_status','rq2_prediction_status'],dropna=False).size().reset_index(name='n_listings')
    keys=d[['listing_id','snapshot','followup_period',WARD,'rq1_model_eligible','rq1_cluster_status','rq2_prediction_status']].copy()
    return {'T3_compare':d,'DP4_ward_summary':ward,'DP4_group_sample_sizes':cells,
            'DP4_pairwise_support':pd.DataFrame(support),'DP4_missingness':pd.DataFrame(missing),
            'DP4_structure_counts':pd.DataFrame(structure),'DP4_join_audit':join,'DP4_join_keys':keys}


def dictionary(d):
    # 6. Từ điển biến: định nghĩa, đơn vị và cách dùng để viết report.
    desc={
      'listing_id':('text ID','Listing identifier; keep as text.','key'),
      'host_id':('text ID','Host identifier; host counts are not additive across groups.','metadata'),
      'snapshot':('period','Baseline snapshot label 2026_Q2; see source dates in provenance.','key'),
      WARD:('ward','Ward at baseline, inherited from DP1.','group'),
      'room_type':('category','DP1 room type.','group'),
      'superhost_status':('category','DP1 superhost status; not a legal or quality certification.','description'),
      'price_numeric':('JPY/night','Observed advertised nightly price; no imputation, clipping or scaling.','price analysis'),
      'accommodates_clean':('guests','DP1 cleaned guest capacity.','description'),
      'bedrooms_clean':('bedrooms','DP1 cleaned bedrooms; missing retained.','description'),
      'beds_clean':('beds','DP1 cleaned beds; missing retained.','description'),
      'bathrooms_clean':('bathrooms','DP1 resolved numeric/text bathroom value; missing retained.','description'),
      'minimum_nights_clean':('nights','DP1 minimum stay requirement.','description'),
      'availability_365_clean':('days','Calendar availability over 365 days; not occupancy.','description'),
      'number_of_reviews_clean':('reviews','DP1 total reviews; not bookings.','description'),
      'reviews_per_month_clean':('reviews/month','DP1 review frequency; missing retained.','description'),
      'review_scores_rating_clean':('rating','DP1 rating; missing retained.','description'),
      'calculated_host_listings_count_clean':('listings/host','Supplied Tokyo-scoped host count; not recalculated within ward.','description'),
      'capacity_group':('category','1-2, 3-4, 5+ guests, inherited and checked against capacity.','group'),
      'host_portfolio_group':('category','1, 2-5, 6+ based on supplied Tokyo host count; distributions count listings.','group'),
      'area_name':('text','Short ward display label.','metadata'),
      'area_role':('category','Focal / primary comparator / sensitivity comparator.','metadata'),
      'comparison_group':('text','room_type | capacity_group; a descriptive stratum, not exact matching.','group'),
      'comparison_group_definition':('text','Explicit grouping rule.','metadata'),
      'followup_period':('period','2026_Q3, label observation period.','key'),
      'present_next_observed':('0/1','1 if ID occurs anywhere in full Tokyo Q3; 0 otherwise. Not survival or compliance.','retrospective outcome; never predictor'),
      'presence_label_source':('text','Source used to construct observed presence.','metadata'),
      'rq1_model_eligible':('0/1','Shinjuku Q2 with observed price, following current DP2 cohort.','metadata'),
      'rq1_cluster':('category','Optional FINAL Shinjuku cluster label; blank until supplied.','optional result'),
      'rq1_cluster_status':('category','matched / awaiting_final_labels / excluded_missing_price / outside_rq1_scope / not_in_supplied_results.','coverage'),
      'rq2_pred_present_probability':('0-1 probability','Optional frozen-model final test P(present_next=1); not the observed outcome.','optional result'),
      'rq2_prediction_status':('category','matched / awaiting_final_predictions / outside_rq2_scope / not_in_supplied_results.','coverage')}
    rows=[]
    for col in d:
        unit,definition,role=desc.get(col,('0/1','1 means missing in the corresponding original numeric variable; 0 means observed.','missingness audit'))
        rows.append({'variable':col,'dtype':str(d[col].dtype),'unit':unit,'definition':definition,'role':role})
    return pd.DataFrame(rows)


def run(input_dir=ROOT/'inputs',output_dir=ROOT/'outputs',clusters=None,predictions=None,min_n=30):
    # 7. Chạy toàn bộ pipeline; kiểm tra trước khi xuất file, không ghi đè dữ liệu nguồn.
    input_dir,output_dir=Path(input_dir),Path(output_dir)
    require(min_n>=1, 'min_n must be positive.')
    require(input_dir.resolve()!=output_dir.resolve(), 'Keep inputs and outputs separate.')
    source,followup=load_sources(input_dir)
    d=add_metadata(source,followup)
    if clusters: d=attach_results(d,Path(clusters),'cluster')
    if predictions: d=attach_results(d,Path(predictions),'prediction')
    pd.testing.assert_frame_equal(source.reset_index(drop=True),d[SOURCE_COLUMNS].reset_index(drop=True))
    # Đối chiếu với DP3 chính thức tại Shinjuku bằng ID, không dựa vào thứ tự dòng.
    reference=read_ids(input_dir/'03_T2_RQ2_test_Q2_to_Q3.csv')
    sj=d[d[WARD].eq('Shinjuku Ku')].set_index('listing_id')
    ref=reference.set_index('listing_id')
    require(set(sj.index)==set(ref.index), 'Shinjuku cohort differs from DP3.')
    disagreements=int((sj.present_next_observed!=ref.present_next.reindex(sj.index)).sum())
    require(disagreements==0,'Presence labels disagree with DP3.')
    tables=build_tables(d,min_n)
    require(tables['DP4_group_sample_sizes'].n_listings.sum()==len(d),'Group count mismatch.')
    require(tables['DP4_missingness'].query("variable == 'price_numeric'").n_missing.sum()==d.price_numeric.isna().sum(),'Price missingness mismatch.')
    require(tables['DP4_join_audit'].n_listings.sum()==len(d),'Join audit count mismatch.')
    tables['DP4_data_dictionary']=dictionary(d)
    qa={'status':'PASS','scope':'DP4 preparation and retrospective labels only; no model evaluation',
        'rows':len(d),'columns':len(d.columns),'source_columns_preserved':True,
        'duplicate_baseline_keys':int(d.duplicated(['listing_id','snapshot']).sum()),
        'row_inflation':len(d)-len(source),'price_missing_retained':int(d.price_numeric.isna().sum()),
        'shinjuku_label_disagreements_with_dp3':disagreements,
        'group_cells':len(tables['DP4_group_sample_sizes']),
        'pairwise_support_rows':len(tables['DP4_pairwise_support']),
        'rq1_labels_loaded':clusters is not None,'rq2_predictions_loaded':predictions is not None,
        'unresolved_rq1_eligible':int((d.rq1_model_eligible.eq(1)&d.rq1_cluster.isna()).sum()),
        'unresolved_rq2_in_scope':int((d[WARD].eq('Shinjuku Ku')&d.rq2_pred_present_probability.isna()).sum()),
        'small_cell_threshold':min_n,'small_cell_rule':'Descriptive caution only, not a significance criterion; no rows dropped.',
        'runtime':{'python':platform.python_version(),'pandas':pd.__version__,'numpy':np.__version__}}
    output_dir.mkdir(parents=True,exist_ok=True)
    for name,table in tables.items():
        table.to_csv(output_dir/(name+'.csv'),index=False,encoding='utf-8-sig')
    (output_dir/'DP4_QA.json').write_text(json.dumps(qa,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    inputs=[input_dir/n for n in ['04_T3_RQ3_Q2_four_wards.csv','Tokyo_Q3_listing_ids.csv','03_T2_RQ2_test_Q2_to_Q3.csv','source_provenance.json']]
    inputs += [Path(x) for x in [clusters,predictions] if x]
    outputs=[output_dir/(n+'.csv') for n in tables]+[output_dir/'DP4_QA.json']
    pd.DataFrame([{'role':role,'file':p.name,'sha256':digest(p),'bytes':p.stat().st_size}
                  for role,paths in [('input',inputs),('code',[Path(__file__)]),('output',outputs)] for p in paths]
                ).to_csv(output_dir/'DP4_manifest.csv',index=False)
    print(f'DP4 PASS | {len(d):,} rows | {len(tables["DP4_group_sample_sizes"])} group cells | no row inflation')
    print(f'Pending: {qa["unresolved_rq1_eligible"]:,} cluster labels; {qa["unresolved_rq2_in_scope"]:,} predictions. These do not block DP4 preparation.')
    return tables,qa


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Prepare DP4 comparison tables.')
    parser.add_argument('--input-dir',type=Path,default=ROOT/'inputs')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs')
    parser.add_argument('--clusters',type=Path)
    parser.add_argument('--predictions',type=Path)
    parser.add_argument('--min-n',type=int,default=30)
    args=parser.parse_args()
    run(args.input_dir,args.output_dir,args.clusters,args.predictions,args.min_n)
