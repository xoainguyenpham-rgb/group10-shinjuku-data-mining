"""Kiểm tra các lỗi có thể làm sai kết quả so sánh hoặc ghép mô hình."""
import sys
from pathlib import Path
import tempfile
import unittest
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import dp4_prepare as m

class DP4Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source,cls.followup=m.load_sources(ROOT/'inputs')
        cls.d=m.add_metadata(cls.source,cls.followup)

    def test_original_values_and_missing_prices_preserved(self):
        pd.testing.assert_frame_equal(self.source,self.d[m.SOURCE_COLUMNS])
        self.assertEqual(int(self.d.price_numeric.isna().sum()),953)
        self.assertEqual(self.d.groupby(m.WARD).size().to_dict(),m.EXPECTED)

    def test_membership_labels_and_empty_cells(self):
        ids=set(self.followup.listing_id)
        self.assertTrue(self.d.present_next_observed.eq(self.d.listing_id.isin(ids).astype(int)).all())
        t=m.build_tables(self.d,30)
        self.assertEqual(len(t['DP4_group_sample_sizes']),48)
        self.assertEqual(t['DP4_group_sample_sizes'].n_listings.sum(),14935)
        self.assertEqual(len(t['DP4_pairwise_support']),36)
        empty=t['DP4_group_sample_sizes'].query('n_listings == 0')
        self.assertTrue(empty.price_median_jpy.isna().all())
        self.assertTrue(empty.presence_pct.isna().all())
        structure=t['DP4_structure_counts']
        for (ward,dimension),g in structure.groupby([m.WARD,'dimension']):
            self.assertEqual(int(g.n_listings.sum()),m.EXPECTED[ward])
            self.assertAlmostEqual(g.share_pct.sum(),100)

    def attempt(self,frame,kind):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'result.csv';frame.to_csv(p,index=False)
            return m.attach_results(self.d,p,kind)

    def cluster(self):
        ids=self.d.loc[self.d.rq1_model_eligible.eq(1),'listing_id'].head(2)
        return pd.DataFrame({'listing_id':ids,'snapshot':'2026_Q2','rq1_cluster':['0','1']})

    def test_partial_cluster_join_keeps_all_rows_and_marks_missing(self):
        out=self.attempt(self.cluster(),'cluster')
        self.assertEqual(len(out),14935)
        self.assertEqual(int(out.rq1_cluster.notna().sum()),2)
        self.assertEqual(int(out.rq1_cluster_status.eq('not_in_supplied_results').sum()),5326)
        self.assertEqual(int(out.rq1_cluster_status.eq('excluded_missing_price').sum()),361)
        pd.testing.assert_frame_equal(self.source,out[m.SOURCE_COLUMNS])

    def test_reject_duplicate_cluster_ids(self):
        r=self.cluster()
        with self.assertRaises(ValueError):self.attempt(pd.concat([r,r.iloc[:1]]),'cluster')

    def test_reject_wrong_period(self):
        r=self.cluster();r['snapshot']='2026_Q1'
        with self.assertRaises(ValueError):self.attempt(r,'cluster')

    def test_reject_comparator_cluster(self):
        r=self.cluster();r.loc[r.index[0],'listing_id']=self.d.loc[self.d[m.WARD].eq('Shibuya Ku'),'listing_id'].iloc[0]
        with self.assertRaises(ValueError):self.attempt(r,'cluster')

    def test_reject_price_missing_cluster(self):
        r=self.cluster();r.loc[r.index[0],'listing_id']=self.d.loc[self.d[m.WARD].eq('Shinjuku Ku')&self.d.price_numeric.isna(),'listing_id'].iloc[0]
        with self.assertRaises(ValueError):self.attempt(r,'cluster')

    def test_probability_zero_is_valid_but_missing_is_not_zero(self):
        r=self.cluster()[['listing_id']].copy()
        r['baseline_period']='2026_Q2';r['followup_period']='2026_Q3'
        r['rq2_pred_present_probability']=[0.,1.]
        out=self.attempt(r,'prediction')
        self.assertEqual(int(out.rq2_pred_present_probability.notna().sum()),2)
        self.assertEqual(int(out.rq2_prediction_status.eq('matched').sum()),2)
        self.assertEqual(int(out.rq2_pred_present_probability.eq(0).sum()),1)
        r['rq2_pred_present_probability']=[1.1,0.5]
        with self.assertRaises(ValueError):self.attempt(r,'prediction')

    def test_reject_wrong_followup_and_missing_result(self):
        r=self.cluster()[['listing_id']].copy()
        r['baseline_period']='2026_Q2';r['followup_period']='2026_Q2';r['rq2_pred_present_probability']=0.5
        with self.assertRaises(ValueError):self.attempt(r,'prediction')
        r['followup_period']='2026_Q3';r.loc[r.index[0],'rq2_pred_present_probability']=None
        with self.assertRaises(ValueError):self.attempt(r,'prediction')

if __name__=='__main__':unittest.main(verbosity=2)
