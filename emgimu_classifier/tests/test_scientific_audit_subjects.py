import unittest
from benchmarks.per_subject_analysis import subject_id,subject_ids_json
from emgimu.feature_bank.unibo_sequence_incremental import check_parent


class ScientificAuditTests(unittest.TestCase):
    def test_native_subjects_survive_summary(self):
        self.assertEqual(subject_id('u01'),'u01')
        self.assertEqual(subject_ids_json([{'subject':'u02'},{'subject':'u01'}]),'["u01", "u02"]')
        self.assertEqual(subject_ids_json([{'subject':'10'},{'subject':'2'}]),'[2, 10]')

    def test_population_rows_do_not_become_subjects(self):
        for value in ('ALL','SAME_USERS','N/A','',None):
            self.assertIsNone(subject_id(value))

    def test_final_day_parent_is_rejected(self):
        with self.assertRaises(ValueError):
            check_parent(dict(train_days=[1,2,3,4,5],target_days=[7,8],final_days_opened=True),[])

    def test_source_temperature_and_evaluation_trials_are_disjoint(self):
        manifest=dict(train_days=[1,2,3,4,5],target_days=[6],final_days_opened=False)
        with self.assertRaises(ValueError):
            check_parent(manifest,[dict(user='u01',train=['a'],calibration=['b'],evaluation=['b'])])


if __name__=='__main__':
    unittest.main()
