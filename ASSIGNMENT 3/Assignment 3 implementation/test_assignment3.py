
import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from main import task1, task2, task3, task4, task5, task6, task7

class Assignment3Tests(unittest.TestCase):
    def test_task1_three_query_outcomes(self):
        r=task1()["query_results"]
        self.assertTrue(r["P11_entails"]["entailed"])
        self.assertTrue(r["Safe11_entails"]["contradicted"])
        self.assertTrue(r["W12_unknown"]["unknown"])
    def test_task2_policy_and_resolution(self):
        r=task2()
        self.assertGreaterEqual(len(r["policy_rules"]),10)
        self.assertTrue(r["proof"]["entailed"])
        self.assertEqual(len(r["chain_scaling"]),4)
    def test_task3_query_mix(self):
        r=task3()["query_counts"]
        self.assertEqual(r,{"success":5,"fail":5,"binding":5})
    def test_task4_occurs_check(self):
        r=task4()["unification_tests"]
        self.assertIsNone(r[-1]["mgu"])
        self.assertEqual(r[2]["mgu"],{"?x":"Alice","?y":"Bob"})
    def test_task5_unifier_used(self):
        r=task5()
        self.assertEqual(len(r["unification_substitutions"]),4)
        self.assertTrue(all(x is not None for x in r["unification_substitutions"]))
    def test_task6_forward_and_backward(self):
        r=task6()
        self.assertEqual(len(r["forward_scaling"]),3)
        self.assertTrue(r["backward_query"]["proved"])
    def test_task7_30_cases_and_scale(self):
        r=task7()
        self.assertEqual(r["test_summary"]["total"],30)
        self.assertEqual([x["rules"] for x in r["scale"]],[40,400,2000])

if __name__=="__main__": unittest.main()
