"""These paired outcomes are authored fixtures, never live tool/model evidence."""
import unittest
from officegym.training.tool_equivalence import compare_tools,freeze


class ToolEquivalenceTests(unittest.TestCase):
    def test_failed_runs_are_retained_and_infra_blocks_acceptance(self):
        before=[dict(success=True,infra_error=False),dict(success=False,infra_error=False)]
        after=[dict(success=True,infra_error=False),dict(success=False,infra_error=False)]
        self.assertTrue(compare_tools(before,after,0,0,.5)['passed'])
        self.assertEqual(compare_tools(before,after,0,0,.5)['before_success_rate'],.5)
        after[1]['infra_error']=True
        self.assertFalse(compare_tools(before,after,0,0,.5)['passed'])
        with self.assertRaises(ValueError):
            compare_tools(before,after[:-1],0,0,.5)

    def test_paired_disagreement_not_hidden_by_equal_success_rate(self):
        before=[dict(success=True,infra_error=False),dict(success=False,infra_error=False)]
        after=[dict(success=False,infra_error=False),dict(success=True,infra_error=False)]
        report=compare_tools(before,after,0,.1,.5)
        self.assertEqual(report['success_rate_delta'],0)
        self.assertEqual(report['disagreement_rate'],1)
        self.assertFalse(report['passed'])

    def test_thresholds_rejected_before_any_source_io(self):
        for changes in ({'max_success_rate_delta':float('inf')},{'min_success_rate':0}):
            with self.assertRaises(ValueError):
                freeze(dict(max_success_rate_delta=.01,max_disagreement_rate=.01,min_success_rate=.9)|changes,'missing')

    def test_new_duplicates_and_threshold_roundoff_do_not_slip_through(self):
        before=[dict(success=index<7,infra_error=False,duplicate_effects=0) for index in range(10)]
        after=[dict(success=index<8,infra_error=False,duplicate_effects=0) for index in range(10)]
        self.assertTrue(compare_tools(before,after,.1,.1,.7)['passed'])
        after[0]['duplicate_effects']=1
        self.assertFalse(compare_tools(before,after,.1,.1,.7)['passed'])

    def test_rule_must_be_frozen_before_tool_results_exist(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            spec=dict(group='sft_full',max_success_rate_delta=.01,max_disagreement_rate=.01,min_success_rate=.9)
            for name in ('numeric_report','numeric_plan','dataset_manifest','dev_plan','before_results','after_results'):
                spec[name]=str(root/(name+'.json'))
            Path(spec['before_results']).write_text('fixture already collected')
            with self.assertRaisesRegex(ValueError,'before collecting'):
                freeze(spec,root/'rule.json')
