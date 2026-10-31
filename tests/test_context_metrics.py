import unittest
from officegym.eval.context_metrics import context_metrics


class ContextMetricTests(unittest.TestCase):
    def test_trigger_calls_cache_and_truncation_are_distinct(self):
        records=[dict(kind='context_selection',context_metrics_version='context-metrics-v1',input_tokens=100),
                 dict(kind='context_selection',context_metrics_version='context-metrics-v1',input_tokens=50,compression_triggered=True,summary_model_call=True,compressed=True),
                 dict(kind='context_selection',context_metrics_version='context-metrics-v1',input_tokens=50,compression_triggered=True,summary_cache_hit=True,compressed=True),
                 dict(kind='context_selection',context_metrics_version='context-metrics-v1',input_tokens=40,compression_triggered=True,summary_model_call=True,removed_groups=2)]
        result=context_metrics(records)
        self.assertEqual(result['context_samples'],4)
        self.assertEqual(result['context_token_sum'],240)
        self.assertEqual(result['peak_context_tokens'],100)
        self.assertEqual(result['compression_trigger_count'],3)
        self.assertEqual(result['summary_model_calls'],2)
        self.assertEqual(result['summary_cache_hits'],1)
        self.assertEqual(result['compressed_context_count'],2)
        self.assertEqual(result['truncated_context_count'],1)

    def test_old_or_missing_context_evidence_is_not_assumed_zero(self):
        with self.assertRaisesRegex(ValueError,'current metrics version'):
            context_metrics([dict(kind='context_selection',input_tokens=100)])
        with self.assertRaisesRegex(ValueError,'lack measured context'):
            context_metrics([dict(kind='model_response')])
