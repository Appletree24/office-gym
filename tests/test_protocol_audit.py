import json
import tempfile
import unittest
from pathlib import Path
from officegym.eval.protocol_audit import audit_trace


class ProtocolAuditTests(unittest.TestCase):
    def test_missing_reused_mismatched_counts_and_multiple_calls_fail(self):
        response=dict(kind='model_response',input_tokens=12,raw=dict(choices=[dict(message=dict(tool_calls=[{}]))]))
        selection=dict(kind='context_selection',input_tokens=12)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'trajectory.jsonl'
            for rows,counts,serial in [([selection,response],True,True),([response],False,True),([selection,response,response],False,True),([dict(selection,input_tokens=13),response],False,True),([selection,dict(response,raw=dict(choices=[dict(message=dict(tool_calls=[{},{}]))]))],True,False),([],False,False)]:
                path.write_text('\n'.join(json.dumps(row) for row in rows))
                result=audit_trace(path)
                self.assertEqual(result['token_counts_match'],counts)
                self.assertEqual(result['single_tool_responses'],serial)
