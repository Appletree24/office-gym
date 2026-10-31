import unittest
from officegym.eval.serial_vllm import serial_schema


class SerialSchemaTests(unittest.TestCase):
    def test_only_explicit_serial_required_requests_are_constrained(self):
        schema=dict(type='array',minItems=1,items={'anyOf':[{'properties':{'name':{'enum':['board.read']}}}]})
        original=repr(schema)
        actual=serial_schema(schema,'required',False)
        self.assertEqual(actual['maxItems'],1)
        self.assertEqual(actual['items'],schema['items'])
        self.assertEqual(repr(schema),original)
        self.assertIs(serial_schema(schema,'required',True),schema)
        self.assertIs(serial_schema(None,'auto',False),None)
        self.assertIs(serial_schema(schema,'required',None),schema)

    def test_unknown_upstream_shape_fails_closed(self):
        for schema in (None,{},dict(type='array'),dict(type='object',minItems=1)):
            with self.subTest(schema=schema),self.assertRaises(ValueError):
                serial_schema(schema,'required',False)
