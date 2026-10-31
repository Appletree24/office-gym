import unittest
from officegym.tasks.integrity import fingerprints,validate_unique,dependency_depth,check_holdout


class IntegrityTests(unittest.TestCase):
    def test_renaming_and_semantic_constraints(self):
        first = fingerprints({'participants':['alice'],'duration':30},{'owner':'alice'},['reply'],'v1',identity_map={'alice':'person_1'})
        renamed = fingerprints({'participants':['bob'],'duration':30},{'owner':'bob'},['reply'],'v1',identity_map={'bob':'person_1'})
        changed = fingerprints({'participants':['bob'],'duration':60},{'owner':'bob'},['reply'],'v1',identity_map={'bob':'person_1'})
        self.assertNotEqual(first['exact_fingerprint'],renamed['exact_fingerprint'])
        self.assertEqual(first['family_fingerprint'],renamed['family_fingerprint'])
        self.assertNotEqual(changed['family_fingerprint'],renamed['family_fingerprint'])
        entries = [dict(instance_id='a',split='train',**first),dict(instance_id='b',split='id_test',**first)]
        with self.assertRaises(ValueError):
            validate_unique(entries)

    def test_dependency_and_holdout(self):
        self.assertEqual(dependency_depth(['plan','approval','invite'],[('plan','approval'),('approval','invite')]),3)
        with self.assertRaises(ValueError):
            dependency_depth(['a','b'],[('a','b'),('b','a')])
        with self.assertRaises(ValueError):
            check_holdout(['attendee_changed','document_updated'],[['attendee_changed','document_updated']])
        check_holdout(['document_updated','attendee_changed'],[['attendee_changed','document_updated']])
