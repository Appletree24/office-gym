import unittest
from officegym.tasks.dependencies import dependency_edges
from officegym.tasks.integrity import dependency_depth


class DependencyTests(unittest.TestCase):
    def test_material_and_replies_have_no_invented_order(self):
        nodes = ['plan','read_material','collect_replies','confirm','invite','meeting_end','minutes','todos']
        edges = dependency_edges(nodes)
        self.assertIn(('read_material','plan'),edges)
        self.assertIn(('collect_replies','plan'),edges)
        self.assertNotIn(('read_material','collect_replies'),edges)
        self.assertNotIn(('plan','read_material'),edges)
        self.assertEqual(dependency_depth(nodes,edges),7)

    def test_preparation_blocks_confirmation_and_handoff_can_be_read_independently(self):
        nodes = ['read_handoff_chain','plan','prepare_todo','complete_preparation','confirm','invite']
        edges = dependency_edges(nodes)
        self.assertIn(('complete_preparation','confirm'),edges)
        self.assertIn(('read_handoff_chain','confirm'),edges)
        self.assertNotIn(('read_handoff_chain','plan'),edges)

    def test_extra_events_do_not_hide_heldout_ordered_combination(self):
        from officegym.tasks.integrity import check_holdout
        forbidden = [('AttendeeChanged','DocumentVersionPublished')]
        with self.assertRaisesRegex(ValueError,'held-out event combination'):
            check_holdout(['ReplyArrived','AttendeeChanged','Notification','DocumentVersionPublished','MeetingEnded'],forbidden)
        check_holdout(['DocumentVersionPublished','AttendeeChanged'],forbidden)
