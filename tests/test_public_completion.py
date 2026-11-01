import json,tempfile,unittest
from pathlib import Path
from officegym.tasks.generate import generate
from officegym.tools.schema import tool_schemas
from officegym.tools.dispatcher import Dispatcher
from officegym.env.rollout import clone_world


class PublicCompletionTests(unittest.TestCase):
    def test_generated_public_task_exposes_duration_and_external_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);manifest=generate(root/'tasks',templates=['T01','T02'],count_per_template=1)
            for entry in manifest['instances']:
                task=json.loads((root/'tasks'/entry['path']/'task.json').read_text())['public_task']
                self.assertEqual(task['duration_seconds'],1800);self.assertIn('formal mail.send invitation',task['completion_requirements']['meeting'])
                if entry['template_id']=='T02':self.assertIn('request.type=availability',task['completion_requirements']['availability'])
            world=clone_world(root/'tasks'/manifest['instances'][0]['path']/'initial_world.sqlite',root/'world.sqlite')
            try:
                with world.db:world.db.execute("INSERT OR REPLACE INTO meta VALUES ('max_tool_calls','24')")
                result=Dispatcher(world).call('office.get_context',{})
                self.assertTrue(result['ok']);self.assertEqual(result['data']['max_tool_calls'],24);self.assertEqual(result['data']['tool_calls_used'],1)
            finally:world.close()

    def test_public_tool_schema_exposes_real_patch_and_request_fields(self):
        schemas={tool['name']:tool['inputSchema']['properties'] for tool in tool_schemas()}
        self.assertNotIn('status',schemas['calendar.update']['patch']['properties']);self.assertFalse(schemas['calendar.update']['patch']['additionalProperties'])
        self.assertEqual(schemas['mail.send']['request']['properties']['type']['enum'],['availability','clarification'])
