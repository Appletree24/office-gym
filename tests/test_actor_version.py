import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from officegym.training.actor_version import actor_identity,bind_actor_metadata,validate_actor_metadata,VersionedServer,validate_actor_batch


class ActorVersionTests(unittest.TestCase):
    def test_controller_binding_preserves_prepared_rows_and_updates_identity(self):
        original={'instance_dir':'/task','actor_version':'initial'}
        first=bind_actor_metadata(original,'initial',0);second=bind_actor_metadata(original,'initial',1)
        self.assertEqual(first['actor_version'],'initial');self.assertNotEqual(first['actor_version'],second['actor_version'])
        self.assertEqual(validate_actor_metadata(second,'initial'),1)
        self.assertEqual(original,{'instance_dir':'/task','actor_version':'initial'})
        for metadata in ({**original,'actor_update_step':0},{**original,'starting_actor_version':'foreign'},{**original,'actor_version':'foreign'}):
            with self.assertRaises(ValueError):bind_actor_metadata(metadata,'initial',1)
        for step in (True,-1,1.0,None):
            with self.assertRaises(ValueError):actor_identity('initial',step)

    def test_zero_is_verified_and_each_later_turn_is_checked_before_use(self):
        with TemporaryDirectory() as directory:
            output=SimpleNamespace(extra_fields={'global_steps':0,'min_global_steps':0,'max_global_steps':0},token_ids=[123],log_probs=[-.5])
            class Server:
                calls=[]
                async def generate(self,**kwargs):self.calls.append(kwargs);return output
            server=Server();guard=VersionedServer(server,0,directory)
            self.assertIs(asyncio.run(guard.generate(prompt_ids=[1])),output)
            self.assertEqual(server.calls,[{'prompt_ids':[1]}])
            output.extra_fields['max_global_steps']=1
            with self.assertRaisesRegex(ValueError,'stale or mixed'):asyncio.run(guard.generate(prompt_ids=[1,123]))
            self.assertFalse(json.loads((Path(directory)/'generation_version_0002.json').read_text())['accepted'])

    def test_missing_boolean_and_range_versions_fail_closed(self):
        for fields in ({},{'min_global_steps':None,'max_global_steps':None},{'min_global_steps':True,'max_global_steps':True},{'min_global_steps':0,'max_global_steps':1},{'min_global_steps':1,'max_global_steps':1,'global_steps':0}):
            with self.subTest(fields=fields),TemporaryDirectory() as directory:
                class Server:
                    async def generate(self,**kwargs):return SimpleNamespace(extra_fields=fields,token_ids=[2],log_probs=[-.1])
                with self.assertRaises(ValueError):asyncio.run(VersionedServer(Server(),1,directory).generate())

    def test_mixed_optimizer_batch_rejected(self):
        valid={'officegym_actor_version':[actor_identity('initial',1)]*2,'officegym_actor_update_step':[1,1],'officegym_actor_version_verified':[True,True],'min_global_steps':[1,1],'max_global_steps':[1,1]}
        validate_actor_batch(valid,'initial',1)
        with self.assertRaisesRegex(ValueError,'every optimizer row'):validate_actor_batch(valid,'initial',1,expected_rows=4)
        for key,value in (('officegym_actor_version','initial'),('officegym_actor_update_step',0),('officegym_actor_version_verified',False),('min_global_steps',0),('max_global_steps',2)):
            bad={name:list(values) for name,values in valid.items()};bad[key][1]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_actor_batch(bad,'initial',1)
