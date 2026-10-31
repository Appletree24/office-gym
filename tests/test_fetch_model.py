import hashlib
import io
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from officegym.training.fetch_model import fetch_file,check_file,run,REVISION,MODEL,SMALL,validate_plan


class Response(io.BytesIO):
    def __init__(self,data,status=200,headers=None):
        super().__init__(data)
        self.status=status
        self.headers=headers or {}


def item(data,algorithm='sha256'):
    digest=hashlib.sha256(data).hexdigest() if algorithm=='sha256' else hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    return dict(name='model-00001-of-00001.safetensors',size=len(data),algorithm=algorithm,digest=digest)


class FetchModelTests(unittest.TestCase):
    def test_invalid_caps_do_not_create_directory_or_access_network(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'model'
            for value in (0,-1,float('nan'),float('inf'),1801,True):
                with self.subTest(value=value),patch('urllib.request.urlopen') as network,self.assertRaises(ValueError):
                    run(root,max_wall_seconds=value)
                network.assert_not_called()
            self.assertFalse(root.exists())

    def test_plan_rejects_traversal_and_oversize(self):
        files=[dict(name=name,size=1,algorithm='sha256',digest='a'*64) for name in SMALL]
        files.append(item(b'valid'))
        plan=dict(model=MODEL,revision=REVISION,files=files,max_bytes=100,max_wall_seconds=60)
        validate_plan(plan)
        with self.assertRaises(ValueError):
            validate_plan(dict(plan,max_bytes=1))
        with self.assertRaises(ValueError):
            validate_plan(dict(plan,files=files+[dict(name='../escape',size=1,algorithm='sha256',digest='a'*64)]))

    def test_partial_range_resume_verifies_complete_sha256(self):
        data=b'0123456789abcdef';metadata=item(data)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);partial=root/(metadata['name']+'.part');partial.write_bytes(data[:5])
            def open_request(request,timeout):
                self.assertEqual(request.get_header('Range'),'bytes=5-')
                self.assertIn('/resolve/'+REVISION+'/',request.full_url)
                return Response(data[5:],206,{'Content-Range':'bytes 5-15/16'})
            with patch('urllib.request.urlopen',side_effect=open_request):
                fetch_file(metadata,root,REVISION,lambda *args:None)
            self.assertEqual((root/metadata['name']).read_bytes(),data)
            self.assertFalse(partial.exists())

    def test_ignored_range_and_digest_failure_are_retained(self):
        data=b'valid payload';metadata=item(data)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);partial=root/(metadata['name']+'.part');partial.write_bytes(data[:3])
            with patch('urllib.request.urlopen',return_value=Response(data)),self.assertRaisesRegex(ValueError,'range resume'):
                fetch_file(metadata,root,REVISION,lambda *args:None)
            self.assertEqual(partial.read_bytes(),data[:3])
            partial.write_bytes(b'X'*len(data))
            with patch('urllib.request.urlopen') as network,self.assertRaisesRegex(ValueError,'digest mismatch'):
                fetch_file(metadata,root,REVISION,lambda *args:None)
            network.assert_not_called()
            self.assertFalse((root/metadata['name']).exists())

    def test_git_blob_hash_and_zero_length_partial(self):
        data=b'configuration';metadata=item(data,'sha1_git_blob')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/(metadata['name']+'.part')).touch()
            with patch('urllib.request.urlopen',return_value=Response(data)):
                fetch_file(metadata,root,REVISION,lambda *args:None)
            self.assertTrue(check_file(root/metadata['name'],metadata))

    def test_symlink_and_existing_corruption_cannot_be_overwritten(self):
        data=b'contents';metadata=item(data)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);external=root/'external';external.write_bytes(data)
            destination=root/metadata['name'];destination.symlink_to(external)
            with self.assertRaisesRegex(ValueError,'symlinks'):
                fetch_file(metadata,root,REVISION,lambda *args:None)
            self.assertEqual(external.read_bytes(),data)
            destination.unlink();destination.write_bytes(b'bad data')
            with self.assertRaisesRegex(ValueError,'immutable identity'):
                fetch_file(metadata,root,REVISION,lambda *args:None)
            self.assertEqual(destination.read_bytes(),b'bad data')

    def test_oversized_metadata_is_retained_as_preflight_failure(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'checkpoint'
            with patch('urllib.request.urlopen',return_value=Response(b'X'*(2*1024*1024+1))),self.assertRaisesRegex(ValueError,'metadata exceeds'):
                run(root)
            failure=json.loads((root/'preflight_failure.json').read_text())
            self.assertEqual(failure['phase'],'metadata')
            self.assertFalse((root/'plan.json').exists())
            self.assertFalse((root/'model').exists())
