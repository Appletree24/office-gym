import multiprocessing
import tempfile
import time
import unittest
from pathlib import Path
from officegym.env.world import World
from officegym.env.service import serve,Client


class ServiceTests(unittest.TestCase):
    def test_separate_process_world_and_fixed_context(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            world = World.create(root/'w.sqlite',sim_time=100,deadline=10000,principal_id='a')
            world.close()
            process = multiprocessing.get_context('spawn').Process(target=serve,args=(root/'w.sqlite',root/'world.sock'))
            process.start()
            try:
                deadline = time.monotonic()+10
                while not (root/'world.sock').exists():
                    if not process.is_alive() or time.monotonic()>deadline:
                        self.fail('world service failed to start')
                    time.sleep(.01)
                client = Client(root/'world.sock')
                self.assertEqual(len(client.request({'method':'tools/list'})['tools']),24)
                self.assertEqual(client.call('office.get_context',{})['data']['principal_id'],'a')
                self.assertEqual(client.request({'method':'tools/call','name':'office.get_context','arguments':{},'principal_id':'b'})['protocol_error'],'INVALID_REQUEST')
                self.assertEqual(client.call('board.read',{})['data']['version'],1)
            finally:
                process.terminate()
                process.join(5)
                if process.is_alive():
                    process.kill()
                    process.join()
