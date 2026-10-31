"""Real worker-kill software probe; not a model comparison or formal fault evaluation."""
import multiprocessing
import os
import sqlite3
import time
from pathlib import Path
from officegym.env.world import World, canonical
from officegym.env.service import serve, Client
from officegym.harness.journal import Journal


def committing_worker(socket_path, runtime, ready):
    journal = Journal(runtime)
    client = Client(socket_path)
    args = dict(recipient_ids=['b'],kind='invitation',subject='Review',body='',meeting_id='meeting_1',meeting_version=1)
    operation = journal.prepare('mail.send',args)
    result = client.call('mail.send',args,operation)
    if not result['ok']:
        raise RuntimeError(result)
    ready.send({'operation_id':operation,'pid':os.getpid(),'committed_result':result})
    # Fault controller kills this worker before any local acknowledgement.
    ready.recv()
    journal.acknowledge(operation,result)
    journal.close()


def recovering_worker(socket_path,runtime,results):
    journal = Journal(runtime)
    recovered = journal.recover(Client(socket_path).call)
    results.send({'pid':os.getpid(),'recovered':recovered})
    journal.close()


def run(out):
    root = Path(out).resolve()
    root.mkdir(parents=True,exist_ok=False)
    world = World.create(root/'world.sqlite',sim_time=100,deadline=10000,principal_id='a')
    from officegym.tools.dispatcher import Dispatcher
    with world.db:
        for person in ('a','b'):
            world.db.execute('INSERT INTO people VALUES (?,?,?,?,?)',(person,person,person+'@x','UTC','["*"]'))
    Dispatcher(world).call('calendar.create',dict(title='Review',participant_ids=['b'],start_at=4000,end_at=5800),'setup')
    world.close()
    context = multiprocessing.get_context('spawn')
    socket_path = root/'world.sock'
    server = context.Process(target=serve,args=(root/'world.sqlite',socket_path))
    worker = recovery = None
    server.start()
    try:
        deadline = time.monotonic()+10
        while not socket_path.exists():
            if not server.is_alive() or time.monotonic()>deadline:
                raise RuntimeError('world startup failed')
            time.sleep(.01)
        parent,child = context.Pipe()
        worker = context.Process(target=committing_worker,args=(socket_path,root/'runtime.sqlite',child))
        worker.start()
        if not parent.poll(15):
            raise RuntimeError('worker did not reach fault boundary')
        committed = parent.recv()
        worker.kill()
        worker.join(5)
        if worker.is_alive() or worker.exitcode != -9:
            raise RuntimeError('worker kill not verified')
        if not server.is_alive():
            raise RuntimeError('world died with worker')
        channel,other = context.Pipe()
        recovery = context.Process(target=recovering_worker,args=(socket_path,root/'runtime.sqlite',other))
        recovery.start()
        if not channel.poll(15):
            raise RuntimeError('recovery timeout')
        recovered = channel.recv()
        recovery.join(5)
        if recovery.exitcode != 0:
            raise RuntimeError('recovery failed')
        world = World(root/'world.sqlite')
        outbox = world.db.execute('SELECT count(*) FROM outbox').fetchone()[0]
        state_hash = world.state_hash()
        world.close()
        runtime = sqlite3.connect(root/'runtime.sqlite')
        status = runtime.execute('SELECT status FROM write_intents').fetchone()[0]
        runtime.close()
        if outbox != 1 or status != 'RECONCILED' or not recovered['recovered'][0]['ok']:
            raise RuntimeError('fault reconciliation invariant failed')
        report = dict(kind='software_fault_probe_not_model_evaluation',protocol='after_world_commit',world_pid=server.pid,world_survived=True,
                      killed_worker_exitcode=worker.exitcode,committed=committed,recovery=recovered,outbox_effects=outbox,journal_status=status,final_state_hash=state_hash)
        (root/'fault_report.json').write_text(canonical(report))
        return report
    finally:
        for process in (worker,recovery,server):
            if process is not None and process.is_alive():
                process.terminate()
                process.join(3)
                if process.is_alive():
                    process.kill()
                    process.join()
