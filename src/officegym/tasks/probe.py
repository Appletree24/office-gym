"""Small hand-designed software probes; never formal evaluation examples."""
import json
from pathlib import Path
from officegym.env.world import World, canonical
from officegym.tools.dispatcher import Dispatcher
from officegym.graders.result import evaluate
from .reference import solve


def run_probes(out):
    out = Path(out)
    out.mkdir(parents=True,exist_ok=True)
    results = []
    for approval in (False,True):
        hashes = []
        for repeat in range(2):
            directory = out / (('approval' if approval else 'basic')+'_'+str(repeat))
            directory.mkdir(exist_ok=False)
            task = dict(principal_id='assistant',required_participant_ids=['owner','client'],allowed_window_utc=[1800010000,1800020000],deadline_at=1800009000,requires_approval=approval,approver_id='owner',title='Client review')
            (directory/'task.json').write_text(canonical(task))
            world = World.create(directory/'world.sqlite',sim_time=1800000000,deadline=task['deadline_at'],principal_id='assistant')
            client = Dispatcher(world)
            with world.db:
                for person in ('assistant','owner','client'):
                    world.db.execute('INSERT INTO people VALUES (?,?,?,?,?)',(person,person,person+'@office.test','UTC','["*"]'))
                client.set_meta('public_task',task)
                client.set_meta('approval_rules',{'owner':{'decision':'grant','delay_seconds':120}})
            with (directory/'reference_trace.jsonl').open('w') as trace:
                def call(name,args,operation=None):
                    result = client.call(name,args,operation)
                    trace.write(canonical(dict(tool=name,arguments=args,operation_id=operation,result=result))+'\n')
                    trace.flush()
                    return result
                solve(task,call)
            grade = evaluate(directory/'world.sqlite',task)
            hashes.append(world.state_hash())
            world.close()
            (directory/'reference_grade.json').write_text(canonical(grade))
            if not grade['full_success']:
                raise RuntimeError(grade)
        if len(set(hashes)) != 1:
            raise RuntimeError('non-deterministic reference replay')
        results.append(dict(case='approval' if approval else 'basic',replay_hash=hashes[0],passed=True))
    (out/'probe_results.json').write_text(json.dumps({'kind':'software_probe_not_model_evaluation','cases':results},indent=2))
    return results
