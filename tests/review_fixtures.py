"""Synthetic review attestations for software tests only, never human evidence."""
import json
import hashlib
from officegym.env.world import canonical
from officegym.training.mask_review import prepare
from officegym.training.review_admission import register,CHECKS
from test_mask_review import fixture


def reviewed_fixture(data,metadata=None,*,count=5):
    tokenizer=fixture(data,count=count)
    manifest=json.loads((data/'manifest.json').read_text())
    # Preserve lineage/branch fields while using valid authored alignment vectors.
    if metadata:
        extra={key:value for key,value in metadata.items() if key not in ('accepted','samples_sha256','manual_alignment_review_status')}
        manifest.update(extra)
    tokenizer=bind_authored_sources(data,manifest)
    (data/'manifest.json').write_text(canonical(manifest)+'\n')
    review=data/'synthetic_review_fixture'
    prepare(data,review,tokenizer,tokenizer_manifest={'fixture':True})
    checklist=json.loads((review/'human_checklist.json').read_text())
    checklist.update(status='passed',reviewer='AUTHORED SOFTWARE FIXTURE — NOT A HUMAN REVIEW',reviewed_at_utc='2000-01-01T00:00:00Z')
    for row in checklist['samples']:
        row.update({key:True for key in CHECKS});row['notes']='Authored protocol fixture; no human inspected teacher data.'
    (review/'human_checklist.json').write_text(canonical(checklist)+'\n')
    register(data,review/'packet.json',review/'human_checklist.json')
    return review


def bind_authored_sources(data,manifest):
    """Build parser fixtures, explicitly not genuine model/isolation evidence."""
    from officegym.graders.versions import EVAL_GRADER_VERSION,REWARD_VERSION
    from officegym.harness.versions import HARNESS_VERSION
    from officegym.harness.isolation import ISOLATION_VERSION
    from officegym.training.teacher import extract_trajectory
    from officegym.training.sft_sources import source_hashes
    from test_mask_review import FixtureTokenizer
    class Tokenizer(FixtureTokenizer):
        def __call__(self,messages,tools):
            index=int(messages[0]['content'].split(':')[0])
            return dict(input_ids=[index+10,2,3],assistant_mask=[0,1,0],token_roles=['user','assistant','tool'])
    tokenizer=Tokenizer()
    conversations=[json.loads(line) for line in (data/'conversations.jsonl').read_text().splitlines()]
    samples=[];recovered=[];accepted=[]
    for index,conversation in enumerate(conversations):
        path=(data/f'authored_teacher_parser_fixture_{index}').resolve();path.mkdir()
        messages=conversation['messages'];messages[0]['content']=str(index)+':AUTHORED PROTOCOL FIXTURE, NOT MODEL DATA'
        evidence=dict(version=ISOLATION_VERSION,verified=True,uid=65534,gid=65534,groups=[],no_new_privs=True,dumpable=0,cannot_regain_root=True,grader_unavailable=True,root_entries=[],inherited_descriptors=[0,1,2],parent_death_signal='SIGKILL',environment_empty=True,protected_paths_denied=['/authored-fixture/'+str(i) for i in range(5)])
        result=dict(source_kind='AUTHORED_SOFTWARE_FIXTURE_NOT_MODEL_OR_ISOLATION_PROOF',split='train',policy_kind='inference_service',instance_id=conversation['instance_id'],model_version='authored-fixture',grade=dict(full_success=True,eval_grader_version=EVAL_GRADER_VERSION,reward_version=REWARD_VERSION),termination=dict(state='DECLARED_DONE'),operation_audit=dict(consistent=True),policy_isolation=evidence)
        records=[dict(kind='run_header',harness_version=HARNESS_VERSION,mode='full',model_version='authored-fixture',initial_messages=messages[:1],tool_schemas=[])]
        for turn in range(2):
            assistant,observation=messages[1+2*turn:3+2*turn];call=assistant['tool_call'];action=str(turn)
            records.extend([dict(kind='model_response',tool=call['name'],arguments=call['arguments'],raw={'authored_software_fixture':True},content=None),dict(kind='action_start',action_id=action,tool=call['name'],arguments=call['arguments']),dict(kind='interaction',action_id=action,messages=[assistant,observation])])
        for name,payload in [('result.json',result),('policy_isolation.json',evidence)]:
            (path/name).write_text(canonical(payload)+'\n')
        (path/'trajectory.jsonl').write_text(''.join(canonical(row)+'\n' for row in records))
        trajectory=extract_trajectory(path);encoded=tokenizer(trajectory['messages'],[]);identity=hashlib.sha256(canonical(trajectory).encode()).hexdigest()
        sample=dict(instance_id=trajectory['instance_id'],source_run=str(path),trajectory_sha256=identity,**encoded,labels=[-100,2,-100])
        samples.append(sample);recovered.append(trajectory);accepted.append(dict(instance_id=sample['instance_id'],source_run=str(path),tokens=3,trajectory_sha256=identity,source_files_sha256=source_hashes(path)))
    (data/'samples.jsonl').write_text(''.join(canonical(row)+'\n' for row in samples));(data/'conversations.jsonl').write_text(''.join(canonical(row)+'\n' for row in recovered))
    manifest.update(accepted=accepted,training_tokens=len(samples)*3,max_length=8192,samples_sha256=hashlib.sha256((data/'samples.jsonl').read_bytes()).hexdigest())
    return tokenizer
