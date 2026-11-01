"""Incremental template generation. Unsupported templates fail explicitly."""
import tempfile
from pathlib import Path
from officegym.env.world import World,canonical
from officegym.tools.dispatcher import Dispatcher
from .artifacts import write_instance
from .reference import REFERENCE_VERSION
from officegym.tools.schema import TOOL_SCHEMA_VERSION
from .integrity import validate_unique,check_holdout
from .dependencies import dependency_edges,annotate_required_dependencies,GENERATOR_VERSION
from .controls import PROFILE,material

from .partitions import TRAIN_TEMPLATES,SUPPORTED,SEED_RANGES,HELDOUT_SIGNATURES


def generate(out,templates=TRAIN_TEMPLATES,count_per_template=3,seed=None,split='probe',heldout_combo=False,depth_level=None,perturbation_level=None):
    if depth_level is not None and (type(depth_level) is not int or depth_level not in (1,2,3) or tuple(templates)!=('T04',)):
        raise ValueError('controlled depth currently requires T04 and level 1/2/3')
    if perturbation_level is not None and (depth_level is None or type(perturbation_level) is not int or perturbation_level not in (0,1,2)):
        raise ValueError('controlled perturbation requires controlled depth and level 0/1/2')
    if depth_level is not None and heldout_combo:
        raise ValueError('controlled scan and combo distribution are separate')
    if not templates or not set(templates)<=set(SUPPORTED):
        raise ValueError('unsupported template; implemented: '+','.join(SUPPORTED))
    if split not in SEED_RANGES:
        raise ValueError('formal test distribution not yet implemented')
    if seed is None:
        seed = SEED_RANGES[split][0]
    if type(count_per_template) is not int or count_per_template<1 or type(seed) is not int or seed<0:
        raise ValueError('invalid generation counts or seed')
    if heldout_combo and split not in ('combo_test','probe'):
        raise ValueError('held-out combinations cannot enter train/dev/id distributions')
    if split=='combo_test' and (not heldout_combo or tuple(templates)!=('T07',)):
        raise ValueError('combo distribution currently requires T07 held-out combination')
    if split=='id_test' and any(template not in TRAIN_TEMPLATES for template in templates):
        raise ValueError('ID test requires training templates')
    if split=='train' and any(template not in TRAIN_TEMPLATES for template in templates):
        raise ValueError('development template cannot enter training')
    if split=='dev' and any(not template.startswith('D') for template in templates):
        raise ValueError('dev split requires development templates')
    if split=='unseen_test' and any(not template.startswith('U') for template in templates):
        raise ValueError('unseen split requires isolated unseen templates')
    low,high = SEED_RANGES[split]
    if not low <= seed <= seed+count_per_template-1 <= high:
        raise ValueError('seed range crosses dataset split boundary')
    if len(set(templates)) != len(templates):
        raise ValueError('duplicate template selection')
    root = Path(out)
    root.mkdir(parents=True,exist_ok=False)
    entries = []
    for template in templates:
        for index in range(count_per_template):
            number = seed+index
            base = 1800000000 + number*86400
            public = dict(principal_id='assistant',required_participant_ids=['owner','client'],allowed_window_utc=[base+10000,base+20000],deadline_at=base+9000,title='Review '+str(number),requires_approval=template!='T01',approver_id='owner')
            if depth_level is not None:
                public.update(deadline_at=base+PROFILE['deadline_offset_seconds'],monitor_until=base+PROFILE['monitor_offset_seconds'],allowed_window_utc=[base+offset for offset in PROFILE['allowed_window_offsets_seconds']],message_cap_per_recipient=PROFILE['message_cap_per_recipient'])
            nodes = ['plan','invite'] if template=='T01' else ['plan','confirm','invite']
            private, rules = {}, {}
            if template in ('T08','U04') or depth_level==3:
                public.update(followup_required=True,deadline_at=base+25000)
                action = dict(title='Send proposal '+str(number),assignee_id='owner',due_at=base+30000)
                private['required_todos'] = [action]
                rules['minutes_rules'] = {'*':{'delay_seconds':120,'action_items':[action]}}
                nodes += ['meeting_end','minutes','todos']
            if template in ('T02','D01') or (depth_level is not None and depth_level>=2):
                public['collect_availability'] = True
                nodes.insert(1,'collect_replies')
                rules['reply_rules'] = {person:{'availability':{'delay_seconds':180,'response':'provided','facts':{'available_windows':[[base+12000,base+18000]]}}} for person in ('owner','client')}
            if template=='U03':
                public['preparation_todos'] = [dict(title='Prepare review material '+str(number),assignee_id='assistant',due_at=base+5000)]
                nodes = ['plan','prepare_todo','complete_preparation','confirm','invite']
            scheduled = []
            if template=='U04':
                public.update(followup_monitor_until=base+18000,allowed_recipient_ids=['owner','client','replacement'])
                nodes += ['owner_change','todo_correction','followup_message']
                scheduled.append(dict(id='owner_change',due_at=base+15000,kind='Notification',payload={'type':'TodoOwnerChanged','meeting_selector':{'organizer_id':'assistant','title':public['title']},'title':action['title'],'assignee_id':'replacement'}))
            if template in ('T06','T07','D02','U02'):
                public.update(monitor_until=base+3000,allowed_recipient_ids=['owner','client','replacement'])
                payload = {'meeting_selector':{'organizer_id':'assistant','title':public['title']}}
                if template in ('T06','D02'):
                    payload.update(start_at=base+14000,end_at=base+15800)
                else:
                    payload['participant_ids'] = ['owner','replacement']
                    if template=='U02':
                        payload['approver_id'] = 'replacement'
                        nodes += ['replacement_confirmation','updated_invite']
                scheduled.append(dict(id='perturbation',due_at=base+1500,kind='MeetingRescheduleRequested' if template in ('T06','D02') else 'AttendeeChanged',payload=payload))
            refs = [{'document_id':'agenda','version':1}] if template in ('T05','D02') or heldout_combo or (depth_level is not None and (depth_level>=2 or perturbation_level==2)) else []
            if refs:
                public['required_document_refs'] = refs
            if template in ('T05','D02') or heldout_combo:
                if template=='T05':
                    public['monitor_until'] = base+3000
                public['track_document_ids'] = ['agenda']
                scheduled.append(dict(id='material_update',due_at=base+2200,kind='DocumentVersionPublished',payload={'document_id':'agenda','expected_version':1,'content':'Updated material '+str(number)}))
            if depth_level is not None:
                level = perturbation_level or 0
                if level:
                    public['monitor_until'] = base+5000
                    scheduled.append(dict(id='scan_reschedule',due_at=base+2500,kind='MeetingRescheduleRequested',payload={'meeting_selector':{'organizer_id':'assistant','title':public['title']},'start_at':base+14000,'end_at':base+15800}))
                if level==2:
                    public['track_document_ids'] = ['agenda']
                    scheduled.append(dict(id='scan_material',due_at=base+3500,kind='DocumentVersionPublished',payload={'document_id':'agenda','expected_version':1,'content':material('Scan material update '+str(number))}))
            signature = [event['kind'] for event in sorted(scheduled,key=lambda event:(event['due_at'],event['id']))]
            if split=='train':
                check_holdout(signature,HELDOUT_SIGNATURES)
            rules['approval_rules'] = {'owner':{'decision':'grant','delay_seconds':120}}
            if template=='U03':
                rules['approval_rules']['owner']['required_preparation'] = public['preparation_todos']
            if template=='U01':
                public.update(handoff_thread_id='handoff',initial_coordinator_id='owner')
                nodes.insert(0,'read_handoff_chain')
                rules['approval_rules']['replacement'] = {'decision':'grant','delay_seconds':180}
            if template=='U02':
                rules['approval_rules']['replacement'] = {'decision':'grant','delay_seconds':180}
            with tempfile.TemporaryDirectory() as temporary:
                world = World.create(Path(temporary)/'seed.sqlite',sim_time=base,deadline=public['deadline_at'],principal_id='assistant')
                try:
                    with world.db:
                        for person in ('assistant','owner','client','replacement'):
                            world.db.execute('INSERT INTO people VALUES (?,?,?,?,?)',(person,person,person+'@office.test','UTC','["*"]'))
                    if depth_level is not None:
                        with world.db:
                            for noise_index in range(PROFILE['background_messages']):
                                identity='background_'+str(noise_index)
                                body=('Background note '+str(noise_index)+' for '+str(number)).ljust(PROFILE['background_message_chars'],'.')
                                world.db.execute('INSERT INTO messages VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',(identity,identity,'owner','["assistant"]','coordination','Archived background note',body,'[]',None,None,None,base-1000+noise_index))
                    if template=='U01':
                        with world.db:
                            world.db.execute('INSERT INTO messages VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',('handoff_1','handoff','owner','["assistant","replacement"]','coordination','Coordinator handoff','Replacement now coordinates this review.','[]',canonical({'type':'coordinator_handoff','new_coordinator_id':'replacement','approver_id':'replacement'}),None,None,base))
                    client = Dispatcher(world)
                    if template in ('T03','D01'):
                        # Trusted construction uses business validation without Agent billing.
                        with world.db:
                            client.execute('calendar.create',dict(title='Existing commitment',participant_ids=['owner'],start_at=base+10000,end_at=base+13000))
                    if refs or depth_level is not None:
                        with world.db:
                            client.execute('documents.put',dict(document_id='agenda',expected_version=0,title='Agenda',content=material('Review material '+str(number)) if depth_level is not None else 'Review material '+str(number)))
                    with world.db:
                        for event in scheduled:
                            world.db.execute('INSERT INTO scheduled_events VALUES (?,?,?,?,?,?,?,?)',(event['id'],event['due_at'],10,event['kind'],'{}',canonical(event['payload']),'["assistant"]','pending'))
                        client.set_meta('public_task',public)
                        for key,value in rules.items():
                            client.set_meta(key,value)
                    task = dict(instance_id=template+'_'+str(number)+(('_D'+str(depth_level)+'P'+str(perturbation_level or 0)) if depth_level is not None else ''),split=split,template_id=template,scenario_family_id='family_'+str(number),depth_level=depth_level if depth_level is not None else (3 if template in ('T08','U04') else 1),perturbation_level=len(scheduled),event_signature=signature,seed=number,public_task=public,dependency_dag={'nodes':nodes,'edges':dependency_edges(nodes)})
                    if depth_level is not None:
                        task.update(controlled_scan=True,control_profile=dict(PROFILE))
                    task=annotate_required_dependencies(task)
                    entries.append(write_instance(root,task,world,private,dict(rules,scheduled_events=scheduled)))
                finally:
                    world.close()
    validate_unique(entries)
    manifest = dict(task_generator_version=GENERATOR_VERSION,reference_solver_version=REFERENCE_VERSION,tool_schema_version=TOOL_SCHEMA_VERSION,status='development_distribution_not_frozen',heldout_signatures=HELDOUT_SIGNATURES,instances=entries)
    (root/'manifest.json').write_text(canonical(manifest)+'\n')
    return manifest
