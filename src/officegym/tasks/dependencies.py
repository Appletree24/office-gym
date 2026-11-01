"""Dependency edges express prerequisites, not one solver's chosen ordering."""


GENERATOR_VERSION='generator-v6-replies-before-invitation'
ANNOTATION_VERSION='required-dependency-annotation-v4'


def dependency_edges(nodes):
    if len(nodes)!=len(set(nodes)) or 'plan' not in nodes or 'invite' not in nodes:
        raise ValueError('unique workflow stages including plan and invite required')
    if 'read_material' in nodes:raise ValueError('optional material reading cannot be an intrinsic dependency')
    if 'read_handoff_chain' in nodes:raise ValueError('optional handoff reading cannot be an intrinsic dependency')
    preparation = {'collect_replies'}
    core = [node for node in nodes if node not in preparation]
    edges = list(zip(core,core[1:]))
    for stage in nodes:
        if stage in preparation:
            target = 'invite'
            edges.append((stage,target))
    return edges


def annotate_required_dependencies(task):
    """Correct only observation metadata; public task and business inputs are untouched."""
    from .integrity import dependency_depth,dependency_width
    updated=dict(task);nodes=[node for node in task['dependency_dag']['nodes'] if node not in ('read_material','read_handoff_chain')]
    edges=dependency_edges(nodes)
    updated.update(dependency_dag=dict(nodes=nodes,edges=edges),dependency_path_length=dependency_depth(nodes,edges),dependency_width=dependency_width(nodes,edges),dependency_annotation_version=ANNOTATION_VERSION,optional_observation_tools=list(optional_observation_scopes(task['public_task'])),optional_observation_scopes=optional_observation_scopes(task['public_task']))
    return updated


def optional_observation_scopes(public):
    scopes={}
    if public.get('required_document_refs'):scopes['documents.read']='material_body'
    if public.get('handoff_thread_id'):scopes['mail.get_thread']='handoff_chain'
    return scopes
