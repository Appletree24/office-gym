"""Dependency edges express prerequisites, not one solver's chosen ordering."""


def dependency_edges(nodes):
    if len(nodes)!=len(set(nodes)) or 'plan' not in nodes or 'invite' not in nodes:
        raise ValueError('unique workflow stages including plan and invite required')
    preparation = {'read_material','collect_replies','read_handoff_chain'}
    core = [node for node in nodes if node not in preparation]
    edges = list(zip(core,core[1:]))
    for stage in nodes:
        if stage in preparation:
            target = 'confirm' if stage=='read_handoff_chain' and 'confirm' in nodes else 'plan'
            edges.append((stage,target))
    return edges
