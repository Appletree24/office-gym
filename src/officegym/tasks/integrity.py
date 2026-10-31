"""Dataset integrity primitives. Seeds and split labels are not task identity."""
import hashlib
from officegym.env.world import canonical


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def fingerprints(target, initial_world, event_rules, env_version, *, identity_map=None):
    payload = dict(target=target,initial_world=initial_world,event_rules=event_rules,env_version=env_version)
    # The caller supplies semantic inputs, never run bookkeeping or seed metadata.
    exact = digest(payload)
    mapping = identity_map or {}
    if len(set(mapping.values())) != len(mapping):
        raise ValueError('identity normalization must not merge distinct entities')
    def normalize(value):
        if isinstance(value,dict):
            return {mapping.get(key,key):normalize(item) for key,item in value.items()}
        if isinstance(value,list):
            return [normalize(item) for item in value]
        if isinstance(value,str):
            return mapping.get(value,value)
        return value
    return {'exact_fingerprint':exact,'family_fingerprint':digest(normalize(payload))}


def validate_unique(entries):
    identities, exact = set(), {}
    for entry in entries:
        identity = entry['instance_id']
        if identity in identities:
            raise ValueError('duplicate instance_id: '+identity)
        identities.add(identity)
        fingerprint = entry['exact_fingerprint']
        if fingerprint in exact:
            previous = exact[fingerprint]
            raise ValueError('exact duplicate: '+previous['instance_id']+' / '+identity+' ('+previous['split']+' / '+entry['split']+')')
        exact[fingerprint] = entry
    return {'unique_instances':len(identities),'exact_duplicates':0}


def dependency_depth(nodes, edges):
    if len(set(nodes)) != len(nodes):
        raise ValueError('duplicate DAG node')
    parents = {node:set() for node in nodes}
    for source,target in edges:
        if source not in parents or target not in parents:
            raise ValueError('unknown DAG node')
        parents[target].add(source)
    depths = {}
    visiting = set()
    def visit(node):
        if node in visiting:
            raise ValueError('cyclic dependency graph')
        if node not in depths:
            visiting.add(node)
            depths[node] = 1 + max((visit(parent) for parent in parents[node]),default=0)
            visiting.remove(node)
        return depths[node]
    return max((visit(node) for node in nodes),default=0)


def check_holdout(event_signature, forbidden_signatures):
    # Event order is substantive: signatures are ordered, not sets.
    signature = tuple(event_signature)
    if signature in {tuple(item) for item in forbidden_signatures}:
        raise ValueError('held-out event combination in training instance')
