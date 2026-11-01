"""Synthetic review attestations for software tests only, never human evidence."""
import json
from officegym.env.world import canonical
from officegym.training.mask_review import prepare
from officegym.training.review_admission import register,CHECKS
from test_mask_review import fixture


def reviewed_fixture(data,metadata=None):
    tokenizer=fixture(data)
    manifest=json.loads((data/'manifest.json').read_text())
    # Preserve lineage/branch fields while using valid authored alignment vectors.
    if metadata:
        extra={key:value for key,value in metadata.items() if key not in ('accepted','samples_sha256','manual_alignment_review_status')}
        manifest.update(extra)
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
