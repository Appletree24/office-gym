"""Assemble a complete-sized development dataset; freezing is a separate gate."""
import json
from pathlib import Path
from officegym.env.world import canonical
from .generate import generate,TRAIN_TEMPLATES,HELDOUT_SIGNATURES
from .reference import REFERENCE_VERSION
from officegym.tools.schema import TOOL_SCHEMA_VERSION
from .integrity import validate_unique
from .validate import validate_dataset


def assemble(config_path,out):
    config = json.loads(Path(config_path).read_text())
    counts = config['counts']
    for split,divisor in (('train',8),('dev',2),('id_test',5),('unseen_test',4)):
        if type(counts[split]) is not int or counts[split]<divisor or counts[split]%divisor:
            raise ValueError('counts must divide template/condition cells: '+split)
    if counts['train']%40:
        raise ValueError('train count must balance eight templates and five T04 conditions')
    if type(counts['combo_test']) is not int or counts['combo_test']<1:
        raise ValueError('positive combo count required')
    if sum(counts[split] for split in ('train','dev','id_test','combo_test','unseen_test'))!=counts['total']:
        raise ValueError('dataset total inconsistent')
    root = Path(out)
    root.mkdir(parents=True,exist_ok=False)
    entries = []
    def add(name,**kwargs):
        manifest = generate(root/name,**kwargs)
        for entry in manifest['instances']:
            entry = dict(entry,path=name+'/'+entry['path'])
            entries.append(entry)
    add('train_atomic',templates=tuple(template for template in TRAIN_TEMPLATES if template!='T04'),count_per_template=counts['train']//8,split='train')
    for depth,perturbation in ((1,1),(2,1),(3,1),(2,0),(2,2)):
        add('train_t04_d'+str(depth)+'p'+str(perturbation),templates=('T04',),count_per_template=counts['train']//40,split='train',depth_level=depth,perturbation_level=perturbation)
    add('dev',templates=('D01','D02'),count_per_template=counts['dev']//2,split='dev')
    for depth,perturbation in ((1,1),(2,1),(3,1),(2,0),(2,2)):
        add('id_d'+str(depth)+'p'+str(perturbation),templates=('T04',),count_per_template=counts['id_test']//5,split='id_test',depth_level=depth,perturbation_level=perturbation)
    add('combo',templates=('T07',),count_per_template=counts['combo_test'],split='combo_test',heldout_combo=True)
    add('unseen',templates=('U01','U02','U03','U04'),count_per_template=counts['unseen_test']//4,split='unseen_test')
    validate_unique(entries)
    manifest = dict(status='development_dataset_not_frozen',task_generator_version='generator-v3-controlled-width',reference_solver_version=REFERENCE_VERSION,tool_schema_version=TOOL_SCHEMA_VERSION,instances=entries,heldout_signatures=HELDOUT_SIGNATURES,
                    limitations=['Combo split uses one template and two perturbations: its held-out signature requires two events, so P0/P1 cells are inapplicable; interpret only within this composition.',
                                 'ID scan currently uses T04 composition only; interpret within this composition.',
                                 'DAG width is 1 at controlled D1 and 2 at D2/D3; stratify by width and do not claim a pure depth effect.',
                                 'Depth dependence requires additional skip-stage adversarial audit before freezing.'])
    (root/'manifest.json').write_text(canonical(manifest)+'\n')
    report = validate_dataset(root/'manifest.json')
    (root/'validation.json').write_text(canonical(report)+'\n')
    return report
