"""Fixed first-version template partitions, seed ranges and ordered holdout."""
TRAIN_TEMPLATES = ('T01','T02','T03','T04','T05','T06','T07','T08')
DEV_TEMPLATES = ('D01','D02')
UNSEEN_TEMPLATES = ('U01','U02','U03','U04')
SUPPORTED = TRAIN_TEMPLATES + DEV_TEMPLATES + UNSEEN_TEMPLATES
SEED_RANGES = {'train':(100000,199999),'dev':(200000,299999),'probe':(900000,909999),'unseen_test':(500000,599999),'id_test':(300000,399999),'combo_test':(400000,499999)}
HELDOUT_SIGNATURES = [('AttendeeChanged','DocumentVersionPublished')]


def validate_partition(template,split,seed=None):
    allowed = {'train':TRAIN_TEMPLATES,'dev':DEV_TEMPLATES,'id_test':TRAIN_TEMPLATES,
               'combo_test':TRAIN_TEMPLATES,'unseen_test':UNSEEN_TEMPLATES,'probe':SUPPORTED}
    if split not in allowed or template not in allowed[split]:
        raise ValueError('template is not permitted in dataset split')
    if split!='probe' or seed is not None:
        low,high=SEED_RANGES[split]
        if type(seed) is not int or not low<=seed<=high:
            raise ValueError('task seed is outside its dataset split range')
