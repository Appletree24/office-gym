"""OfficeGym command entry point."""
import argparse
import json


def main():
    parser = argparse.ArgumentParser(prog='officegym')
    commands = parser.add_subparsers(dest='command',required=True)
    probe = commands.add_parser('probe-reference',help='Run hand-designed CPU integration probes')
    probe.add_argument('--out',required=True)
    server = commands.add_parser('serve-world',help='Serve one fixed episode over a private Unix socket')
    server.add_argument('--world',required=True)
    server.add_argument('--socket',required=True)
    server.add_argument('--max-calls',type=int,default=None)
    fault = commands.add_parser('probe-fault',help='Run actual worker-kill recovery software probe')
    fault.add_argument('--out',required=True)
    graders = commands.add_parser('test-graders',help='Run labelled positive and counterexample cases')
    graders.add_argument('--suite',required=True)
    graders.add_argument('--out',default='artifacts/grader_results')
    cases = commands.add_parser('generate-grader-cases',help='Generate labelled tool-executed software cases')
    cases.add_argument('--out',required=True)
    dataset = commands.add_parser('validate-dataset',help='Validate concrete dataset files and reference artifacts')
    dataset.add_argument('--manifest',required=True)
    generate = commands.add_parser('generate',allow_abbrev=False,help='Generate implemented templates with verified reference artifacts')
    generate.add_argument('--out',required=True)
    generate.add_argument('--config',help='Assemble the full configured development distribution')
    generate.add_argument('--templates',default='T01,T02,T03,T04,T05,T06,T07,T08')
    generate.add_argument('--count-per-template',type=int,default=3)
    generate.add_argument('--seed',type=int,default=None)
    generate.add_argument('--depth-level',type=int)
    generate.add_argument('--perturbation-level',type=int)
    generate.add_argument('--heldout-combo',action='store_true')
    generate.add_argument('--split',default='probe')
    reference = commands.add_parser('reference',help='Re-execute public reference solutions and verify final hashes')
    reference.add_argument('--manifest',required=True)
    reference.add_argument('--out',default='artifacts/reference_replay')
    mcp = commands.add_parser('serve-mcp',help='Official MCP stdio adapter to running world service')
    mcp.add_argument('--socket',required=True)
    single = commands.add_parser('evaluate-instance',help='Run one configured local model episode with finite budgets')
    single.add_argument('--instance',required=True)
    single.add_argument('--config',required=True)
    single.add_argument('--out',required=True)
    sft = commands.add_parser('build-sft',help='Build audited assistant-only training samples')
    sft.add_argument('--runs',required=True)
    sft.add_argument('--out',required=True)
    sft.add_argument('--tokenizer',required=True)
    sft.add_argument('--revision',required=True)
    sft.add_argument('--max-length',type=int,default=8192)
    sft.add_argument('--reject-compacted',action='store_true',default=True)
    sft.add_argument('--verify-masks',action='store_true',default=True)
    teacher = commands.add_parser('collect-teacher',help='Collect finite-budget training demonstrations')
    teacher.add_argument('--manifest',required=True)
    teacher.add_argument('--config',required=True)
    teacher.add_argument('--out',required=True)
    teacher.add_argument('--split',choices=['train'],default='train')
    teacher.add_argument('--max-attempts-per-task',type=int,default=3)
    teacher.add_argument('--max-tasks',type=int,default=40)
    teacher.add_argument('--selection-seed',default='officegym-teacher-pilot-v1')
    teacher.add_argument('--max-episodes',type=int,required=True)
    teacher.add_argument('--max-generated-tokens',type=int,required=True)
    batch = commands.add_parser('evaluate',help='Evaluate selected runs from a hashed plan')
    batch.add_argument('--manifest',required=True)
    batch.add_argument('--plan',required=True)
    batch.add_argument('--config',required=True)
    batch.add_argument('--out',required=True)
    batch.add_argument('--experiment',required=True)
    batch.add_argument('--split',required=True)
    batch.add_argument('--resume',action='store_true')
    batch.add_argument('--max-runs',type=int,required=True)
    batch.add_argument('--max-generated-tokens',type=int,required=True)
    train = commands.add_parser('train-sft',help='Run locked TRL SFT on verified pretokenized data')
    train.add_argument('--config',required=True)
    train.add_argument('--data',required=True)
    train.add_argument('--out',required=True)
    assembled = commands.add_parser('assemble-dataset',help='Assemble configured development dataset, not frozen')
    assembled.add_argument('--config',required=True)
    assembled.add_argument('--out',required=True)
    run_plan = commands.add_parser('freeze-run-plan',help='Write immutable run keys/checksum; does not freeze or admit the dataset')
    run_plan.add_argument('--manifest',required=True)
    run_plan.add_argument('--out',required=True)
    run_plan.add_argument('--priority',choices=('P0','P1'),default='P0')
    run_plan.add_argument('--splits',default='id_test,combo_test,unseen_test')
    operator = commands.add_parser('check-operator-inputs',help='Check teacher and total resource choices without reading credentials or launching jobs')
    operator.add_argument('--config',required=True)
    args = parser.parse_args()
    if args.command == 'check-operator-inputs':
        from officegym.training.operator_inputs import check_operator_inputs
        report=check_operator_inputs(args.config)
        print(json.dumps(report,indent=2))
        if not report['operator_inputs_ready']:raise SystemExit(2)
        return
    if args.command == 'freeze-run-plan':
        from pathlib import Path
        from officegym.tasks.validate import validate_dataset
        from officegym.eval.plan import build_plan,freeze_plan
        validate_dataset(args.manifest)
        manifest=json.loads(Path(args.manifest).read_text())
        rows=build_plan(manifest['instances'],priority=args.priority,splits=tuple(args.splits.split(',')))
        digest=freeze_plan(rows,args.out)
        print(json.dumps(dict(planned_runs=len(rows),sha256=digest,dataset_status=manifest['status'],scope='Run-key plan only; dataset admission and experiment chronology require separate evidence.')))
        return
    if args.command == 'assemble-dataset':
        from officegym.tasks.assemble import assemble
        print(json.dumps(assemble(args.config,args.out),indent=2))
    if args.command == 'train-sft':
        from officegym.training.sft import train
        print(json.dumps(train(args.config,args.data,args.out),indent=2))
    if args.command == 'evaluate':
        from officegym.eval.batch import evaluate_batch
        print(json.dumps(evaluate_batch(args.manifest,args.plan,args.config,args.out,group=args.experiment,split=args.split,max_runs=args.max_runs,max_generated_tokens=args.max_generated_tokens,resume=args.resume),indent=2))
    if args.command == 'collect-teacher':
        from officegym.training.collect import collect
        print(json.dumps(collect(args.manifest,args.config,args.out,max_attempts_per_task=args.max_attempts_per_task,max_episodes=args.max_episodes,max_generated_tokens=args.max_generated_tokens,max_tasks=args.max_tasks,selection_seed=args.selection_seed),indent=2))
    if args.command == 'build-sft':
        from officegym.training.build import build_from_files
        print(json.dumps(build_from_files(args.runs,args.out,args.tokenizer,args.revision,args.max_length),indent=2))
    if args.command == 'evaluate-instance':
        from officegym.eval.configured import evaluate_instance
        print(json.dumps(evaluate_instance(args.instance,args.out,args.config),indent=2))
    if args.command == 'serve-mcp':
        import asyncio
        from officegym.tools.mcp_server import run
        asyncio.run(run(args.socket))
    if args.command == 'reference':
        from officegym.tasks.replay import reference_manifest
        print(json.dumps(reference_manifest(args.manifest,args.out),indent=2))
    if args.command == 'generate' and args.config:
        import sys
        tuning={'--templates','--count-per-template','--seed','--depth-level','--perturbation-level','--heldout-combo','--split'}
        if any(arg.split('=',1)[0] in tuning for arg in sys.argv[1:]):
            parser.error('--config cannot be combined with individual template generation options')
        from officegym.tasks.assemble import assemble
        print(json.dumps(assemble(args.config,args.out),indent=2))
        return
    if args.command == 'generate':
        from officegym.tasks.generate import generate
        manifest = generate(args.out,args.templates.split(','),args.count_per_template,args.seed,args.split,heldout_combo=args.heldout_combo,depth_level=args.depth_level,perturbation_level=args.perturbation_level)
        print(json.dumps({'generated_instances':len(manifest['instances'])}))
    if args.command == 'validate-dataset':
        from officegym.tasks.validate import validate_dataset
        print(json.dumps(validate_dataset(args.manifest),indent=2))
    if args.command == 'generate-grader-cases':
        from officegym.graders.cases import generate_cases
        print(json.dumps({'generated_cases':generate_cases(args.out)}))
    if args.command == 'test-graders':
        from officegym.graders.suite import run_suite
        report = run_suite(args.suite,args.out)
        print(json.dumps(report,indent=2))
        if report['label_mismatches']:
            raise SystemExit(1)
    if args.command == 'probe-fault':
        from officegym.eval.fault_probe import run
        print(json.dumps(run(args.out),indent=2))
    if args.command == 'serve-world':
        from officegym.env.service import serve
        serve(args.world,args.socket,args.max_calls)
    if args.command == 'probe-reference':
        from officegym.tasks.probe import run_probes
        print(json.dumps(run_probes(args.out),indent=2))


if __name__ == '__main__':
    main()
