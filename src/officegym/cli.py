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
    generate = commands.add_parser('generate',help='Generate implemented templates with verified reference artifacts')
    generate.add_argument('--out',required=True)
    generate.add_argument('--templates',default='T01,T02,T03,T04,T05,T06,T07,T08')
    generate.add_argument('--count-per-template',type=int,default=3)
    generate.add_argument('--seed',type=int,default=None)
    generate.add_argument('--split',default='probe')
    reference = commands.add_parser('reference',help='Re-execute public reference solutions and verify final hashes')
    reference.add_argument('--manifest',required=True)
    reference.add_argument('--out',default='artifacts/reference_replay')
    args = parser.parse_args()
    if args.command == 'reference':
        from officegym.tasks.replay import reference_manifest
        print(json.dumps(reference_manifest(args.manifest,args.out),indent=2))
    if args.command == 'generate':
        from officegym.tasks.generate import generate
        manifest = generate(args.out,args.templates.split(','),args.count_per_template,args.seed,args.split)
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
