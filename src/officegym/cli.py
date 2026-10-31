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
    args = parser.parse_args()
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
