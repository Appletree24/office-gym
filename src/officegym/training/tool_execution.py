"""Match evaluation's single-call protocol and bill all attempted tool calls."""
import json


def dispatch_generated_call(environment,name,arguments):
    # Invalid JSON still passes through the canonical dispatcher, consuming the
    # same tool/time budget and producing the same structured error as evaluation.
    try:
        parsed = json.loads(arguments)
    except (ValueError,TypeError):
        parsed = None
    return environment.call(name,parsed)


def single_call(calls):
    return len(calls)==1
