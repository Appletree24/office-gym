"""Measured context selection events, without inferring missing old metrics."""
CONTEXT_METRICS_VERSION='context-metrics-v1'
FIELDS=('context_samples','context_token_sum','peak_context_tokens','compression_trigger_count',
        'summary_model_calls','summary_cache_hits','compressed_context_count','truncated_context_count')


def context_metrics(records):
    result={field:0 for field in FIELDS}
    result['context_metrics_version']=CONTEXT_METRICS_VERSION
    model_responses=0
    for record in records:
        model_responses+=record.get('kind')=='model_response'
        if record.get('kind')!='context_selection':
            continue
        if record.get('context_metrics_version')!=CONTEXT_METRICS_VERSION:
            raise ValueError('context evidence is missing current metrics version')
        tokens=record.get('input_tokens')
        if type(tokens) is not int or tokens<0:
            raise ValueError('invalid measured context token count')
        result['context_samples']+=1
        result['context_token_sum']+=tokens
        result['peak_context_tokens']=max(result['peak_context_tokens'],tokens)
        for field,flag in [('compression_trigger_count','compression_triggered'),('summary_model_calls','summary_model_call'),('summary_cache_hits','summary_cache_hit'),('compressed_context_count','compressed')]:
            result[field]+=bool(record.get(flag))
        result['truncated_context_count']+=record.get('removed_groups',0)>0
    if result['context_samples']<model_responses:
        raise ValueError('model responses lack measured context selection evidence')
    return result
