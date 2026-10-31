"""Exact-source, text-preserving generation annotations for Qwen3-2507 SFT."""
import hashlib

QWEN3_2507_SOURCE_SHA256='64f85b198065d0fba2a81f37e10ed68161ce2c19a754c7100e67e0ca2ee9c326'
ADAPTER_VERSION='qwen3-2507-assistant-generation-v1'


def annotated_template(template):
    if not isinstance(template,str) or hashlib.sha256(template.encode()).hexdigest()!=QWEN3_2507_SOURCE_SHA256:
        return None
    start="        {{- '<|im_start|>' + message.role + '\\n' + content }}"
    stop="        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}"
    if template.count(start)!=1 or template.count(stop)!=1:
        raise ValueError('locked Qwen3 template annotation boundary mismatch')
    template=template.replace(start,"        {{- '<|im_start|>' + message.role + '\\n' }}\n        {%- generation %}\n        {{- content }}")
    return template.replace(stop,"        {{- '<|im_end|>' }}\n        {%- endgeneration %}\n        {{- '\\n' }}\n    {%- elif message.role == \"tool\" %}")
