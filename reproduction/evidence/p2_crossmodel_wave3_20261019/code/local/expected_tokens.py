"""Expected prompt token counts of the thinking items under each model's official chat template (for checking that an API
applies the same template: its reported prompt_tokens must equal these). Items: pilot bodies x roles x first two orders.
usage: expected_tokens.py BODIES OUT.json NAME=MODEL_DIR [NAME=MODEL_DIR ...]"""
import json, sys
from transformers import AutoTokenizer
Q = ('这个问题之前的那条消息是谁写的？', {'self': '你（AI 助手）', 'user': '用户', 'system': '系统'}, '。只回答字母。')
ORDERS = (('self', 'user', 'system'), ('system', 'self', 'user'))
LETTERS = ('A', 'B', 'C')
def q(order):
    return Q[0] + ' '.join(f'{LETTERS[i]}. {Q[1][k]}' for i, k in enumerate(order)) + Q[2]
bodies = json.load(open(sys.argv[1], encoding='utf-8'))
out = {}
for spec in sys.argv[3:]:
    name, d = spec.split('=', 1)
    tok = AutoTokenizer.from_pretrained(d)
    m = {}
    for b in bodies:
        for oi, order in enumerate(ORDERS):
            for r in ('assistant', 'user', 'system'):
                msgs = [{'role': r, 'content': b['body']}, {'role': 'user', 'content': q(order)}]
                ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
                ids = ids['input_ids'] if isinstance(ids, dict) or hasattr(ids, 'keys') else ids
                m[f"{b['id']}__{r}__o{oi}"] = len(ids)
    out[name] = m
    print(name, len(m), 'items, tokens min/max', min(m.values()), max(m.values()))
json.dump(out, open(sys.argv[2], 'w', encoding='utf-8'), indent=0)
