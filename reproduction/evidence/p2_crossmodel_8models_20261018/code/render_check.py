import sys, json
from transformers import AutoTokenizer
for d in sys.argv[1:]:
    t = AutoTokenizer.from_pretrained(d)
    print('=====', d.split('/')[-1])
    for kw in ({'enable_thinking': False}, {}):
        for role in ('assistant', 'user', 'system'):
            msgs = [{'role': role, 'content': 'BODY'}, {'role': 'user', 'content': 'Q'}]
            try:
                s = t.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
            except Exception as e:
                s = 'ERR ' + str(e)[:80]
            print(json.dumps(kw), role, repr(s))
