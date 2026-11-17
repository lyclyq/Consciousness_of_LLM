import sys, json, os
import jinja2
for d in sys.argv[1:]:
    p = os.path.join(d, 'chat_template.jinja')
    src = open(p).read() if os.path.exists(p) else json.load(open(os.path.join(d, 'tokenizer_config.json')))['chat_template']
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=['jinja2.ext.loopcontrols'])
    env.globals['raise_exception'] = lambda m: (_ for _ in ()).throw(Exception(m))
    tpl = env.from_string(src)
    print('=====', d.split('/')[-1])
    for kw in ({'enable_thinking': False}, {}):
        for role in ('assistant', 'user', 'system'):
            msgs = [{'role': role, 'content': 'BODY'}, {'role': 'user', 'content': 'Q'}]
            try:
                s = tpl.render(messages=msgs, add_generation_prompt=True, bos_token='', **kw)
            except Exception as e:
                s = 'ERR ' + str(e)[:100]
            print(json.dumps(kw), role, repr(s))
