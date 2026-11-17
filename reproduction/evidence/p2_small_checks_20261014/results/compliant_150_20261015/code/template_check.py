"""Read-only: compare the experiment's fragment inputs with the model's own chat-template renderings."""
import glob
import json
import numpy as np
from transformers import AutoTokenizer

M = '/root/data/hf_models/Qwen/Qwen3.8-27B'
tok = AutoTokenizer.from_pretrained(M, local_files_only=True)
for s in ('<|im_start|>', '<|im_end|>', '<think>', '</think>', 'assistant', 'user', 'system'):
    print(f'{s!r:16} -> {tok.encode(s, add_special_tokens=False)}')
USER_TURN = '请根据上文回应。'
runs = {'reply_noturn': ('behavior_reply', False, None), 'think_noturn': ('behavior_think', True, None),
        'reply_turn': ('turn_reply', False, USER_TURN), 'think_turn': ('turn_think', True, USER_TURN)}
for name, (run, thinking, ut) in runs.items():
    bid = json.load(open(f'{run}/identity.json'))['bodies'][0]
    body = json.load(open(f'{run}/bodies/{bid}.json'))['body']
    print(f'\n===== {name}')
    for role in ('assistant', 'user', 'system'):
        ours = np.load(f'{run}/features/{bid}__{role}.npz')['input_token_ids'].tolist()
        msgs = [{'role': role, 'content': body}] + ([{'role': 'user', 'content': ut}] if ut else [])
        for pt in (False, True):
            try:
                ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                              enable_thinking=thinking, preserve_thinking=pt)
                ids = ids['input_ids'] if hasattr(ids, 'keys') else ids
                ids = list(ids[0]) if ids and isinstance(ids[0], list) else list(ids)
            except Exception as exc:
                print(f'  {role:9s} preserve_thinking={pt}: TEMPLATE ERROR {type(exc).__name__}: {str(exc)[:80]}')
                continue
            same = ids == ours
            if same:
                print(f'  {role:9s} preserve_thinking={pt}: IDENTICAL ({len(ids)} tokens)')
            else:
                i = next((k for k, (a, b) in enumerate(zip(ids, ours)) if a != b), min(len(ids), len(ours)))
                print(f'  {role:9s} preserve_thinking={pt}: DIFFERS at token {i} (native {len(ids)} vs ours {len(ours)})')
                print('     native:', tok.convert_ids_to_tokens(ids[max(0, i-3):i+6]))
                print('     ours  :', tok.convert_ids_to_tokens(ours[max(0, i-3):i+6]))
