"""P2 check L: role geometry of message content in the chat template's default rendering (150 bodies, native roles).

Each condition is the role message exactly as the template renders it in a conversation (enable_thinking=False,
default settings): '<|im_start|>' + role + '\\n' [+ the template's history formatting for Assistant] + body +
'<|im_end|>\\n'. The fragment is asserted to be the exact token prefix of the template's rendering of
[role message, user message] + generation prompt, so every state is that of a native conversation (causal model).
Only native role words; no substituted headers; one format (the template default) throughout.

Captured per condition at all 64 blocks (same definitions as D3 / the pilot collector): Q/K/V, FFN output before the
residual add, post-block residual. Positions: input position 1 (role word), 2 (newline after the role word), and
20 endpoint-inclusive body samples chosen by body-token index (same indexes for all roles). Full-vocabulary readout
at the 20 body positions for all 64 layers (layer 63 = native logits, agreement checked; middle layers diagnostic).
Mode: teacher-forced single forward, no generation, no cache. Output format = the pilot collector's, so
p2_geometry_analysis.py --source pilot reads it unchanged.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2_null_pilot as base  # noqa: E402
from p2_body_sets import all_bodies  # noqa: E402

ROLES = ('assistant', 'user', 'system')
TRIPLETS = [{'name': 'native', 'kind': 'native_role_default_template', 'native': True,
             'members': {'A': 'assistant', 'U': 'user', 'S': 'system'}}]


def default_fragment(tok, role, body):
    """Template-default rendering of the role message, asserted to be a token prefix of a native conversation."""
    msgs = [{'role': role, 'content': body}, {'role': 'user', 'content': 'PREFIX_CHECK_7731'}]
    full = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    end = full.index('<|im_end|>\n', full.index(body)) + len('<|im_end|>\n')
    text = full[:end]
    ids = base.ids_of(tok(text, add_special_tokens=False))
    full_ids = base.ids_of(tok(full, add_special_tokens=False))
    assert full_ids[:len(ids)] == ids, (role, 'fragment is not a token prefix of the native conversation')
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc['offset_mapping']
    if offsets and isinstance(offsets[0][0], (list, tuple)):
        offsets = offsets[0]
    s = text.index(body)
    e = s + len(body)
    body_pos = [i for i, (a, b) in enumerate(offsets) if b > s and a < e and b > a]
    assert body_pos == list(range(body_pos[0], body_pos[-1] + 1))
    assert all(offsets[i][0] >= s and offsets[i][1] <= e for i in body_pos), 'body/format token merge'
    return text, ids, body_pos


def render(tok, row):
    conds = {}
    for role in ROLES:
        text, ids, body_pos = default_fragment(tok, role, row['body'])
        toks = tok.convert_ids_to_tokens(ids)
        assert toks[0] == '<|im_start|>' and toks[1] == role and toks[2] == 'Ċ' and toks[-2] == '<|im_end|>', toks[:4]
        samples = base.uniform_positions(len(body_pos))
        positions = [1, 2] + [body_pos[i] for i in samples]
        conds[role] = {'word': role, 'input_token_ids': ids, 'positions': positions, 'body_start': body_pos[0],
                       'body_relative_sample_indexes': samples, 'body_token_count': len(body_pos),
                       'observed_next_token_ids': [ids[p + 1] for p in positions]}
    ref = conds['assistant']
    for role in ROLES[1:]:
        c = conds[role]
        assert [c['input_token_ids'][p] for p in c['positions'][2:]] == [ref['input_token_ids'][p] for p in ref['positions'][2:]]
        assert c['observed_next_token_ids'][2:] == ref['observed_next_token_ids'][2:], 'next-token targets differ'
        assert c['body_relative_sample_indexes'] == ref['body_relative_sample_indexes']
    return conds


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch, tok, model, layers, norm, head, embed, placement, dtypes = base.load_model(args)
    layer_types = list(model.config.text_config.layer_types)
    bodies = all_bodies(args.bodies)
    k, n = (int(x) for x in args.shard.split('/'))
    mine = [b for i, b in enumerate(bodies) if i % n == k]
    words = list(ROLES)
    word_ids = {w: base.single_token_header(tok, w)[1] for w in words}
    if k == 0:
        with torch.no_grad():
            ids = torch.tensor([word_ids[w] for w in words], device=embed.weight.device)
            emb = embed(ids).float().cpu().numpy()
            unemb = head.weight[ids.to(head.weight.device)].float().cpu().numpy()
        base.arrays(out / 'word_embeddings.npz', words=np.asarray(words), word_ids=np.asarray([word_ids[w] for w in words]),
                    input_embedding=emb, output_unembedding=unemb)
        base.atomic(out / 'identity.json', {
            'experiment': 'P2_default_template_geometry_L', 'script_sha256': base.sha_file(__file__),
            'base_sha256': base.sha_file(base.__file__), 'model_path': args.model,
            'config_sha256': base.sha_file(Path(args.model) / 'config.json'), 'device_map': placement,
            'torch': torch.__version__, 'transformers': __import__('transformers').__version__,
            'host': __import__('socket').gethostname(), 'layer_types': layer_types,
            'mode': 'native_template_default_fragment_teacher_forced_forward_no_generation_no_cache',
            'triplets': TRIPLETS, 'words': words, 'word_ids': word_ids, 'bodies': [b['id'] for b in bodies],
            'positions': 'index0=role word (input pos 1), index1=newline (pos 2), index2..21 = 20 body samples by body-token index',
            'lens': 'uncalibrated logit lens diagnostic; layer 63 = native same-forward logits'})
    collector = base.Collector(torch, layers, layer_types)
    first = embed.weight.device
    started = time.time()
    for b_i, row in enumerate(mine):
        done = out / 'bodies' / (row['id'] + '.json')
        if done.exists():
            continue
        conds = render(tok, row)
        feats, natives, timings = {}, {}, []
        for role in words:
            c = conds[role]
            ids_list, pos = c['input_token_ids'], c['positions']
            collector.reset(pos)
            t0 = time.time()
            x = torch.tensor([ids_list], dtype=torch.long, device=first)
            offs = torch.arange(len(ids_list), device=first)
            with torch.no_grad():
                o = model(input_ids=x, attention_mask=torch.ones_like(x), position_ids=offs.view(1, 1, -1).expand(4, 1, -1),
                          use_cache=False, return_dict=True, logits_to_keep=0)
                native = o.logits[0, pos[2:]].detach()
                rep = head(norm(collector.tail.to(next(norm.parameters()).device)).to(next(head.parameters()).device))[0, pos[2:]]
                delta = float((rep.float().to(native.device) - native.float()).abs().max())
            del o
            if len(collector.features) != base.BLOCKS * 5:
                raise ValueError('expected 320 component-layer captures')
            if delta >= 1e-4:
                raise ValueError(('native final lens disagreement', delta))
            feats[role] = dict(collector.features)
            natives[role] = native.float().cpu()
            receipt = base.arrays(out / 'features' / f"{row['id']}__{role}.npz", input_token_ids=np.asarray(ids_list, np.int64),
                                  positions=np.asarray(pos, np.int64), **feats[role])
            timings.append({'word': role, 'seconds': time.time() - t0, 'native_final_lens_max_abs_delta': delta, 'bytes': receipt['bytes']})
        observed = conds['assistant']['observed_next_token_ids'][2:]
        metrics = base.lens_metrics(torch, norm, head, feats, natives, words, TRIPLETS, observed)
        lens_receipt = base.arrays(out / 'lens' / f"{row['id']}.npz", words=np.asarray(words), **metrics)
        base.atomic(done, {'state': 'complete', 'id': row['id'], 'split': row['split'], 'category': row['category'],
                           'template_group': row['template_group'], 'conditions': conds, 'timings': timings,
                           'lens_receipt': lens_receipt, 'layer_collection': 'all_64_blocks_5_components_22_positions'})
        base.atomic(out / f'status_{k}.json', {'state': 'running', 'bodies_done': b_i + 1, 'bodies_planned': len(mine),
                                              'elapsed_s': time.time() - started, 'heartbeat_epoch': time.time()})
    base.atomic(out / f'status_{k}.json', {'state': 'complete', 'bodies_planned': len(mine), 'elapsed_s': time.time() - started})


def check(args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
    rows = all_bodies(args.bodies)
    starts = set()
    for row in rows:
        conds = render(tok, row)
        starts.add(tuple(conds[r]['body_start'] for r in ROLES))
    print('ok', len(rows), 'bodies; body start (assistant, user, system):', starts)
    print('assistant fragment head:', tok.convert_ids_to_tokens(conds['assistant']['input_token_ids'][:8]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--bodies', required=True)
    p.add_argument('--out', default='')
    p.add_argument('--shard', default='0/1')
    p.add_argument('--max-memory-gib', type=float, default=38)
    p.add_argument('--check', action='store_true', help='tokenizer-only compliance check, no model')
    args = p.parse_args()
    if args.check:
        return check(args)
    try:
        run(args)
    except Exception as exc:
        k = args.shard.split('/')[0]
        base.atomic(Path(args.out) / f'status_{k}.json', {'state': 'failed', 'error': str(exc), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
