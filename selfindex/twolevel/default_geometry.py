"""Message-content geometry (run label L): the role message in the chat template's default rendering, 150 bodies x 3 roles.

Each condition is the role message exactly as the template renders it in a conversation (enable_thinking=False, default
settings): '<|im_start|>' + role + newline [+ the template's history formatting for Assistant] + body + '<|im_end|>' + newline.
The fragment is checked to be the exact token prefix of the template's rendering of [role message, user message] + generation
prompt, so every state is that of a native conversation (causal model). Only native role words; one format (the template
default) throughout.

Captured per condition at every decoder block (base.Collector): Q/K/V, the FFN output before the residual add, and the post-block
residual. Positions: input position 1 (role word), 2 (newline after the role word), and 20 endpoint-inclusive body samples chosen
by body-token index (the same indexes for all roles). Full-vocabulary readout at the 20 body positions for every block (last
block = the model's own logits, agreement checked; middle blocks diagnostic). Mode: teacher-forced single forward, no generation,
no cache.
Outputs: identity.json and word_embeddings.npz (shard 0 only); per body features/<body>__<role>.npz (input_token_ids, positions,
one [22, width] uint16 array of BF16 bits per component and block), lens/<body>.npz (base.lens_metrics), bodies/<body>.json
(renderings, positions, timings, receipts); status_<k>.json. Every .npz has a .receipt.json.
usage: python -m selfindex.twolevel.default_geometry --model DIR --out DIR [--shard K/N] [--max-memory-gib G]
       python -m selfindex.twolevel.default_geometry --model DIR --check     (tokenizer-only check of every rendering)
"""
from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path

import numpy as np

from .. import __version__
from ..render import ids_of
from . import base as B

ROLES = B.ROLES
TRIPLETS = [{'name': 'native', 'kind': 'native_role_default_template', 'native': True,
             'members': {'A': 'assistant', 'U': 'user', 'S': 'system'}}]
PREFIX_CHECK = 'PREFIX_CHECK_7731'   # user turn after the message, only to render the conversation the fragment must prefix


def default_fragment(tok, role, body):
    """Template-default rendering of the role message, checked to be a token prefix of a native conversation."""
    msgs = [{'role': role, 'content': body}, {'role': 'user', 'content': PREFIX_CHECK}]
    full = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    end = full.index('<|im_end|>\n', full.index(body)) + len('<|im_end|>\n')
    text = full[:end]
    ids = ids_of(tok(text, add_special_tokens=False))
    full_ids = ids_of(tok(full, add_special_tokens=False))
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
        samples = B.uniform_positions(len(body_pos))
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


def run(args, loaded=None):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    L_ = B.load(args, loaded)
    torch, tok, model = L_.torch, L_.tok, L_.model
    layers, norm, head, embed = B.model_parts(model)
    n_layers = len(layers)
    layer_types = list(model.config.text_config.layer_types)
    bodies = B.all_bodies()
    k, n = (int(x) for x in args.shard.split('/'))
    mine = [b for i, b in enumerate(bodies) if i % n == k]
    words = list(ROLES)
    word_ids = {w: B.single_token_header(tok, w)[1] for w in words}
    if k == 0:
        with torch.no_grad():
            ids = torch.tensor([word_ids[w] for w in words], device=embed.weight.device)
            emb = embed(ids).float().cpu().numpy()
            unemb = head.weight[ids.to(head.weight.device)].float().cpu().numpy()
        B.write_arrays(out / 'word_embeddings.npz', words=np.asarray(words), word_ids=np.asarray([word_ids[w] for w in words]),
                       input_embedding=emb, output_unembedding=unemb)
        B.write_json(out / 'identity.json', {
            'experiment': 'P2_default_template_geometry_L', 'selfindex_version': __version__, 'model_path': args.model,
            'config_sha256': B.sha_file(Path(args.model) / 'config.json'), 'device_map': L_.placement,
            'torch': torch.__version__, 'transformers': __import__('transformers').__version__, 'layer_types': layer_types,
            'mode': 'native_template_default_fragment_teacher_forced_forward_no_generation_no_cache',
            'triplets': TRIPLETS, 'words': words, 'word_ids': word_ids, 'bodies': [b['id'] for b in bodies],
            'positions': 'index0=role word (input pos 1), index1=newline (pos 2), index2..21 = 20 body samples by body-token index',
            'lens': f'uncalibrated logit lens diagnostic; layer {n_layers - 1} = native same-forward logits'})
    collector = B.Collector(torch, layers, layer_types)
    first = embed.weight.device
    started = time.time()
    try:
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
                if len(collector.features) != n_layers * 5:
                    raise ValueError(f'expected {n_layers * 5} component-layer captures')
                if delta >= 1e-4:
                    raise ValueError(('native final lens disagreement', delta))
                feats[role] = dict(collector.features)
                natives[role] = native.float().cpu()
                receipt = B.write_arrays(out / 'features' / f"{row['id']}__{role}.npz", input_token_ids=np.asarray(ids_list, np.int64),
                                         positions=np.asarray(pos, np.int64), **feats[role])
                timings.append({'word': role, 'seconds': time.time() - t0, 'native_final_lens_max_abs_delta': delta,
                                'bytes': receipt['bytes']})
            observed = conds['assistant']['observed_next_token_ids'][2:]
            metrics = B.lens_metrics(torch, norm, head, feats, natives, words, TRIPLETS, observed, n_layers)
            lens_receipt = B.write_arrays(out / 'lens' / f"{row['id']}.npz", words=np.asarray(words), **metrics)
            B.write_json(done, {'state': 'complete', 'id': row['id'], 'split': row['split'], 'category': row['category'],
                                'template_group': row['template_group'], 'conditions': conds, 'timings': timings,
                                'lens_receipt': lens_receipt, 'layer_collection': f'all_{n_layers}_blocks_5_components_22_positions'})
            B.write_json(out / f'status_{k}.json', {'state': 'running', 'bodies_done': b_i + 1, 'bodies_planned': len(mine),
                                                    'elapsed_s': time.time() - started, 'heartbeat_epoch': time.time()})
    finally:
        collector.remove()
    B.write_json(out / f'status_{k}.json', {'state': 'complete', 'bodies_planned': len(mine), 'elapsed_s': time.time() - started})


def check(args):
    """Tokenizer only: render every body under the three roles and print the body start positions."""
    tok = B.tokenizer(args.model)
    rows = B.all_bodies()
    starts = set()
    for row in rows:
        conds = render(tok, row)
        starts.add(tuple(conds[r]['body_start'] for r in ROLES))
    print('ok', len(rows), 'bodies; body start (assistant, user, system):', starts)
    print('assistant fragment head:', tok.convert_ids_to_tokens(conds['assistant']['input_token_ids'][:8]))


def parser():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--model', required=True)
    p.add_argument('--out', default='')
    p.add_argument('--shard', default='0/1', help='K/N: run the bodies whose index i in the fixed order has i %% N == K')
    p.add_argument('--max-memory-gib', type=float, default=None, help='memory cap per GPU (default: what accelerate finds free)')
    p.add_argument('--check', action='store_true', help='tokenizer-only check of every rendering, no model')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.check:
        return check(args)
    try:
        run(args)
    except Exception as exc:
        k = args.shard.split('/')[0]
        B.write_json(Path(args.out) / f'status_{k}.json', {'state': 'failed', 'error': str(exc), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
