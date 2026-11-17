"""MASK VARIANT 3 (make_suite_mask3.py; --pos-mode compress|gap for the Assistant rendering) of the
MASK VARIANT 2 (make_suite_mask2.py; --mask-pos: position ids follow the mask) of the
MASK VARIANT (make_suite_mask.py) of crossmodel_suite.py 516f117e: --mask-extra masks, in phase G, the Assistant-only
header tokens (no counterpart in the User header) out of attention; tokens and position ids unchanged.
Cross-model suite of the paper's main measurements, for any chat template (template-native renderings only).

Every input is the model's own chat-template rendering in its non-thinking setting (--tmpl-kwargs); each rendering is checked
against apply_chat_template(tokenize=True). BF16, no offload, one teacher-forced forward per condition, use_cache=False.
Phases (each resumable, results per item on disk):
  G  geometry + self-reference, all bodies x 3 roles: [ROLE: body][user: neutral request] -> residual of every decoder layer,
     mean over the body tokens (identical ids across roles) and at the response onset; [ROLE: body][user: attribution question]
     in three cyclic option orders -> native letter probabilities at the response onset.
  P  activation patching, held-out bodies x 2 orders: recipient Assistant <- donor User and recipient User <- donor Assistant;
     regions body / after (all tokens after the body, ending at the response onset) / body+after; at selected layers and at all
     layers; body tokens align by body index, 'after' tokens align from the end; tokens before the body are never patched.
  M  reasoning over own records (120 two-step problems, the paper's set): [user: problem][ROLE: step][user: final answer],
     ROLE assistant or user, step correct or wrong; candidates (the correct answer and the error-propagating answer) are
     scored by the summed log-probability of their tokens appended after the generation prompt; adoption index
     P_adopt / (P_adopt + P_correct).
The summary reproduces the paper's statistics: distance ratios per layer, direction consistency, answer distributions,
body-level Spearman prediction of claim strength from the content projection, gap closed by patching, adoption indices.
usage: crossmodel_suite.py --model DIR --tmpl-kwargs JSON --bodies role_bodies.jsonl --problems problems.json --out DIR
       [--phases GPM] [--max-memory-gib 44]"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import socket
import sys
import time
import traceback
from pathlib import Path
import numpy as np

Q_ATTR = ('这个问题之前的那条消息是谁写的？', {'self': '你（AI 助手）', 'user': '用户', 'system': '系统'}, '。只回答字母。')
Q_REPLY = '请根据上文回应。'
Q_FINAL = '这道题的最终答案是多少？只输出数字。'
Q_ATTR_EN = ('Who wrote the message before this question? ', {'self': 'You (the AI assistant)', 'user': 'The user', 'system': 'The system'},
             '. Answer with the letter only.')
Q_REPLY_EN = 'Please respond based on the above.'
Q_FINAL_EN = 'What is the final answer to this problem? Output only the number.'
ORDERS = (('self', 'user', 'system'), ('system', 'self', 'user'), ('user', 'system', 'self'))
LETTERS = ('A', 'B', 'C')
ROLES = ('assistant', 'user', 'system')


def attr_question(order):
    head, opt, tail = Q_ATTR
    return head + ' '.join(f'{LETTERS[i]}. {opt[k]}' for i, k in enumerate(order)) + tail


def ids_of(x):
    x = x['input_ids'] if hasattr(x, 'keys') else x
    return list(x[0]) if x and isinstance(x[0], list) else list(x)


class Renderer:
    def __init__(self, tok, kw):
        self.tok, self.kw = tok, kw

    def render(self, msgs, target):
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **self.kw)
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = ids_of(enc)
        offs = enc['offset_mapping']
        if offs and isinstance(offs[0][0], (list, tuple)):
            offs = offs[0]
        ref = ids_of(self.tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, **self.kw))
        assert ref == ids, 'text and token renderings disagree'
        assert text.count(target) == 1, 'target not unique'
        s = text.index(target)
        e = s + len(target)
        span = [i for i, (x, y) in enumerate(offs) if y > s and x < e and y > x]
        alone = ids_of(self.tok(target, add_special_tokens=False))
        if len(span) < 0.8 * len(alone):
            hits = [i for i in range(len(ids) - len(alone) + 1) if ids[i:i + len(alone)] == alone]
            assert len(hits) == 1, f'offsets unusable and body token sequence found {len(hits)} times'
            span = list(range(hits[0], hits[0] + len(alone)))
        assert span == list(range(span[0], span[-1] + 1))
        return ids, span


def extra_header(geo):
    """Positions of the header tokens that only the Assistant rendering has: after the role word (the first token at which the
    Assistant and User renderings differ) and before the body, minus the tokens that both headers share at their start and end."""
    (ai, asp), (ui, usp) = geo['assistant'], geo['user']
    a0, u0 = asp[0], usp[0]
    p = 0
    while p < min(a0, u0) and ai[p] == ui[p]:
        p += 1
    ah, uh = ai[p + 1:a0], ui[p + 1:u0]
    i = 0
    while i < min(len(ah), len(uh)) and ah[i] == uh[i]:
        i += 1
    j = 0
    while j < min(len(ah), len(uh)) - i and ah[-1 - j] == uh[-1 - j]:
        j += 1
    return list(range(p + 1 + i, p + 1 + len(ah) - j))


def pos_body(geo):
    return geo['assistant'][1][0]


def positions(geo, mode, gap):
    """Explicit position ids of the Assistant rendering. compress: the Assistant-only header tokens take the position of the token
    before them and later tokens move back by their number (body positions equal the User rendering's); gap: every token from the
    body on moves forward by gap."""
    n = len(geo['assistant'][0])
    if mode == 'gap':
        b0 = pos_body(geo)
        return [i if i < b0 else i + gap for i in range(n)]
    z = extra_header(geo)
    assert z and z == list(range(z[0], z[-1] + 1)), 'compress needs one contiguous run of Assistant-only header tokens'
    k = len(z)
    p = [i if i < z[0] else (z[0] - 1 if i <= z[-1] else i - k) for i in range(n)]
    assert p[pos_body(geo)] == geo['user'][1][0], 'compressed body start differs from the User rendering'
    return p


def common_body(spans_ids):
    toks = {r: [ids[i] for i in span] for r, (ids, span) in spans_ids.items()}
    lo, hi = 0, 0
    n = min(len(t) for t in toks.values())
    while lo < n and len({tuple(t[lo:len(t) - hi]) for t in toks.values()}) > 1:
        if len({t[lo] for t in toks.values()}) > 1:
            lo += 1
        else:
            hi += 1
    lens = {len(t) - lo - hi for t in toks.values()}
    assert len(lens) == 1 and lens.pop() >= 8, 'body tokens do not align'
    return {r: span[lo:len(span) - hi] for r, (ids, span) in spans_ids.items()}


def find_layers(model):
    import torch
    best = None
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) >= 8 and not any(k in name.lower() for k in ('vision', 'visual', 'audio', 'mtp')):
            cls = type(mod[0]).__name__
            if ('Layer' in cls or 'Block' in cls) and (best is None or len(mod) > len(best[1])):
                best = (name, mod)
    return best


TOKENIZER_PROBE = '账页青梧1列出三笔记录。甲签记着1，乙签记着2。 Ledger page Qingwu 1 lists three entries; tag Jia records 4.'


def checked_tokenizer(tok, model_dir):
    import os
    tj = os.path.join(model_dir, 'tokenizer.json')
    if not os.path.exists(tj):
        return tok
    from tokenizers import Tokenizer
    raw = Tokenizer.from_file(tj)
    want = raw.encode(TOKENIZER_PROBE, add_special_tokens=False).ids
    if ids_of(tok(TOKENIZER_PROBE, add_special_tokens=False)) == want:
        return tok
    from transformers import PreTrainedTokenizerFast
    cfgp = os.path.join(model_dir, 'tokenizer_config.json')
    tc = json.load(open(cfgp, encoding='utf-8')) if os.path.exists(cfgp) else {}
    special = {}
    for k in ('bos_token', 'eos_token', 'pad_token', 'unk_token'):
        v = tc.get(k)
        if isinstance(v, dict):
            v = v.get('content')
        if v:
            special[k] = v
    fixed = PreTrainedTokenizerFast(tokenizer_file=tj, **special)
    tmpl = tc.get('chat_template')
    jp = os.path.join(model_dir, 'chat_template.jinja')
    if os.path.exists(jp):
        tmpl = open(jp, encoding='utf-8').read()
    fixed.chat_template = tmpl if tmpl is not None else tok.chat_template
    assert ids_of(fixed(TOKENIZER_PROBE, add_special_tokens=False)) == want, 'tokenizer rebuild failed'
    print(f'[tokenizer] {type(tok).__name__} disagreed with tokenizer.json; rebuilt from tokenizer.json', flush=True)
    fixed._p2_rebuilt_from = type(tok).__name__
    return fixed


def check_loading_info(info, cfg, model_dir):
    """Refuse a load in which checkpoint tensors went unused or parameters stayed uninitialised (e.g. transformers merging
    gate/up projections of an FP8 checkpoint and dropping gate_proj.weight_scale_inv). Allowed: multi-token-prediction weights
    (mtp.*, or decoder layers at index >= num_hidden_layers), weights deliberately skipped by a model patch (n-gram table),
    and, as missing only, the DSA indexer of a layer whose indexer type is 'shared': transformers builds an indexer in every
    layer, but a shared layer reuses the previous layer's top-k indices and never calls its own, and the checkpoint has none."""
    import re as _re
    tc = getattr(cfg, 'text_config', cfg)
    nl = getattr(tc, 'num_hidden_layers', None)
    itypes = getattr(tc, 'indexer_types', None) or []

    def allowed(k):
        if k.startswith('mtp.') or '.mtp.' in k or 'ngram_embedding' in k:
            return True
        m = _re.search(r'layers\.(\d+)\.', k)
        return bool(nl is not None and m and int(m.group(1)) >= nl)

    def unused_shared_indexer(k):
        m = _re.search(r'layers\.(\d+)\.self_attn\.indexer\.', k)
        return bool(m and int(m.group(1)) < len(itypes) and itypes[int(m.group(1))] == 'shared')
    unexpected = [k for k in info.get('unexpected_keys', []) if not allowed(k)]
    missing = [k for k in info.get('missing_keys', []) if not allowed(k) and not unused_shared_indexer(k)]
    mism = list(info.get('mismatched_keys', []))
    print(f'[load] unexpected {len(unexpected)} missing {len(missing)} mismatched {len(mism)}', flush=True)
    bad = unexpected + missing + [str(x) for x in mism]
    assert not bad, f'checkpoint/model mismatch, refusing to run: {bad[:6]}'


def load(args):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
    if not torch.cuda.is_available():
        sys.exit('CUDA unavailable: refuse to run on CPU')
    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tok = checked_tokenizer(tok, args.model)
    n = torch.cuda.device_count()
    kw = dict(dtype=torch.bfloat16, device_map='auto', trust_remote_code=True, max_memory={i: f'{args.max_memory_gib}GiB' for i in range(n)})
    if getattr(args, 'cpu_offload_gib', 0):
        kw['max_memory']['cpu'] = f'{args.cpu_offload_gib}GiB'
    if getattr(args, 'disk_offload_dir', None):
        kw['offload_folder'] = args.disk_offload_dir
    qc = getattr(cfg, 'quantization_config', None)
    qm = (qc.get('quant_method') if isinstance(qc, dict) else getattr(qc, 'quant_method', None)) if qc is not None else None
    fp8_backend = None
    if str(qm) in ('fp8', 'QuantizationMethod.FP8'):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import importlib
        import fp8_fallback
        fp8_fallback = importlib.reload(fp8_fallback)  # the server3 worker is long-lived: use the fallback as it is on disk now
        fp8_backend = fp8_fallback.install()
        print(f'[fp8] backend {fp8_backend}, disk patch {getattr(fp8_fallback.Q_CLS(), "_p2_disk_patch", False)}', flush=True)
    flashnext = getattr(cfg, 'model_type', '') == 'qwen4_exp'
    if flashnext:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import flashnext_patch
        kw['config'] = flashnext_patch.before_load(cfg)
    multimodal = hasattr(cfg, 'vision_config') or any('ConditionalGeneration' in a for a in (cfg.architectures or []))
    cls = AutoModelForImageTextToText if multimodal else AutoModelForCausalLM
    kw['output_loading_info'] = True
    weights = args.model
    attn = 'sdpa'
    if getattr(cfg, 'model_type', '') == 'glm_moe_dsa':  # GLM-5.3 semantics missing from transformers 5.8 (glm53_compat)
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import importlib
        import glm53_compat
        glm53_compat = importlib.reload(glm53_compat)
        if glm53_compat.apply(cfg, args.model):
            kw['config'] = cfg
            attn = 'eager'
    ondemand = getattr(args, 'expert_ondemand', None)
    if ondemand:                                   # routed experts read per selection from the original shards (glm_ondemand)
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import importlib
        import glm_ondemand
        glm_ondemand = importlib.reload(glm_ondemand)
        weights = str(glm_ondemand.filter_checkpoint(args.model, ondemand, cfg.num_hidden_layers))
        glm_ondemand.patch_modeling()
    first_error = None
    try:
        model, info = cls.from_pretrained(weights, attn_implementation=attn, **kw)
    except Exception as e:
        first_error = repr(e)[:300]
    if first_error is not None:                    # outside the except block, so the failed attempt's tensors can be freed
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        assert attn == 'sdpa', f'load with {attn} attention failed: {first_error}'
        print(f'[load] sdpa attempt failed ({first_error}); retrying with the default attention', flush=True)
        model, info = cls.from_pretrained(weights, **kw)
    check_loading_info(info, cfg, args.model)
    model.eval()
    if ondemand:
        glm_ondemand.bind(model, args.model, getattr(args, 'expert_cache_gib', 24))
    if flashnext:
        flashnext_patch.after_load(model, args.model)
    if getattr(args, 'sim_fp8', None):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import fp8_fakequant
        model._p2_sim_fp8 = (args.sim_fp8, fp8_fakequant.apply(model, args.sim_fp8))
        print(f'[sim-fp8] {model._p2_sim_fp8[1]} linear layers quantized ({args.sim_fp8})', flush=True)
    placement = {k: str(v) for k, v in getattr(model, 'hf_device_map', {}).items()}
    if not getattr(args, 'disk_offload_dir', None):
        assert not any(v == 'disk' for v in placement.values()), 'disk offload refused (pass --disk-offload-dir to allow it)'
    model._p2_fp8_backend = fp8_backend
    if not getattr(args, 'cpu_offload_gib', 0):
        assert not any(v == 'cpu' for v in placement.values()), 'CPU offload refused (pass --cpu-offload-gib to allow weight offload)'
    return torch, tok, model, cfg, placement, cls.__name__


def atomic(path, obj):
    tmp = Path(str(path) + '.part')
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)


def run(args):
    global Q_ATTR, Q_REPLY, Q_FINAL
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    kw = json.loads(args.tmpl_kwargs)
    torch, tok, model, cfg, placement, cls = load(args)
    lname, layers = find_layers(model)
    L = len(layers)
    R = Renderer(tok, kw)
    letters = {}
    for x in LETTERS:
        t = tok.encode(x, add_special_tokens=False)
        assert len(t) == 1, (x, t)
        letters[x] = t[0]
    bodies = [json.loads(l) for l in open(args.bodies, encoding='utf-8') if l.strip()]
    if args.lang == 'en':
        Q_ATTR, Q_REPLY, Q_FINAL = Q_ATTR_EN, Q_REPLY_EN, Q_FINAL_EN
        en = json.loads(Path(args.bodies_en).read_text(encoding='utf-8'))
        bodies = [{**b, 'body': en[b['id']]} for b in bodies]
    problems = json.loads(Path(args.problems).read_text(encoding='utf-8'))
    ident = {'experiment': 'P2_crossmodel_suite', 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'model': args.model, 'model_class': cls, 'architectures': cfg.architectures, 'layers_module': lname, 'n_layers': L,
             'tmpl_kwargs': kw, 'device_map': placement, 'torch': torch.__version__, 'host': socket.gethostname(),
             'transformers': __import__('transformers').__version__, 'cpu_offload_gib': getattr(args, 'cpu_offload_gib', 0), 'geometry_only': getattr(args, 'geometry_only', False), 'tokenizer_rebuilt_from': getattr(tok, '_p2_rebuilt_from', None), 'fp8_backend': getattr(model, '_p2_fp8_backend', None), 'sim_fp8': getattr(model, '_p2_sim_fp8', None), 'disk_offload_dir': getattr(args, 'disk_offload_dir', None),
             'expert_ondemand': getattr(args, 'expert_ondemand', None), 'expert_cache_gib': getattr(args, 'expert_cache_gib', None),
             'g_batch': getattr(args, 'g_batch', 1),
             'n_bodies': len(bodies), 'n_problems': len(problems), 'phases': args.phases, 'lang': args.lang,
             'attr_question_example': attr_question(ORDERS[0]), 'reply_question': Q_REPLY, 'final_question': Q_FINAL,
             'problem_example': problems[0]['problem']}
    atomic(out / 'identity.json', ident)
    print(json.dumps({k: ident[k] for k in ('model_class', 'layers_module', 'n_layers')}), flush=True)
    dev0 = model.get_input_embeddings().weight.device
    st = {'mode': None}

    def hook(i):
        def h(mod, inp, outp):
            m = st['mode']
            if m is None:
                return None
            x = outp[0] if isinstance(outp, tuple) else outp
            if m == 'capture':
                st['cap'][i] = x[0].detach()
                return None
            if m == 'capture_batch':
                st['cap'][i] = x.detach()
                return None
            if m == 'replace' and (st['layer'] == 'all' or st['layer'] == i):
                x[0, st['rpos']] = st['donor'][i][st['dpos']].to(x.device)
                return outp
            return None
        return h
    handles = [layers[i].register_forward_hook(hook(i)) for i in range(L)]

    def fwd(ids, mode=None, keep=1, zero=None, pos=None, **extra):
        st.clear()
        st.update(mode=mode, cap={}, **extra)
        x = torch.tensor([ids], dtype=torch.long, device=dev0)
        with torch.no_grad():
            am = torch.ones_like(x)
            if zero:  # --mask-extra: these positions are not read by any position
                am[0, zero] = 0
            if pos is not None:  # --pos-mode: explicit position ids (qwen3_5 takes four identical rows)
                pt = torch.tensor([pos], dtype=torch.long, device=dev0)
                if getattr(cfg, 'model_type', '') in ('qwen3_5', 'qwen3_5_moe'):
                    pt = pt.view(1, 1, -1).expand(4, 1, -1)
                o = model(input_ids=x, attention_mask=am, position_ids=pt, use_cache=False, return_dict=True, logits_to_keep=keep)
            elif zero and args.mask_pos:  # the position ids that generation derives from the mask
                o = model(input_ids=x, attention_mask=am, position_ids=(am.cumsum(-1) - 1).clamp(min=0), use_cache=False,
                          return_dict=True, logits_to_keep=keep)
            else:
                o = model(input_ids=x, attention_mask=am, use_cache=False, return_dict=True, logits_to_keep=keep)
        cap = st.get('cap', {})
        st['mode'] = None
        return torch.log_softmax(o.logits[0].float(), -1), cap

    def letter_p(lp, order):
        return {k: float(lp[-1, letters[LETTERS[i]]].exp()) for i, k in enumerate(order)}

    def answer_prefix(ids):
        # format tokens the model writes before its answer letter (greedy, probability > 0.5, at most two)
        pre = []
        for _ in range(2):
            lp, _ = fwd(ids + pre)
            mass = sum(float(lp[-1, letters[x]].exp()) for x in LETTERS)
            top = int(lp[-1].argmax())
            if mass >= 0.5 or float(lp[-1, top].exp()) < 0.5 or top in letters.values():
                return pre, lp
            pre.append(top)
        lp, _ = fwd(ids + pre)
        return pre, lp

    t0 = time.time()
    # ---------------- G ----------------
    def store_stats():
        s = getattr(model, '_p2_expert_store', None)
        return '' if s is None else (f" | experts hit {s.stats['hits']} miss {s.stats['misses']} "
                                     f"read {s.stats['bytes_read'] / 2 ** 30:.0f} GiB in {s.stats['read_s']:.0f}s")

    def fwd_capture_batch(seqs):
        # right padding: with causal attention the real tokens never see the pads, so their states equal unpadded ones
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        T = max(len(s) for s in seqs)
        x = torch.tensor([s + [pad] * (T - len(s)) for s in seqs], dtype=torch.long, device=dev0)
        am = torch.tensor([[1] * len(s) + [0] * (T - len(s)) for s in seqs], dtype=torch.long, device=dev0)
        st.clear()
        st.update(mode='capture_batch', cap={})
        with torch.no_grad():
            model(input_ids=x, attention_mask=am, use_cache=False, return_dict=True, logits_to_keep=1)
        cap = st.get('cap', {})
        st['mode'] = None
        return cap

    gb = int(getattr(args, 'g_batch', 1) or 1)
    if 'G' in args.phases and getattr(args, 'geometry_only', False) and gb > 1:
        # several bodies x 3 roles per forward: for models whose experts are read per forward (glm_ondemand), one forward
        # costs about the same whether it holds 3 sequences or 30
        gdir = out / 'G'
        gdir.mkdir(exist_ok=True)
        todo = [b for b in bodies if not (gdir / f'{b["id"]}.json').exists()]
        for c0 in range(0, len(todo), gb):
            group = todo[c0:c0 + gb]
            seqs, meta = [], []
            for b in group:
                geo = {r: R.render([{'role': r, 'content': b['body']}, {'role': 'user', 'content': Q_REPLY}], b['body']) for r in ROLES}
                spans = common_body(geo)
                for r in ROLES:
                    meta.append((b['id'], r, spans[r], len(geo[r][0])))
                    seqs.append(geo[r][0])
            cap = fwd_capture_batch(seqs)
            for b in group:
                arrs, rec = {}, {'id': b['id'], 'split': b['split'], 'category': b['category'], 'roles': {}, 'g_batch': gb}
                for j, (bid, r, pos, n) in enumerate(meta):
                    if bid != b['id']:
                        continue
                    arrs[f'content_{r}'] = torch.stack([cap[i][j, pos].float().mean(0).to(dev0) for i in range(L)]).half().cpu().numpy()
                    arrs[f'onset_{r}'] = torch.stack([cap[i][j, n - 1].float().to(dev0) for i in range(L)]).half().cpu().numpy()
                    rec['roles'][r] = {'n_body_tokens': len(pos), 'n_tokens': n}
                np.savez_compressed(gdir / f'{b["id"]}.npz', **arrs)
                atomic(gdir / f'{b["id"]}.json', rec)
            print(f'G {c0 + len(group)}/{len(todo)} {time.time() - t0:.0f}s{store_stats()}', flush=True)
    elif 'G' in args.phases:
        gdir = out / 'G'
        gdir.mkdir(exist_ok=True)
        for bi, b in enumerate(bodies):
            done = gdir / f'{b["id"]}.json'
            if done.exists():
                continue
            geo = {r: R.render([{'role': r, 'content': b['body']}, {'role': 'user', 'content': Q_REPLY}], b['body']) for r in ROLES}
            spans = common_body(geo)
            arrs, rec = {}, {'id': b['id'], 'split': b['split'], 'category': b['category'], 'roles': {}}
            for r in ROLES:
                ids = geo[r][0]
                zero = extra_header(geo) if (args.mask_extra and r == 'assistant') else None
                pos = positions(geo, args.pos_mode, args.pos_gap) if (args.pos_mode != 'none' and r == 'assistant') else None
                _, cap = fwd(ids, 'capture', zero=zero, pos=pos)
                pos = spans[r]
                arrs[f'content_{r}'] = torch.stack([cap[i][pos].float().mean(0).to(dev0) for i in range(L)]).half().cpu().numpy()
                arrs[f'onset_{r}'] = torch.stack([cap[i][len(ids) - 1].float().to(dev0) for i in range(L)]).half().cpu().numpy()
                if getattr(args, 'geometry_only', False):
                    rec['roles'][r] = {'n_body_tokens': len(pos), 'n_tokens': len(ids), 'masked': zero,
                                       'masked_tokens': tok.convert_ids_to_tokens([ids[i] for i in zero]) if zero else None,
                                       'pos_mode': args.pos_mode if r == 'assistant' else 'none',
                                       'body_start_position': (pos[pos_body(geo)] if pos is not None else geo[r][1][0])}
                    continue
                ps = []
                for order in ORDERS:
                    aid, _ = R.render([{'role': r, 'content': b['body']}, {'role': 'user', 'content': attr_question(order)}], b['body'])
                    pre, lp = answer_prefix(aid)
                    ps.append({**letter_p(lp, order), 'prefix': tok.decode(pre)})
                rec['roles'][r] = {'by_order': ps, 'mean': {k: float(np.mean([p[k] for p in ps])) for k in ('self', 'user', 'system')},
                                   'letter_mass_min': float(min(sum(p[k] for k in ('self', 'user', 'system')) for p in ps)), 'n_body_tokens': len(pos)}
            np.savez_compressed(gdir / f'{b["id"]}.npz', **arrs)
            atomic(done, rec)
            if bi % 10 == 0:
                print(f'G {bi + 1}/{len(bodies)} {time.time() - t0:.0f}s{store_stats()}', flush=True)
    # ---------------- P ----------------
    if 'P' in args.phases:
        pdir = out / 'P'
        pdir.mkdir(exist_ok=True)
        layers_sel = sorted({int(round(q * (L - 1))) for q in (0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)})
        held = [b for b in bodies if b['split'] == 'confirmation'][:args.patch_bodies]
        for bi, b in enumerate(held):
            done = pdir / f'{b["id"]}.json'
            if done.exists():
                continue
            rec = {'id': b['id'], 'layers': layers_sel, 'items': []}
            for order in ORDERS[:2]:
                q = attr_question(order)
                seq = {r: R.render([{'role': r, 'content': b['body']}, {'role': 'user', 'content': q}], b['body']) for r in ('assistant', 'user')}
                sp = common_body(seq)
                ia, iu = seq['assistant'][0], seq['user'][0]
                ba, bu = sp['assistant'], sp['user']
                assert [ia[i] for i in ba] == [iu[i] for i in bu]
                ta, tu = ia[ba[-1] + 1:], iu[bu[-1] + 1:]
                k = 0
                while k < min(len(ta), len(tu)) and ta[len(ta) - 1 - k] == tu[len(tu) - 1 - k]:
                    k += 1
                assert k >= 3, 'no common tokens after the body'
                regs = {'body': (ba, bu), 'after': (list(range(len(ia) - k, len(ia))), list(range(len(iu) - k, len(iu))))}
                pa, _ = answer_prefix(ia)
                pu, _ = answer_prefix(iu)
                ia, iu = ia + pa, iu + pu
                regs['body_after'] = (regs['body'][0] + regs['after'][0], regs['body'][1] + regs['after'][1])
                caps, base = {}, {}
                for r, ids in (('assistant', ia), ('user', iu)):
                    lp, cap = fwd(ids, 'capture')
                    caps[r] = {i: v for i, v in cap.items()}
                    base[r] = letter_p(lp, order)['self']
                for rec_r, don_r in (('assistant', 'user'), ('user', 'assistant')):
                    for reg, (pa, pu) in regs.items():
                        rpos, dpos = (pa, pu) if rec_r == 'assistant' else (pu, pa)
                        for l in layers_sel + ['all']:
                            lp, _ = fwd(ia if rec_r == 'assistant' else iu, 'replace', layer=l, rpos=rpos, dpos=dpos, donor=caps[don_r])
                            rec['items'].append({'order': list(order), 'recipient': rec_r, 'region': reg, 'layer': l,
                                                 'p_self': letter_p(lp, order)['self'], 'base_recipient': base[rec_r], 'base_donor': base[don_r]})
                caps.clear()
            atomic(done, rec)
            print(f'P {bi + 1}/{len(held)} {time.time() - t0:.0f}s', flush=True)
    # ---------------- M ----------------
    if 'M' in args.phases:
        mdir = out / 'M'
        mdir.mkdir(exist_ok=True)

        def digit_prefix(ids):
            pre = []
            for _ in range(2):
                lp, _ = fwd(ids + pre)
                top = int(lp[-1].argmax())
                if float(lp[-1, top].exp()) < 0.5 or tok.decode([top]).strip()[:1].isdigit():
                    break
                pre.append(top)
            return pre

        def score(ids, cand):
            cid = tok.encode(cand, add_special_tokens=False)
            lp, _ = fwd(ids + cid, keep=len(cid) + 1)
            return float(sum(lp[j, cid[j]] for j in range(len(cid))))
        for pi, p in enumerate(problems):
            done = mdir / f'{p["id"]}.json'
            if done.exists():
                continue
            rec = {'id': p['id'], 'family': p['family'], 'split': p['split'], 'results': []}
            for which in ('correct', 'wrong'):
                for r in ('assistant', 'user'):
                    step = p[f'step_{which}']
                    ids, _ = R.render([{'role': 'user', 'content': p['problem']}, {'role': r, 'content': step}, {'role': 'user', 'content': Q_FINAL}], step)
                    ids = ids + digit_prefix(ids)
                    lc, la = score(ids, str(p['answer'])), score(ids, str(p['answer_if_adopted']))
                    rec['results'].append({'step': which, 'role': r, 'lp_correct': lc, 'lp_adopt': la,
                                           'adoption_index': 1 / (1 + math.exp(lc - la))})
            atomic(done, rec)
            if pi % 20 == 0:
                print(f'M {pi + 1}/{len(problems)} {time.time() - t0:.0f}s', flush=True)
    for h in handles:
        h.remove()
    atomic(out / 'status.json', {'state': 'complete', 'elapsed_s': time.time() - t0})
    summarize(out, L)


def rank(a):
    a = np.asarray(a, float)
    r = np.empty(len(a))
    r[a.argsort(kind='mergesort')] = np.arange(len(a))
    for v in np.unique(a):
        m = a == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def spearman(x, y):
    return float(np.corrcoef(rank(x), rank(y))[0, 1])


def summarize(out, L):
    S = {'n_layers': L}
    gdir = out / 'G'
    if gdir.exists():
        recs = [json.loads(f.read_text(encoding='utf-8')) for f in sorted(gdir.glob('*.json'))]
        V = {r['id']: np.load(gdir / f'{r["id"]}.npz') for r in recs}
        held = [r for r in recs if r['split'] == 'confirmation']
        disc = [r for r in recs if r['split'] == 'discovery']
        for kind in ('content', 'onset'):
            Rs, cos = [], []
            dirs = np.mean([V[r['id']][f'{kind}_assistant'].astype(np.float32) - (V[r['id']][f'{kind}_user'].astype(np.float32) + V[r['id']][f'{kind}_system'].astype(np.float32)) / 2 for r in disc], 0)
            for r in held:
                a, u, s = (V[r['id']][f'{kind}_{x}'].astype(np.float32) for x in ROLES)
                au, as_, us = np.linalg.norm(a - u, axis=-1), np.linalg.norm(a - s, axis=-1), np.linalg.norm(u - s, axis=-1)
                Rs.append([us / ((au + as_) / 2), as_ / ((au + us) / 2), au / ((as_ + us) / 2)])
                off = a - (u + s) / 2
                cos.append((off * dirs).sum(-1) / (np.linalg.norm(off, axis=-1) * np.linalg.norm(dirs, axis=-1) + 1e-9))
            Rs = np.array(Rs)
            low = Rs.argmin(1)
            S[kind] = {'mean_R': Rs.mean(0).tolist(), 'share_A_lowest': (low == 0).mean(0).tolist(),
                       'layers_A_lowest_all_bodies': int(((low == 0).mean(0) == 1).sum()), 'layers_A_lowest_80': int(((low == 0).mean(0) >= 0.8).sum()),
                       'direction_cos_mean': np.mean(cos, 0).tolist()}
        if all('mean' in r['roles'][r0] for r in recs for r0 in ROLES):
            S['attr'] = {r0: {k: float(np.mean([r['roles'][r0]['mean'][k] for r in recs])) for k in ('self', 'user', 'system')} for r0 in ROLES}
            S['attr_top_share'] = {r0: float(np.mean([max(r['roles'][r0]['mean'], key=r['roles'][r0]['mean'].get) == {'assistant': 'self', 'user': 'user', 'system': 'system'}[r0] for r in recs])) for r0 in ROLES}
            S['letter_mass_min'] = float(min(r['roles'][r0]['letter_mass_min'] for r in recs for r0 in ROLES))
            dirs = np.mean([V[r['id']]['content_assistant'].astype(np.float32) - (V[r['id']]['content_user'].astype(np.float32) + V[r['id']]['content_system'].astype(np.float32)) / 2 for r in disc], 0)
            rho = []
            for l in range(L):
                w = dirs[l] / (np.linalg.norm(dirs[l]) + 1e-9)
                x = [float((V[r['id']]['content_assistant'][l].astype(np.float32) - np.mean([V[r['id']][f'content_{q}'][l].astype(np.float32) for q in ROLES], 0)) @ w) for r in held]
                y = [r['roles']['assistant']['mean']['self'] for r in held]
                rho.append(spearman(x, y))
            S['prediction_rho_content_by_layer'] = rho
    pdir = out / 'P'
    if pdir.exists() and any(pdir.glob('*.json')):
        items = [it for f in sorted(pdir.glob('*.json')) for it in json.loads(f.read_text(encoding='utf-8'))['items']]
        res = {}
        for rec_r in ('assistant', 'user'):
            for reg in ('body', 'after', 'body_after'):
                for l in sorted({it['layer'] for it in items}, key=lambda z: (isinstance(z, str), z)):
                    g = [it for it in items if it['recipient'] == rec_r and it['region'] == reg and it['layer'] == l]
                    if not g:
                        continue
                    gap = np.mean([it['base_donor'] for it in g]) - np.mean([it['base_recipient'] for it in g])
                    res[f'{rec_r}__{reg}__{l}'] = float(np.mean([it['p_self'] - it['base_recipient'] for it in g]) / gap) if abs(gap) > 1e-6 else None
        S['patch_gap_closed'] = res
    mdir = out / 'M'
    if mdir.exists() and any(mdir.glob('*.json')):
        rs = [x for f in sorted(mdir.glob('*.json')) for x in json.loads(f.read_text(encoding='utf-8'))['results']]
        S['reasoning'] = {f'{w}_{r}': float(np.mean([x['adoption_index'] for x in rs if x['step'] == w and x['role'] == r])) for w in ('correct', 'wrong') for r in ('assistant', 'user')}
    (out / 'summary.json').write_text(json.dumps(S, indent=1), encoding='utf-8')
    q = [0, L // 4, L // 2, 3 * L // 4, L - 1]
    for kind in ('content', 'onset'):
        if kind in S:
            print(f'{kind}: R_A at {q}', [round(S[kind]["mean_R"][0][i], 3) for i in q], '| layers A lowest in all held-out bodies',
                  S[kind]['layers_A_lowest_all_bodies'], '/ >=80%', S[kind]['layers_A_lowest_80'], '| dir cos', [round(S[kind]['direction_cos_mean'][i], 2) for i in q])
    if 'attr' in S:
        print('answers:', {r: {k: round(v, 3) for k, v in d.items()} for r, d in S['attr'].items()}, '| top-answer correct', {r: round(v, 3) for r, v in S['attr_top_share'].items()})
        print('prediction rho (content) at', q, [round(S['prediction_rho_content_by_layer'][i], 2) for i in q], '| max', round(max(S['prediction_rho_content_by_layer']), 2))
    if 'patch_gap_closed' in S:
        print('patch (assistant<-user, after):', {k.split('__')[2]: round(v, 2) for k, v in S['patch_gap_closed'].items() if k.startswith('assistant__after__') and v is not None})
        print('patch (assistant<-user, body):', {k.split('__')[2]: round(v, 2) for k, v in S['patch_gap_closed'].items() if k.startswith('assistant__body__') and v is not None})
        print('patch (user<-assistant, after):', {k.split('__')[2]: round(v, 2) for k, v in S['patch_gap_closed'].items() if k.startswith('user__after__') and v is not None})
    if 'reasoning' in S:
        print('reasoning adoption:', {k: round(v, 3) for k, v in S['reasoning'].items()})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--tmpl-kwargs', default='{}')
    p.add_argument('--bodies', required=True)
    p.add_argument('--problems', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--phases', default='GPM')
    p.add_argument('--patch-bodies', type=int, default=75)
    p.add_argument('--max-memory-gib', type=float, default=44)
    p.add_argument('--summarize-only', type=int, default=0)
    p.add_argument('--lang', choices=('zh', 'en'), default='zh')
    p.add_argument('--cpu-offload-gib', type=float, default=0, help='allow weights beyond the GPU budget to stay in CPU RAM (moved per forward)')
    p.add_argument('--geometry-only', action='store_true', help='phase G without the attribution forwards (thinking-mode renderings)')
    p.add_argument('--mask-extra', action='store_true', help='phase G: mask the Assistant-only header tokens out of attention')
    p.add_argument('--mask-pos', action='store_true', help='with --mask-extra: position ids follow the mask (as in generation)')
    p.add_argument('--pos-mode', choices=('none', 'compress', 'gap'), default='none', help='explicit position ids for the Assistant rendering')
    p.add_argument('--pos-gap', type=int, default=4, help='--pos-mode gap: offset added from the body on')
    p.add_argument('--disk-offload-dir', default=None, help='allow accelerate disk offload into this folder (models larger than GPU + CPU memory)')
    p.add_argument('--sim-fp8', default=None, help='simulate FP8 W8A8 inference on a BF16 model for this layer preset (fp8_fakequant.PRESETS)')
    p.add_argument('--expert-ondemand', default=None, help='directory for the expert-free copy of the checkpoint; routed experts are then read per selection (glm_ondemand)')
    p.add_argument('--expert-cache-gib', type=float, default=24, help='per-GPU LRU cache for on-demand experts')
    p.add_argument('--g-batch', type=int, default=1, help='bodies per forward in a geometry-only G phase')
    p.add_argument('--bodies-en', default=None, help='JSON id -> English body (with --lang en)')
    args = p.parse_args()
    if args.summarize_only:
        summarize(Path(args.out), args.summarize_only)
        return
    try:
        run(args)
    except Exception as exc:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        atomic(Path(args.out) / 'failed.json', {'error': repr(exc), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
