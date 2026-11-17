"""Tokenizer-only check (CPU, no model) for the natural3 KL words of the two Ministral-3-14B checkpoints (11-12). Exploratory and
internal, like the rest of this archive.
kl_words_orders_20261111.py cannot run these cells unchanged: its renderings() also renders the main probe, which the Mistral
template refuses ("Conversation must start with a user or system message"). The paper's natural3 KL run (mode_switch_nat_20261108.py
--order natural3, PRO, modeswitch/nat3/<tag>) rendered the roles with crossmodel_natural3.render and the fix_mistral_regex
tokenizer; the words-along natural runs rendered roles and words with words_along_20261110.plan(order 'natural') and the tokenizer of
crossmodel_suite.load (checked_tokenizer, no regex fix). This script compares, on the 75 held-out bodies (English), per checkpoint:
  roles   natural3 renderings (render_order 'natural3') under the fixed and the default tokenizer, and against the words-along
          role renderings (plan with roles, default tokenizer): role names, token ids, common body positions
  control the control-token sets of the two tokenizers (the content mask) and len(tokenizer)
  words   the words-along natural word renderings under the two tokenizers
usage (crossmodel directory, PYTHONPATH tf_next): test_nat3_ministral_render_20261112.py TOKDIR [TOKDIR ...]"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import crossmodel_suite as cs  # noqa: E402
from crossmodel_suite import checked_tokenizer  # noqa: E402
from persistence_readout_20261106 import natural_system_ok  # noqa: E402
from mode_switch_20261107 import REQ  # noqa: E402
from mode_switch_nat_20261108 import render_order  # noqa: E402
from words_along_20261110 import load_words, plan  # noqa: E402

B1 = '/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl'


def ctrl(tok):
    return sorted(set(tok.all_special_ids) | set(getattr(tok, 'added_tokens_decoder', {}).keys()))


def main():
    from transformers import AutoTokenizer
    allb = [json.loads(l) for l in open(B1, encoding='utf-8') if l.strip()]
    en = json.loads(Path('bodies_en_150.json').read_text(encoding='utf-8'))
    allb = [{**b, 'body': en[b['id']]} for b in allb]
    bodies = [b for b in allb if b.get('split') == 'confirmation']
    assert len(bodies) == 75
    q = cs.Q_REPLY_EN
    W, pairs = load_words('words_v2.json', 'en', 'all')
    out = {}
    for d in sys.argv[1:]:
        tok_d = checked_tokenizer(AutoTokenizer.from_pretrained(d, trust_remote_code=True), d)
        tok_f = AutoTokenizer.from_pretrained(d, fix_mistral_regex=True)
        kw = {}
        sys_ok = natural_system_ok(tok_d, kw, q, REQ['en'], allb[0]['body'])
        role_bad, word_bad, n_common, wa_bad = [], [], [], []
        for b in bodies:
            ra = render_order('natural3', tok_d, kw, b['body'], 'en', None)
            rb = render_order('natural3', tok_f, kw, b['body'], 'en', None)
            same = set(ra) == set(rb) and all(ra[k]['ids'] == rb[k]['ids'] and list(ra[k]['common']) == list(rb[k]['common']) for k in ra)
            if not same:
                role_bad.append(b['id'])
            n_common.append(len(next(iter(ra.values()))['common']))
            _, rw, wd, sd = plan(tok_d, kw, b['body'], 'natural', q, REQ['en'], sys_ok, pairs, True)
            if not (set(rw) == set(rb) and all(rw[k][0] == rb[k]['ids'] and list(sd['roles'][k]) == list(rb[k]['common']) for k in rw)):
                wa_bad.append(b['id'])
            _, _, wf, sf = plan(tok_f, kw, b['body'], 'natural', q, REQ['en'], sys_ok, pairs, False)
            for p in pairs:
                pid = p['id']
                if any(wd[pid][w][0] != wf[pid][w][0] or list(sd[pid][w]) != list(sf[pid][w]) for w in ('w1', 'w2')):
                    word_bad.append(f"{b['id']}:{pid}")
        cd, cf = ctrl(tok_d), ctrl(tok_f)
        out[d] = {'natural_system_ok': sys_ok, 'roles': sorted(ra), 'tok_default': type(tok_d).__name__, 'tok_fixed': type(tok_f).__name__,
                  'role_mismatch_bodies': role_bad, 'words_along_role_mismatch_bodies': wa_bad, 'word_mismatch': word_bad[:20], 'n_word_mismatch': len(word_bad),
                  'n_control_default': len(cd), 'n_control_fixed': len(cf), 'control_sets_equal': cd == cf,
                  'len_tok_default': len(tok_d), 'len_tok_fixed': len(tok_f), 'common_body_min_max': [min(n_common), max(n_common)]}
        print(json.dumps({d: out[d]}, indent=1), flush=True)
    ok = all(not v['role_mismatch_bodies'] and not v['words_along_role_mismatch_bodies'] and not v['n_word_mismatch'] and v['control_sets_equal'] for v in out.values())
    print('RENDER CHECK', 'OK' if ok else 'FAILED', flush=True)


if __name__ == '__main__':
    main()
