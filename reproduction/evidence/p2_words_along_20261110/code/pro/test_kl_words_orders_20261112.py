"""CPU plumbing test of kl_words_orders_20261112.py --orders natural3 (11-12), no GPU, no real weights. Exploratory and internal.
crossmodel_suite.load is replaced by a loader that returns the real tokenizer of TOKDIR (the Ministral tokenizer files, as the
real run loads them) and a tiny random Llama with the same vocabulary size. The check criterion is lifted (a random model cannot
match the paper's KL), so every code path runs: natural3 role renderings, the fix_mistral_regex tokenizer, the reading of the
paper's nat3 reference (role pair names, control-token count), the 32 word pairs, the per-body records and the summary.
Asserts: the role pairs equal the reference's (A-U, A-S, U-S), 75 bodies on disk, summary with to_US and to_AU, words of all
32 pairs in both directions, identity with order natural3 and the tokenizer note.
usage (crossmodel directory, PYTHONPATH tf_next): test_kl_words_orders_20261112.py TOKDIR REFDIR OUTDIR"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))


def main():
    tokdir, refdir, outdir = sys.argv[1:4]
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(0)
    import crossmodel_suite as cs
    import kl_words_orders_20261112 as K

    def fake_load(largs):
        from transformers import AutoConfig, AutoTokenizer, LlamaConfig, LlamaForCausalLM
        cfg = AutoConfig.from_pretrained(largs.model, trust_remote_code=True)
        tok = cs.checked_tokenizer(AutoTokenizer.from_pretrained(largs.model, trust_remote_code=True), largs.model)
        lc = LlamaConfig(vocab_size=131072, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                         num_key_value_heads=2, max_position_embeddings=4096)
        model = LlamaForCausalLM(lc).eval()
        return torch, tok, model, cfg, {}, 'LlamaForCausalLM(test)'

    K.load = fake_load
    K.CHECK_REL = 1e18
    torch.cuda.is_available = lambda: True
    sys.argv = ['kl_words_orders_20261112.py', '--model', tokdir, '--tmpl-kwargs', '{}', '--bodies',
                '/root/selfindex_three_actual_20261011_v2/role_bodies.jsonl', '--words', 'words_v2.json', '--lang', 'en',
                '--bodies-en', 'bodies_en_150.json', '--orders', 'natural3', '--out-natural', outdir, '--check-natural', refdir,
                '--check-natural-kind', 'nat']
    K.main()
    out = Path(outdir)
    rows = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((out / 'bodies').glob('*.json'))]
    assert len(rows) == 75, len(rows)
    assert all(set(r['roles']) == {'A-U', 'A-S', 'U-S'} for r in rows)
    assert all(len(r['words']) == 32 and all(set(v) >= {'kl_content_12', 'kl_content_21', 'kl_body_12', 'kl_body_21'}
                                              for v in r['words'].values()) for r in rows)
    ident = json.loads((out / 'identity.json').read_text(encoding='utf-8'))
    assert ident['order'] == 'natural3' and ident['tokenizer_note'] and ident['tokenizer_note']['fix_mistral_regex'], ident
    s = json.loads((out / 'summary.json').read_text(encoding='utf-8'))
    c = s['kinds']['kl_content']
    assert 'to_US' in c and 'to_AU' in c and 'word_median_to_AU_12' in c, list(c)
    assert (out / 'DONE').exists()
    print('PLUMBING TEST OK', json.dumps({'n_bodies': s['n_bodies'], 'families': s['families'], 'roles_pairs': s['roles_pairs'],
                                          'check_kinds': rows[0]['check']['kinds']}), flush=True)


if __name__ == '__main__':
    main()
