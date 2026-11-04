"""Moonlight-16B-A3B-Instruct under transformers 5.8, without changing any measurement code.
Two problems found when comparing it with the repository tokenizer:
  1. The tokenizer transformers builds from the tiktoken file (AutoTokenizer, trust_remote_code=True) gives the special tokens
     the wrong ids (<|im_system|> 163589 instead of 163594, <|im_middle|> 163590 instead of 163601): its converter appends the
     special tokens in a different order. Plain text tokenizes identically. fast_tokenizer() rebuilds it with the same converter,
     the repository's own split pattern (TikTokenTokenizer.pat_str) and every special token at the id the repository's
     TikTokenTokenizer gives it, and asserts those ids.
  2. The repository's modeling_deepseek.py (for transformers 4.48) does not import under 5.8. The model is DeepSeek-V3
     (model_type deepseek_v3: MLA without q LoRA, sigmoid scores, noaux_tc top-k), which transformers implements natively, so
     config and model load with trust_remote_code=False. The checkpoint's per-layer rotary inv_freq copies are not parameters of
     the native model (it computes inv_freq from the config) and are ignored on load; nothing else is.
install(MODEL_DIR) patches AutoTokenizer, AutoConfig and AutoModelForCausalLM.from_pretrained for that directory only.
run: python -m selfindex.patches.moonlight_compat MODEL_DIR MODULE_OR_SCRIPT [arguments]  (installs, then runs e.g. selfindex.main_probe as __main__)"""
import importlib.util
import json
import runpy
import sys
from pathlib import Path


def official_tokenizer(d):
    d = Path(d)
    from transformers import AddedToken
    spec = importlib.util.spec_from_file_location('tokenization_moonshot', d / 'tokenization_moonshot.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tc = json.load(open(d / 'tokenizer_config.json', encoding='utf-8'))
    kw = {k: tc[k] for k in ('bos_token', 'eos_token', 'unk_token', 'pad_token', 'additional_special_tokens') if k in tc}
    kw['added_tokens_decoder'] = {int(i): AddedToken(**v) for i, v in tc.get('added_tokens_decoder', {}).items()}
    return mod.TikTokenTokenizer(vocab_file=str(d / 'tiktoken.model'), **kw)


def fast_tokenizer(d):
    d = Path(d)
    from transformers import PreTrainedTokenizerFast
    from transformers.convert_slow_tokenizer import TikTokenConverter
    ref = official_tokenizer(d)
    sp = sorted(ref.special_tokens.items(), key=lambda kv: kv[1])
    base = sp[0][1]
    assert [i for _, i in sp] == list(range(base, base + len(sp))), 'special token ids are not contiguous'
    import inspect
    arg = 'extra_special_tokens' if 'extra_special_tokens' in inspect.signature(TikTokenConverter).parameters else 'additional_special_tokens'
    tk = TikTokenConverter(vocab_file=str(d / 'tiktoken.model'), pattern=ref.pat_str, **{arg: [t for t, _ in sp]}).converted()
    assert tk.get_vocab_size(with_added_tokens=True) == base + len(sp), 'special tokens not appended after the BPE vocabulary'
    tc = json.load(open(d / 'tokenizer_config.json', encoding='utf-8'))
    fast = PreTrainedTokenizerFast(tokenizer_object=tk, **{k: tc[k] for k in ('bos_token', 'eos_token', 'unk_token', 'pad_token')})
    jp = d / 'chat_template.jinja'
    fast.chat_template = jp.read_text(encoding='utf-8') if jp.exists() else tc['chat_template']
    bad = [(t, i, fast.convert_tokens_to_ids(t)) for t, i in sp if fast.convert_tokens_to_ids(t) != i]
    assert not bad, f'special token ids differ from TikTokenTokenizer: {bad[:4]}'
    fast._p2_rebuilt_from = 'TikTokenTokenizer ids (moonlight_compat_20261103)'
    return fast


def install(d):
    import transformers
    from transformers.models.deepseek_v3 import modeling_deepseek_v3 as ds
    d = str(Path(d).resolve())

    def same(p):
        try:
            return str(Path(p).resolve()) == d
        except TypeError:
            return False
    AT, AC, AM = transformers.AutoTokenizer, transformers.AutoConfig, transformers.AutoModelForCausalLM
    ot, oc, om = AT.from_pretrained, AC.from_pretrained, AM.from_pretrained

    def tok_fp(p, *a, **k):
        return fast_tokenizer(p) if same(p) else ot(p, *a, **k)

    def cfg_fp(p, *a, **k):
        if same(p):
            k['trust_remote_code'] = False
        return oc(p, *a, **k)

    def model_fp(p, *a, **k):
        if same(p):
            k['trust_remote_code'] = False
            cfg = k.get('config')
            if cfg is not None and type(cfg).__module__.startswith('transformers_modules'):
                k['config'] = oc(p, trust_remote_code=False)
        return om(p, *a, **k)
    AT.from_pretrained, AC.from_pretrained, AM.from_pretrained = tok_fp, cfg_fp, model_fp
    ign = list(getattr(ds.DeepseekV3PreTrainedModel, '_keys_to_ignore_on_load_unexpected', None) or [])
    ds.DeepseekV3PreTrainedModel._keys_to_ignore_on_load_unexpected = ign + [r'model\.layers\.\d+\.self_attn\.rotary_emb\.inv_freq']
    print(f'[moonlight_compat] {d}: tokenizer with TikTokenTokenizer special ids, native DeepseekV3 (trust_remote_code=False)', flush=True)


if __name__ == '__main__':
    model_dir, script = sys.argv[1], sys.argv[2]
    install(model_dir)
    sys.argv = [script] + sys.argv[3:]
    if script.endswith('.py'):
        sys.path.insert(0, str(Path(script).resolve().parent))
        runpy.run_path(script, run_name='__main__')
    else:  # a module of this package, e.g. selfindex.main_probe
        runpy.run_module(script, run_name='__main__', alter_sys=True)
