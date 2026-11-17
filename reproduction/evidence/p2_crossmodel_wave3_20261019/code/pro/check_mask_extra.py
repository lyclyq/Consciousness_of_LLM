"""Tokenizer-only check of crossmodel_suite_mask.extra_header (user 10-28, inducer test): for each model and template setting, the
Assistant-only header tokens that --mask-extra would mask, on the first and the last body. No model is loaded.
usage: check_mask_extra.py BODIES_JSON MODEL:LANG:KWARGS ..."""
import json
import sys

from transformers import AutoTokenizer

sys.path.insert(0, '.')
import crossmodel_suite_mask as S  # noqa: E402

bodies = json.load(open(sys.argv[1], encoding='utf-8'))
for spec in sys.argv[2:]:
    path, lang, kw = spec.split(':', 2)
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    R = S.Renderer(tok, json.loads(kw))
    q = S.Q_REPLY_EN if lang == 'en' else S.Q_REPLY
    for b in (bodies[0], bodies[-1]):
        body = b['body'] if isinstance(b, dict) else b
        geo = {r: R.render([{'role': r, 'content': body}, {'role': 'user', 'content': q}], body) for r in S.ROLES}
        z = S.extra_header(geo)
        ids = geo['assistant'][0]
        print(path.split('/')[-1], kw, 'masked', z, tok.convert_ids_to_tokens([ids[i] for i in z]),
              '| assistant header', tok.convert_ids_to_tokens(ids[max(0, geo['assistant'][1][0] - 8):geo['assistant'][1][0]]))
