"""Template-native renderings and token positions.

Every input is the model's own chat-template rendering. `Renderer.render` checks that the text rendering and the token rendering
of the template agree, and returns the token ids with the positions of one target text (a message body or a question).
`common_body` restricts the body positions of several renderings of the same body to the tokens they share, and `align` matches
the positions of a recipient and a donor rendering for the replacement of the causal tests.
"""
from __future__ import annotations

# end-of-reasoning markers of the thinking templates (Qwen, Gemma, Seed-OSS, gpt-oss); the answer is read after the last one
CLOSE = ('</think>', '<channel|>', '</seed:think>', '<|channel|>final<|message|>')


def ids_of(x):
    x = x['input_ids'] if hasattr(x, 'keys') else x
    return list(x[0]) if x and isinstance(x[0], list) else list(x)


class Renderer:
    def __init__(self, tok, kw):
        self.tok, self.kw = tok, kw

    def render(self, msgs, target):
        """Token ids of the rendering of `msgs` with the generation prompt, and the positions of `target` (which must occur
        exactly once in the rendered text)."""
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


def common_body(spans_ids):
    """{role: (ids, span)} -> {role: positions}: the body tokens shared by all renderings (a template may merge the first or the
    last body token with its own tokens)."""
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


def align(r, d):
    """Aligned positions of a recipient and a donor rendering, each (ids, body start, body end, generation prompt start). Two
    segments, matched token by token: the body (the same text under every role; its first or last token may be merged with the
    template, so the common prefix and suffix of the two body token runs), and the template tokens after the body up to the
    generation prompt (their common suffix; gpt-oss renders a System message as a developer message with two extra newlines after
    the body, which stay unedited). Returns the recipient and donor position lists."""
    (rid, rs, re_, rg), (did, ds, de, dg) = r, d
    rb, db = rid[rs:re_], did[ds:de]
    p = 0
    while p < min(len(rb), len(db)) and rb[p] == db[p]:
        p += 1
    q = 0
    while q < min(len(rb), len(db)) - p and rb[-1 - q] == db[-1 - q]:
        q += 1
    ri = list(range(rs, rs + p)) + list(range(re_ - q, re_))
    di = list(range(ds, ds + p)) + list(range(de - q, de))
    assert len(rid) - rg == len(did) - dg, 'generation prompts differ'
    n = 0
    while n < min(len(rid) - re_, len(did) - de) and rid[-1 - n] == did[-1 - n]:
        n += 1
    ri += list(range(len(rid) - n, rg))
    di += list(range(len(did) - n, dg))
    assert [rid[i] for i in ri] == [did[i] for i in di], 'aligned tokens differ'
    assert len(ri) >= 0.8 * ((re_ - rs) + (rg - re_)) - 4, 'too few aligned positions'
    return ri, di
