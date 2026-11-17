"""The message orders of the next-token measurement: msgs_for and natural_system_ok.

Verbatim copies of lines 71-73 (msgs_for) and 87-93 (natural_system_ok) of the as-run source with sha256
fab9c7b22dd202330bf8b2ec125dbafd1aa8e39e065347dc454dfda4ef1fe55c.
"""
from __future__ import annotations


def msgs_for(order, role, content, q, request):
    m = [{'role': role, 'content': content}, {'role': 'user', 'content': q}]
    return m if order == 'main' else [{'role': 'user', 'content': request}] + m


def natural_system_ok(tok, kw, q, request, text):
    """the template keeps a System message after a user turn, in place (else: error, dropped, or moved)"""
    try:
        t = tok.apply_chat_template(msgs_for('natural', 'system', text, q, request), tokenize=False, add_generation_prompt=True, **kw)
    except Exception:
        return False
    return t.count(text) == 1 and t.index(text) > t.index(request)
