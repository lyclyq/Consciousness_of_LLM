"""Expected prompt tokens for the DeepSeek-V4.1 thinking-900 natural-order items under DeepSeek's own encoder (encoding.py,
thinking mode, reasoning effort "high" = the encoder default) and tokenizer. Items: 150 bodies x the first two option orders
x roles assistant / user / system, [user: request] [ROLE: body] [user: question]; the User body merges with its neighbours
in one user turn (DeepSeek's native rendering), the System body is a mid-conversation <System> message kept in place.
Also writes the rendered prompt of the first body for each role, for inspection.
usage: uv run --with tokenizers ds_expected_think900.py ENCODING_DIR BODIES TOKENIZER_JSON OUT.json"""
import json
import sys

sys.path.insert(0, sys.argv[1])
import encoding as E  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

REQ = '请写一段简短的描述。'
QH, OPT, QT = '这个问题之前的那条消息是谁写的？', {'self': '你（AI 助手）', 'user': '用户', 'system': '系统'}, '。只回答字母。'
ORDERS = (('self', 'user', 'system'), ('system', 'self', 'user'))
LETTERS = ('A', 'B', 'C')
ROLES = ('assistant', 'user', 'system')
bodies = json.load(open(sys.argv[2], encoding='utf-8'))
tok = Tokenizer.from_file(sys.argv[3])
out, shown = {}, {}
for b in bodies:
    for oi, order in enumerate(ORDERS):
        q = QH + ' '.join(f'{LETTERS[i]}. {OPT[k]}' for i, k in enumerate(order)) + QT
        for role in ROLES:
            msgs = [{'role': 'user', 'content': REQ}, {'role': role, 'content': b['body']}, {'role': 'user', 'content': q}]
            s = E.encode_messages(msgs, thinking_mode='thinking', reasoning_effort='high')
            if role == 'system':  # the body must stay between the request and the question, in its own System turn
                assert s.index(REQ) < s.index(E.SYSTEM_SP_TOKEN + b['body']) < s.rindex(q), 'system body moved'
            out[f"{b['id']}__{role}__o{oi}"] = len(tok.encode(s, add_special_tokens=False).ids)
            if b is bodies[0] and oi == 0:
                shown[role] = s
json.dump({'deepseek-flash': out}, open(sys.argv[4], 'w'), indent=0)
json.dump(shown, open(sys.argv[4].replace('.json', '_rendered.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(len(out), 'items')
