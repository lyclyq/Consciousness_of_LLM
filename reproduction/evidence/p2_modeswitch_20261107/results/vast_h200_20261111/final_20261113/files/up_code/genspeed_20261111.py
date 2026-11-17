"""Generation speed of GLM-5.3-Flash on the rented machine (11-11): greedy decoding of the attribution prompt of the first held-out
body, batch 16 and batch 1, 64 new tokens, as the mechanism runner calls generate (left padding, pad token, do_sample=False).
usage: genspeed_20261111.py --model DIR --max-memory-gib G --cpu-offload-gib G"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crossmodel_suite import Renderer, load, attr_question, ORDERS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True)
ap.add_argument('--max-memory-gib', type=float, required=True)
ap.add_argument('--cpu-offload-gib', type=float, required=True)
a = ap.parse_args()
torch, tok, model, cfg, placement, cls = load(argparse.Namespace(model=a.model, max_memory_gib=a.max_memory_gib, cpu_offload_gib=a.cpu_offload_gib))
b = json.loads(Path('/root/glm53/ref/bodies_zh_150_list.json').read_text(encoding='utf-8'))[0]
R = Renderer(tok, {})
ids, _ = R.render([{'role': 'assistant', 'content': b['body']}, {'role': 'user', 'content': attr_question(ORDERS[0])}], b['body'])
dev0 = model.get_input_embeddings().weight.device
pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
for bs in (16, 1):
    x = torch.tensor([ids] * bs, device=dev0)
    t = time.time()
    with torch.no_grad():
        g = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=64, min_new_tokens=64, do_sample=False, pad_token_id=pad)
    torch.cuda.synchronize()
    dt = time.time() - t
    print(f'GENSPEED batch {bs}: prompt {len(ids)} tokens, 64 new tokens in {dt:.1f} s = {dt / 64:.3f} s per step, '
          f'{bs * 64 / dt:.1f} tokens/s; memory GiB {[round(torch.cuda.max_memory_allocated(i) / 2**30, 1) for i in range(torch.cuda.device_count())]}',
          flush=True)
print('sample:', tok.decode(g[0, len(ids):].tolist())[:200].replace('\n', ' '), flush=True)
