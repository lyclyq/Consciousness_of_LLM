"""Appendix A (10-29): tokenizer, template and config files (no weights) of the four models whose weights are no longer on PRO,
at the revision of their runs, so that render_appendix_a_20261029.py can render them. Ministral-3-14B: the revisions pinned by
download_ministral_pro.sh. Qwen3.6-27B (run 10-16) and OLMo-2-13B (run 10-28): no pinned revision; take the newest commit
before the run day and report whether it is also the newest commit now (then the run used it).
usage: fetch_tokenizers_appendix_a_20261029.py OUT_ROOT"""
import os
import sys

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
from huggingface_hub import HfApi, snapshot_download  # noqa: E402

SPECS = [('Qwen/Qwen3.6-27B', None, '2026-10-16'), ('allenai/OLMo-2-1124-13B-Instruct', None, '2026-10-28'),
         ('mistralai/Ministral-3-14B-Instruct-2512-BF16', '3cea74c1ebaf5ce5f5a2553de470e2ceab825142', None),
         ('mistralai/Ministral-3-14B-Reasoning-2512', '51f9210f3cd20f3452a80d5819d15dc61cc50630', None)]
api = HfApi()
for repo, rev, before in SPECS:
    try:
        commits = api.list_repo_commits(repo)
        print(repo, 'newest commit now', commits[0].commit_id[:12], commits[0].created_at, flush=True)
        if rev is None:
            ok = [c for c in commits if str(c.created_at)[:10] < before]
            rev = ok[0].commit_id
            print('  newest before', before, rev[:12], ok[0].created_at, '| also newest now:', rev == commits[0].commit_id, flush=True)
    except Exception as e:
        info = api.model_info(repo)
        print(repo, 'commit list unavailable', type(e).__name__, '| model_info sha', info.sha[:12], 'lastModified', info.last_modified, flush=True)
        rev = rev or info.sha
    p = snapshot_download(repo, revision=rev, local_dir=os.path.join(sys.argv[1], repo), max_workers=4,
                          allow_patterns=['*.json', '*.jinja', '*.model', '*.txt', 'tokenizer*'])
    print('  downloaded', rev[:12], 'to', p, flush=True)
