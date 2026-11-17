"""Complete a partial Hugging Face download file by file (11-12).

snapshot_download fails as a whole when hf-mirror answers 503 for any one file. For Ministral-3-14B-Reasoning-2512 this happened
on .gitattributes, README.md and chat_template.jinja, intermittently. This helper lists the repository at the pinned commit,
drops the ignore patterns, and downloads each missing file on its own with hf_hub_download, retrying each one. Files already in
local_dir are kept: hf_hub_download skips them when their metadata match. Exit code 0 only when every wanted file is present.

usage: python hf_complete_20261112.py REPO REVISION LOCAL_DIR [--ignore PATTERN ...] [--tries N] [--sleep S]
"""
import argparse
import fnmatch
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def main():
    p = argparse.ArgumentParser()
    p.add_argument('repo')
    p.add_argument('revision')
    p.add_argument('local_dir')
    p.add_argument('--ignore', nargs='*', default=[])
    p.add_argument('--tries', type=int, default=20)
    p.add_argument('--sleep', type=float, default=20)
    a = p.parse_args()
    files = None
    for t in range(a.tries):
        try:
            files = [s.rfilename for s in HfApi().model_info(a.repo, revision=a.revision).siblings]
            break
        except Exception as e:  # noqa: BLE001
            print(f'listing try {t + 1}: {type(e).__name__}: {str(e)[:160]}', flush=True)
            time.sleep(a.sleep)
    if files is None:
        print('listing FAILED', flush=True)
        return 1
    want = [f for f in files if not any(fnmatch.fnmatch(f, pat) for pat in a.ignore)]
    print(f'{len(files)} files in the repository, {len(want)} wanted after ignore {a.ignore}', flush=True)
    bad = []
    for f in want:
        ok = False
        for t in range(a.tries):
            try:
                hf_hub_download(a.repo, f, revision=a.revision, local_dir=a.local_dir)
                ok = True
                break
            except Exception as e:  # noqa: BLE001
                print(f'{f} try {t + 1}: {type(e).__name__}: {str(e)[:120]}', flush=True)
                time.sleep(a.sleep)
        print(('ok ' if ok else 'FAILED ') + f, flush=True)
        if not ok:
            bad.append(f)
    missing = [f for f in want if not (Path(a.local_dir) / f).exists()]
    print(f'complete: {not bad and not missing}; failed {bad}; missing {missing}', flush=True)
    return 0 if not bad and not missing else 1


if __name__ == '__main__':
    sys.exit(main())
