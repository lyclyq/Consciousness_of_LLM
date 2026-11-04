"""Small file helpers shared by the experiments."""
from __future__ import annotations

import json
from pathlib import Path


def atomic(path, obj):
    """Write obj as JSON through a .part file, so that a result file on disk is always complete (runs resume per item)."""
    tmp = Path(str(path) + '.part')
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)
