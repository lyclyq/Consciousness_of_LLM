"""Body sets for the scaled P2 checks: all 150 bodies in a fixed order, and explicit id lists."""
import json
from pathlib import Path


def all_bodies(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    rows.sort(key=lambda r: (r['category'], r['split'], r['index']))
    assert len(rows) == 150 and len({r['id'] for r in rows}) == 150, len(rows)
    return rows


def bodies_from_ids(path, ids_file):
    ids = json.loads(Path(ids_file).read_text(encoding='utf-8'))
    by_id = {r['id']: r for r in all_bodies(path)}
    return [by_id[i] for i in ids]
