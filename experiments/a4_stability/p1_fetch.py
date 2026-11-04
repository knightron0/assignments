"""Download P1/P2 reference-run histories from the public W&B project.

    uv run python -m experiments.a4_stability.p1_fetch

Writes one JSON per run to results/a4_stability/p{1,2}_runs/ (skips cached runs).
"""
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import wandb

from .sweeps import PUBLIC_PROJECT, VARIANTS

RESULTS = Path('results/a4_stability')
OUTPUT = RESULTS / 'p1_runs'


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def compact(rows):
    """scan_history fills every key on every row; keep {key: [[optimizer_step, value], ...]}."""
    series = {}
    for row in rows:
        step = row.get('optimizer_step')
        for key, value in row.items():
            if key.startswith('_') or key == 'optimizer_step' or not isinstance(value, (int, float)):
                continue
            series.setdefault(key, []).append([step, value])
    return series


def fetch(run, output=OUTPUT):
    path = output / f'{run.name}.json'
    if path.exists():
        return run.name
    series = compact(run.scan_history(page_size=1000))
    val = series.get('val_loss', [])
    variant = next(tag for tag in run.tags if tag in VARIANTS)
    record = dict(name=run.name, variant=variant, lr=run.config['learning_rate'],
                  state=run.state, url=run.url, series=series,
                  final_val_loss=val[-1][1] if val else None,
                  diverged=run.state != 'finished' or not val or not finite(val[-1][1]))
    path.write_text(json.dumps(record))
    return run.name


def main(problems=(1, 2)):
    for problem in problems:
        output = RESULTS / f'p{problem}_runs'
        output.mkdir(parents=True, exist_ok=True)
        runs = wandb.Api(timeout=300).runs(PUBLIC_PROJECT, filters={'tags': {'$all': ['a4-reference', f'p{problem}']}}, per_page=10)
        with ThreadPoolExecutor(8) as pool:
            for name in pool.map(lambda run: fetch(run, output), runs):
                print(name)


if __name__ == '__main__':
    main()
