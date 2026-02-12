#!/usr/bin/env python3
import argparse
import csv
import glob
from collections import defaultdict
from pathlib import Path
from typing import Optional


def pick_root(arg_root: Optional[str]) -> Path:
    if arg_root:
        return Path(arg_root)
    roots = sorted(glob.glob('output/train/case_gpu_bench_*'))
    if not roots:
        raise FileNotFoundError('No case_gpu_bench_* directory found')
    return Path(roots[-1])


def to_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=None, help='benchmark root, e.g. output/train/case_gpu_bench_20260212_002319')
    ap.add_argument('--out', default=None, help='output markdown path')
    args = ap.parse_args()

    root = pick_root(args.root)
    tsv = root / 'best_by_case.tsv'
    if not tsv.exists():
        raise FileNotFoundError(f'{tsv} not found (benchmark may still be running)')

    rows = []
    with tsv.open() as f:
        r = csv.DictReader(f, delimiter='\t')
        rows = list(r)

    by_method = defaultdict(list)
    for row in rows:
        by_method[row['method']].append(row)

    out_path = Path(args.out) if args.out else root / 'summary.md'
    with out_path.open('w') as f:
        f.write(f'# Case GPU Benchmark Summary\n\n')
        f.write(f'- root: `{root}`\n')
        f.write(f'- total cases: `{len(rows)}`\n\n')

        f.write('## Method Summary\n\n')
        f.write('| method | ok/total | avg min util | avg samples/s |\n')
        f.write('|---|---:|---:|---:|\n')
        for method in sorted(by_method):
            ms = by_method[method]
            ok = [x for x in ms if x.get('status') == 'ok']
            avg_util = sum(to_float(x.get('min_avg_util')) for x in ok) / len(ok) if ok else 0
            avg_sps = sum(to_float(x.get('samples_per_s')) for x in ok) / len(ok) if ok else 0
            f.write(f'| `{method}` | `{len(ok)}/{len(ms)}` | `{avg_util:.2f}` | `{avg_sps:.2f}` |\n')

        f.write('\n## Case Details\n\n')
        f.write('| case | method | status | variant | num_envs | min_avg_util | samples/s | out_dir |\n')
        f.write('|---|---|---|---|---:|---:|---:|---|\n')
        for row in sorted(rows, key=lambda x: (x['method'], x['case'])):
            f.write(
                f"| `{row['case']}` | `{row['method']}` | `{row['status']}` | `{row.get('variant','')}` | "
                f"`{row.get('num_envs','')}` | `{to_float(row.get('min_avg_util')):.2f}` | "
                f"`{to_float(row.get('samples_per_s')):.2f}` | `{row.get('out_dir','')}` |\n"
            )

    print(out_path)


if __name__ == '__main__':
    main()
