#!/usr/bin/env python3
# scripts/merge_probe_results.py
# Usage: python scripts/merge_probe_results.py
# Merges <OUTPUT_DIR>/pip-audit.json and <OUTPUT_DIR>/conda-audit.json into docs/deps-audit.md
import json,os,time,sys
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'output')
if not os.path.isdir(OUTPUT_DIR):
    print(f"ERROR: OUTPUT_DIR '{OUTPUT_DIR}' does not exist. Please run probes first or set OUTPUT_DIR correctly.")
    sys.exit(1)

pip_path = os.path.join(OUTPUT_DIR, 'pip-audit.json')
conda_path = os.path.join(OUTPUT_DIR, 'conda-audit.json')
if not os.path.exists(pip_path):
    print('Missing', pip_path)
    raise SystemExit(1)

pipr = json.load(open(pip_path))
condar = json.load(open(conda_path)) if os.path.exists(conda_path) else {}
all_pkgs = sorted(set(list(pipr.keys()) + list(condar.keys())))

# Check docs dir writability
if not os.access('docs', os.W_OK):
    print("ERROR: no write permission for docs/ directory. Please fix permissions.")
    sys.exit(1)

with open('docs/deps-audit.md','a',encoding='utf8') as f:
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    f.write('\n\n## 动态探测结果（完整）\n\n')
    f.write(f'生成时间: {ts}\n\n')
    f.write('| package | verdict | note | log_path |\n|---|---|---|---|\n')
    for p in all_pkgs:
        r = condar.get(p) or pipr.get(p) or {}
        verdict = r.get('verdict','unknown')
        note = r.get('note','')
        log = r.get('log','')
        f.write(f'| {p} | {verdict} | {note} | {log} |\n')
print('Appended dynamic results to docs/deps-audit.md')
