#!/usr/bin/env bash
set -euo pipefail

# scripts/probe_deps.sh
# Usage: bash scripts/probe_deps.sh
# This script performs the dynamic dependency probes described in docs/deps-audit.md:
# 1) creates .venv_audit and runs pip --no-cache-dir installs per candidate package
# 2) records per-package pip logs in ${OUTPUT_DIR}/pip-logs and summary in ${OUTPUT_DIR}/pip-audit.json
# 3) creates conda env alpha-audit-conda (python=3.11) and attempts conda-forge installs for failures
# 4) records conda logs in ${OUTPUT_DIR}/conda-logs and summary in ${OUTPUT_DIR}/conda-audit.json
# 5) runs pre-commit (only records output) and writes ${OUTPUT_DIR}/pre-commit-run.log
# 6) cleans up temporary venv and (optionally) conda env

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

# Output directory can be overridden by environment variable OUTPUT_DIR
OUTPUT_DIR="${OUTPUT_DIR:-output}"
PIP_LOGS_DIR="$OUTPUT_DIR/pip-logs"
CONDA_LOGS_DIR="$OUTPUT_DIR/conda-logs"

# Create directories and check writability
mkdir -p "$PIP_LOGS_DIR" "$CONDA_LOGS_DIR" || { echo "Failed to create $PIP_LOGS_DIR or $CONDA_LOGS_DIR. Check permissions."; exit 1; }
# quick write test
if ! touch "$OUTPUT_DIR/.probe_write_test" 2>/dev/null; then
  echo "ERROR: No write permission in $OUTPUT_DIR. Please choose a different OUTPUT_DIR or fix permissions." >&2
  exit 1
fi
rm -f "$OUTPUT_DIR/.probe_write_test"

echo "Using OUTPUT_DIR=$OUTPUT_DIR"

echo "Creating venv .venv_audit..."
python -m venv .venv_audit
source .venv_audit/bin/activate
python -m pip install -U pip setuptools wheel

# Build package list from requirements.txt and some common extras
PKGS=()
if [ -f requirements.txt ]; then
  while IFS= read -r line; do
    line_trimmed="$(echo "$line" | sed -e 's/^\s*//' -e 's/\s*$//')"
    if [ -z "$line_trimmed" ] || [[ "$line_trimmed" == \#* ]]; then
      continue
    fi
    PKGS+=("$line_trimmed")
  done < requirements.txt
fi
# extras to ensure we probe common imports
EXTRAS=(polars numpy pandas scipy fastcluster loguru deap lightgbm tensorboardx talib cvxpy)
for e in "${EXTRAS[@]}"; do
  skip=0
  for p in "${PKGS[@]}"; do
    # Check if the package list already contains this extra (exact match or starts with the extra name,
    # e.g. "numpy>=1.24") without using regex with angle-brackets which can break the shell.
    if [[ "$p" = "$e" || "$p" == ${e}* ]]; then
      skip=1
      break
    fi
  done
  if [ $skip -eq 0 ]; then
    PKGS+=("$e")
  fi
done

# special conda-forced set
COND_A_FORCED=(TA-Lib talib cvxpy cvxopt)

# Export OUTPUT_DIR so python heredocs can read it
export OUTPUT_DIR

python - <<'PY'
import json,subprocess,sys,os
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'output')
pkgs = []
if os.path.exists('requirements.txt'):
    with open('requirements.txt','r',encoding='utf8') as f:
        reqs = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
else:
    reqs = []
extras = ['polars','numpy','pandas','scipy','fastcluster','loguru','deap','lightgbm','tensorboardX','talib','cvxpy']
pkgs = sorted(set(reqs + extras))

special_conda = {'TA-Lib','talib','cvxpy','cvxopt'}
results = {}

pip_logs_dir = os.path.join(OUTPUT_DIR, 'pip-logs')
if not os.path.isdir(pip_logs_dir):
    os.makedirs(pip_logs_dir, exist_ok=True)

for p in pkgs:
    pkg = p.strip()
    if not pkg:
        continue
    if any(s.lower() == pkg.lower() for s in special_conda):
        results[pkg] = {'verdict':'conda-forced','note':'Forced to conda (TA-Lib/cvxpy rules)'}
        continue
    outf = os.path.join(pip_logs_dir, f"{pkg.replace('/','_')}.log")
    try:
        r = subprocess.run([sys.executable,'-m','pip','install','--no-cache-dir',pkg],capture_output=True,text=True,timeout=900)
        with open(outf,'w',encoding='utf8') as fh:
            fh.write('STDOUT:\n'+r.stdout+'\n\nSTDERR:\n'+r.stderr)
        if r.returncode != 0:
            results[pkg] = {'verdict':'pip-install-failed','note':'pip install failed','log':outf}
            continue
        modname = pkg.split('==')[0].split('>=')[0].split()[0].split('[')[0]
        mapping = {'Pillow':'PIL','pyyaml':'yaml','tensorboardx':'tensorboardX'}
        mod = mapping.get(modname,modname)
        try:
            __import__(mod)
            results[pkg] = {'verdict':'poetry','note':'installed and import ok','log':outf}
        except Exception as e:
            with open(outf,'a',encoding='utf8') as fh:
                fh.write('\nIMPORT ERROR: '+repr(e))
            results[pkg] = {'verdict':'pip-install-no-import','note':'installed but import failed','log':outf}
    except Exception as e:
        results[pkg] = {'verdict':'pip-probe-exception','note':str(e)}

with open(os.path.join(OUTPUT_DIR, 'pip-audit.json'),'w',encoding='utf8') as fh:
    json.dump(results,fh,indent=2)
print('Wrote pip-audit.json to', os.path.join(OUTPUT_DIR, 'pip-audit.json'))
PY

# Create conda env and run conda probes
if command -v mamba >/dev/null 2>&1; then
  installer=mamba
else
  installer=conda
fi

echo "Creating temporary conda env 'alpha-audit-conda' with python=3.11..."
$installer create -y -n alpha-audit-conda python=3.11 || true
# Activate conda env
source $(conda info --base)/etc/profile.d/conda.sh
conda activate alpha-audit-conda

python - <<'PY'
import json,subprocess,sys,os
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'output')

pipres_path = os.path.join(OUTPUT_DIR, 'pip-audit.json')
pipres = json.load(open(pipres_path)) if os.path.exists(pipres_path) else {}
to_try = [p for p,r in pipres.items() if r.get('verdict') != 'poetry']
results = {}
which_mamba = subprocess.run(['which','mamba'],capture_output=True).returncode==0
conda_logs_dir = os.path.join(OUTPUT_DIR, 'conda-logs')
if not os.path.isdir(conda_logs_dir):
    os.makedirs(conda_logs_dir, exist_ok=True)
for p in to_try:
    outp = os.path.join(conda_logs_dir, f"{p.replace('/','_')}.log")
    try:
        installer = 'mamba' if which_mamba else 'conda'
        r = subprocess.run([installer,'install','-y','-c','conda-forge',p],capture_output=True,text=True,timeout=900)
        with open(outp,'w',encoding='utf8') as fh:
            fh.write('STDOUT:\n'+r.stdout+'\n\nSTDERR:\n'+r.stderr)
        if r.returncode != 0:
            results[p] = {'verdict':'conda-install-failed','log':outp}
            continue
        mod = p.split('==')[0].split('>=')[0].split()[0].split('[')[0]
        try:
            __import__(mod)
            results[p] = {'verdict':'conda','note':'conda install & import success','log':outp}
        except Exception as e:
            with open(outp,'a',encoding='utf8') as fh:
                fh.write('\nIMPORT ERROR: '+repr(e))
            results[p] = {'verdict':'conda-installed-no-import','note':str(e),'log':outp}
    except Exception as e:
        results[p] = {'verdict':'conda-probe-exception','note':str(e)}
with open(os.path.join(OUTPUT_DIR, 'conda-audit.json'),'w',encoding='utf8') as fh:
    json.dump(results,fh,indent=2)
print('Wrote conda-audit.json to', os.path.join(OUTPUT_DIR, 'conda-audit.json'))
PY

# Run pre-commit (record only)
python -m pip install --upgrade pre-commit || true
pre-commit run --all-files --show-diff-on-failure --color=always > "$OUTPUT_DIR/pre-commit-run.log" 2>&1 || true

# Deactivate conda env
conda deactivate || true

# Cleanup venv (kept conda env for optional inspection; user can remove it later)
deactivate 2>/dev/null || true
rm -rf .venv_audit

echo "Dynamic probes finished. Check $OUTPUT_DIR/ for logs and JSON summaries."
chmod +x scripts/probe_deps.sh
