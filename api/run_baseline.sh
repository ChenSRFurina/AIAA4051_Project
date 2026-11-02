#!/usr/bin/env bash
# Activate NLP_project first. Optional arguments: searchqa hotpotqa.
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_dir}"

if [[ "$#" -eq 0 ]]; then
  datasets=(searchqa hotpotqa)
else
  datasets=("$@")
fi
for dataset in "${datasets[@]}"; do
  case "${dataset}" in
    searchqa|hotpotqa) ;;
    *) printf 'Unknown dataset: %s. Use searchqa or hotpotqa.\n' "${dataset}" >&2; exit 2 ;;
  esac
done

run_tag="$(date +%Y%m%d_%H%M%S)-$$"
for dataset in "${datasets[@]}"; do
  config_path="Baseline/configs/${dataset}_api.json"
  data_dir="Baseline/data/prepared/${dataset}"
  run_dir="Baseline/runs/api/${dataset}/${run_tag}"

  python -u Baseline/download_data.py --dataset "${dataset}"
  mkdir -p "${run_dir}"
  printf '\n[%s] Output: %s\n' "${dataset}" "${run_dir}"

  printf '[%s] Stage: pilot\n' "${dataset}"
  python -u Baseline/run.py pilot \
    --config "${config_path}" \
    --data "${data_dir}" \
    --output "${run_dir}/pilot" \
    2>&1 | tee "${run_dir}/pilot.log"

  printf '[%s] Stage: shared initialization\n' "${dataset}"
  python -u Baseline/run.py initialize \
    --config "${config_path}" \
    --data "${data_dir}" \
    --output "${run_dir}/shared_init" \
    2>&1 | tee "${run_dir}/initialize.log"

  printf '[%s] Stage: N\n' "${dataset}"
  python -u Baseline/run.py static \
    --condition N \
    --config "${config_path}" \
    --data "${data_dir}" \
    --output "${run_dir}/N" \
    2>&1 | tee "${run_dir}/N.log"

  printf '[%s] Stage: I\n' "${dataset}"
  python -u Baseline/run.py static \
    --condition I \
    --config "${config_path}" \
    --data "${data_dir}" \
    --output "${run_dir}/I" \
    2>&1 | tee "${run_dir}/I.log"

  printf '[%s] Stage: B\n' "${dataset}"
  python -u Baseline/run.py train \
    --config "${config_path}" \
    --data "${data_dir}" \
    --initialization "${run_dir}/shared_init" \
    --output "${run_dir}/B" \
    2>&1 | tee "${run_dir}/B.log"

  printf '[%s] Finished. Check status and completed_rounds in: %s/B/summary.json\n' "${dataset}" "${run_dir}"
done
