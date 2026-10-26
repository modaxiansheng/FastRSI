#!/usr/bin/env bash
set -uo pipefail
project=/mnt/data/task/yuchen/FastPhyagentos/FastRSI
assets=/mnt/data/task/yuchen/FastPhyagentos/frozen_archive/extra-freeze-20260929T145255Z/frozen/assets-20260929-v1
image=sha256:3171fda2362d773433482f79c4fd53f6f428413efbf41c3fdfb43d8c799d6c95
run=public_bowl_replay_v3_20261009
exports=public_bowl_replay_v1_20261009
receipts="$project/planning/$run/receipts"
mkdir -p "$receipts"
run_demo() {
  demo="$1"
  name="yuchen-fastrsi-public-bowl-${demo}-v3-20261009"
  # Duplicate container names fail rather than repeating actions unnoticed.
  start=$(date +%s)
  timeout 480 docker run --name "$name" --network none --cpus 4 --memory 12g \
    --pids-limit 1024 --shm-size 2g --entrypoint python \
    -e PYTHONPATH=/workspace/openpi/third_party/libero:/project/scripts \
    -e LIBERO_CONFIG_PATH=/libero-config -e MUJOCO_GL=egl -e LIBGL_ALWAYS_SOFTWARE=1 \
    -v "$project:/project" -v "$assets/dependencies/openpi:/workspace/openpi:ro" \
    -v "$assets/dependencies/PhyAgentOS-core:/core:ro" -v "$project/configs/libero:/libero-config:ro" \
    "$image" /project/scripts/replay_public_bowl_demonstration_v1.py \
    --demo "/project/planning/$exports/exports/$demo" --output "/project/runs/$run/$demo" \
    > "$receipts/${demo}.log" 2>&1
  code=$?
  if [ "$code" -eq 124 ]; then docker stop -t 10 "$name" >> "$receipts/${demo}.log" 2>&1; fi
  docker inspect "$name" > "$receipts/${demo}_CONTAINER.json" 2>&1
  python3 - "$receipts/${demo}_HOST.json" "$code" "$start" <<'PY'
import json,sys,time
with open(sys.argv[1], 'x') as stream:
    json.dump(dict(host_exit_code=int(sys.argv[2]), host_wall_seconds=time.time()-int(sys.argv[3]),
                   allocated_cpus=4, memory_limit_gib=12, gpu_devices=0,
                   scope='Offline demo qualification, three CPU workers in parallel; not method speed comparison'), stream, indent=2)
PY
  echo "$demo exit=$code"
}
if [ "$#" -eq 0 ]; then set -- demo_0 demo_25 demo_49; fi
for demo in "$@"; do run_demo "$demo" & done
wait
