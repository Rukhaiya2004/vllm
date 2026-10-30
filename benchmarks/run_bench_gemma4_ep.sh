#!/usr/bin/env bash
# Benchmark Gemma-4-26B-A4B-it with Expert Parallel on Power (pwr-ep-enable branch).
#
# vLLM's OMPProcessManager automatically assigns each rank its own NUMA node's
# cores and sets OMP_NUM_THREADS/OMP_PLACES/OMP_PROC_BIND per rank.
# Do NOT set OMP_NUM_THREADS manually — it overrides vLLM's binding logic.
#
# Usage (run from a shell with the vllm_dev env active):
#   MODEL=/home/rukhaiya/pwr_ep/vllm/gemma-4-26B-A4B-it \
#   MODEL_GB=55 \
#   EP=4 \
#   LABEL=ep4 \
#   PORT=8001 \
#   numactl --interleave=all \
#   ./run_bench_gemma4_ep.sh
#
# For TP=2 comparison on main branch:
#   MODEL=/home/rukhaiya/pwr_ep/vllm/gemma-4-26B-A4B-it \
#   MODEL_GB=55 \
#   TP=2 \
#   LABEL=tp2-main \
#   numactl --interleave=all \
#   ./run_bench_gemma4_ep.sh
#
# Keep summarize.py in the same folder as this script.

MODEL=${MODEL:-/home/rukhaiya/pwr_ep/vllm/gemma-4-26B-A4B-it}
NAME=${NAME:-$(basename "$MODEL" | tr 'A-Z' 'a-z')}
PORT=${PORT:-8001}
TP=${TP:-1}
EP=${EP:-4}                           # data-parallel-size (EP ranks); set 1 to disable
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
MAX_BATCHED_TOKENS=${MAX_BATCHED_TOKENS:-8192}
KV_GB=${KV_GB:-8}                    # per-rank KV cache in GiB (use --kv-cache-memory-bytes)
MODEL_GB=${MODEL_GB:-55}
PROMPTS="${PROMPTS:-1 4 8 16}"
SHAPES="${SHAPES:-512:1024}"
MAX_NUM_SEQS=${MAX_NUM_SEQS:-16}
LABEL=${LABEL:-ep${EP}}
OUT=${OUT:-$HOME/bench_results/$NAME-$LABEL-$(date +%Y%m%d_%H%M%S)}

# ---------- 1. memory check ----------
avail_gb=$(awk '/MemAvailable/ {printf "%d", $2/1024/1024}' /proc/meminfo)
# EP=4 uses EP ranks * KV_GB for total KV, but each rank only owns KV_GB
total_kv_gb=$((EP * KV_GB))
need_gb=$((total_kv_gb + MODEL_GB))
echo ">>> memory check: available ${avail_gb} GiB, need ~${need_gb} GiB (KV ${total_kv_gb} + weights ~${MODEL_GB})"
if (( avail_gb < need_gb )); then
  echo "!!! ERROR: not enough free memory. Free some or lower KV_GB."
  exit 1
fi

command -v vllm >/dev/null || { echo "!!! vllm not on PATH (activate vllm_dev)"; exit 1; }
mkdir -p "$OUT/json"

# ---------- hardware / software info ----------
{
  echo "### date";   date -Is
  echo "### lscpu";  lscpu
  echo "### memory"; free -h
  echo "### versions"
  python3 -c "import vllm, torch; print('vllm', vllm.__version__); print('torch', torch.__version__)" 2>&1
  echo "### env";    env | grep -E '^(VLLM|OMP|HF_|NUMA)' | sed 's/HF_TOKEN=.*/HF_TOKEN=<hidden>/'
  echo "### top at start"; top -bn1 -w 200 | head -15
} > "$OUT/hardware_info.txt" 2>&1

# ---------- which vllm build / branch ----------
VLLM_SRC=$(python3 -c "import os, vllm; print(os.path.dirname(os.path.dirname(vllm.__file__)))" 2>/dev/null)
{
  echo "label:        ${LABEL}"
  echo "vllm version: $(python3 -c 'import vllm; print(vllm.__version__)' 2>/dev/null)"
  if git -C "$VLLM_SRC" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "source:       $VLLM_SRC"
    echo "branch:       $(git -C "$VLLM_SRC" rev-parse --abbrev-ref HEAD)"
    echo "commit:       $(git -C "$VLLM_SRC" rev-parse HEAD)"
    echo "subject:      $(git -C "$VLLM_SRC" log -1 --format=%s)"
    echo "uncommitted changes (first 20):"; git -C "$VLLM_SRC" status --short | head -20
  else
    echo "vllm source is not a git checkout: ${VLLM_SRC:-unknown}"
  fi
  echo "TP=${TP}  EP=${EP}  OMP_NUM_THREADS=${OMP_NUM_THREADS:-unset}"
} > "$OUT/branch_info.txt" 2>&1
echo ">>> build under test:"; cat "$OUT/branch_info.txt"

# ---------- refuse to start if port is busy ----------
if curl -sf localhost:$PORT/health >/dev/null 2>&1; then
  echo "!!! something is already serving on port $PORT."
  echo "    Stop it first: fuser -k ${PORT}/tcp 2>/dev/null; pkill -f 'vllm serve'"
  exit 1
fi

# ---------- 2. start server ----------
# EP mode uses --data-parallel-size; TP mode uses --tensor-parallel-size.
# --kv-cache-memory-bytes is the correct absolute-bytes flag for CPU backend
#   (avoids NUMA-node OOM that --gpu-memory-utilization causes on Power).
KV_BYTES=$(( KV_GB * 1024 * 1024 * 1024 ))

# AOT compile (inductor) on first request takes ~5-6 min per rank on Power.
# The default VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS is 300s which is exactly at
# the compile boundary — bump to 900s so first-compile requests don't timeout.
# --limit-mm-per-prompt expects JSON: '{"image":0,"video":0}'
LIMIT_MM=${LIMIT_MM:-'{"image":0,"video":0}'}
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-900}
echo ">>> VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS}"

if (( EP > 1 )); then
  echo ">>> starting server: EP=${EP} TP=${TP} port=${PORT}"
  vllm serve "$MODEL" \
    --served-model-name "$NAME" \
    --dtype bfloat16 \
    --tensor-parallel-size "$TP" \
    --data-parallel-size "$EP" \
    --max-model-len $MAX_MODEL_LEN \
    --max-num-batched-tokens $MAX_BATCHED_TOKENS \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --kv-cache-memory-bytes $KV_BYTES \
    --no-enable-prefix-caching \
    --limit-mm-per-prompt "$LIMIT_MM" \
    --port $PORT ${SERVE_ARGS:-} > >(tee "$OUT/server.log") 2>&1 &
else
  echo ">>> starting server: TP=${TP} port=${PORT}"
  vllm serve "$MODEL" \
    --served-model-name "$NAME" \
    --dtype bfloat16 \
    --tensor-parallel-size "$TP" \
    --max-model-len $MAX_MODEL_LEN \
    --max-num-batched-tokens $MAX_BATCHED_TOKENS \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --kv-cache-memory-bytes $KV_BYTES \
    --no-enable-prefix-caching \
    --limit-mm-per-prompt "$LIMIT_MM" \
    --port $PORT ${SERVE_ARGS:-} > >(tee "$OUT/server.log") 2>&1 &
fi

SERVER_PID=$!
trap 'echo ">>> killing server (pid $SERVER_PID)"; kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null' EXIT

# ---------- wait for server ready (AOT compile happens lazily — wait for first token) ----------
echo ">>> waiting for /health ..."
while ! curl -sf localhost:$PORT/health >/dev/null; do
  kill -0 $SERVER_PID 2>/dev/null || { echo "!!! server died, see $OUT/server.log"; exit 1; }
  sleep 5
done
echo ">>> /health OK — waiting for AOT compile (first real request may take ~75s) ..."

# ---------- 3. AOT compile warmup — one concurrent request per DP rank ----------
# With EP=4, each DP rank compiles independently on its first forward pass
# (~33s on Power).  A single curl only warms one rank; the other 3 compile
# during execute_dummy_batch, hit the 60s RPC timeout, and crash.
# Fix: send EP concurrent requests simultaneously so all ranks compile at
# once before the timed sweep starts.  max-time 300s covers the compile.
echo ">>> AOT compile warmup (${EP} concurrent requests — one per DP rank) ..."
python3 - <<PYWARMUP
import asyncio, aiohttp, sys

async def req(session, port, name):
    url = f"http://localhost:{port}/v1/chat/completions"
    payload = {"model": name,
               "messages": [{"role": "user", "content": "What is 2+2?"}],
               "max_tokens": 16, "temperature": 0}
    async with session.post(url, json=payload) as r:
        body = await r.json()
        return r.status, body["choices"][0]["message"]["content"]

async def main():
    port, name, ep = $PORT, "$NAME", $EP
    timeout = aiohttp.ClientTimeout(total=900)
    async with aiohttp.ClientSession(timeout=timeout) as s:
        results = await asyncio.gather(*[req(s, port, name) for _ in range(ep)],
                                       return_exceptions=True)
    ok = all(isinstance(r, tuple) and r[0] == 200 for r in results)
    for i, r in enumerate(results):
        print(f"  rank~{i}: {r}")
    sys.exit(0 if ok else 1)

asyncio.run(main())
PYWARMUP
if [[ $? -ne 0 ]]; then
  echo "!!! AOT warmup failed. Check $OUT/server.log"; exit 1
fi
echo ">>> AOT warmup done — all DP ranks compiled"

# ---------- 4. bench helper ----------
bench() {  # in out concurrency num_prompts result_name
  vllm bench serve \
    --backend openai-chat --endpoint /v1/chat/completions \
    --base-url http://localhost:$PORT --model "$NAME" --tokenizer "$MODEL" \
    --dataset-name random --random-input-len $1 --random-output-len $2 --ignore-eos \
    --max-concurrency $3 --num-prompts $4 --seed 0 --disable-tqdm \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,99 \
    --save-result --result-dir "$OUT/json" --result-filename "$5.json"
}

# ---------- 6. sweep ----------
for s in $SHAPES; do
  for n in $PROMPTS; do
    name=i${s%%:*}_o${s##*:}_c$n
    echo; echo ">>> $name  (num-prompts=$n, max-concurrency=$n)"
    bench ${s%%:*} ${s##*:} $n $n $name 2>&1 | tee "$OUT/$name.log"
  done
done

# ---------- 7. summary ----------
echo
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ -f "$SCRIPT_DIR/summarize.py" ]]; then
  python3 "$SCRIPT_DIR/summarize.py" "$OUT" | tee "$OUT/summary.txt"
else
  echo "!!! summarize.py not found next to this script ($SCRIPT_DIR). Copy it there."
fi
echo "Results in $OUT"
