#!/usr/bin/env bash
# Benchmark google/gemma-4-26B-A4B-it: vllm serve + vllm bench serve (CPU / Power10)
#
# Flow: memory check -> start server (logs in terminal + server.log) -> wait for ready
#       -> sanity-check curl -> warmup -> sweep (shapes x num-prompts) -> summary table
#
# Other model:  MODEL=ibm-granite/granite-4.2-8b MODEL_GB=17 ./run_bench_gemma4.sh
#   (MODEL_GB = approx bf16 weight size, used by the memory check)
#
# Run once per vllm build, then compare:
#   (feature branch checked out + rebuilt)  LABEL=feature ./run_bench_gemma4.sh
#   (main checked out + rebuilt)            LABEL=main    ./run_bench_gemma4.sh
#
# Run from a shell where vllm_dev is active and your env vars are set (nothing is sourced here).
# The model is fetched from Hugging Face on first start. If the repo is gated for you,
# run `huggingface-cli login` (or export HF_TOKEN) first.

MODEL=${MODEL:-google/gemma-4-26B-A4B-it}
NAME=${NAME:-$(basename "$MODEL" | tr 'A-Z' 'a-z')}   # served name, derived from MODEL
PORT=${PORT:-8000}
TP=${TP:-1}
MAX_MODEL_LEN=8192
MAX_BATCHED_TOKENS=8192
KV_GB=${KV_GB:-80}                 # VLLM_CPU_KVCACHE_SPACE (GiB)
MODEL_GB=${MODEL_GB:-55}           # approx bf16 weight footprint of the 26B model; set 0 to check KV only
PROMPTS="${PROMPTS:-1 4 8 16}"     # --num-prompts; --max-concurrency is set equal (one wave)
SHAPES="${SHAPES:-512:1024}"   # input:output
MAX_NUM_SEQS=${MAX_NUM_SEQS:-16}   # >= highest concurrency
LABEL=${LABEL:-}                   # e.g. LABEL=feature or LABEL=main, used in the results folder name
OUT=${OUT:-$HOME/bench_results/$NAME${LABEL:+-$LABEL}-$(date +%Y%m%d_%H%M%S)}

# ---------- 1. memory check: fail early if KV cache (+ weights) will not fit ----------
avail_gb=$(awk '/MemAvailable/ {printf "%d", $2/1024/1024}' /proc/meminfo)
need_gb=$((KV_GB + MODEL_GB))
echo ">>> memory check: available ${avail_gb} GiB, need ${need_gb} GiB (KV ${KV_GB} + weights ~${MODEL_GB})"
if (( avail_gb < need_gb )); then
  echo "!!! ERROR: not enough free memory for a ${KV_GB} GiB KV cache. Free some memory or lower KV_GB."
  exit 1
fi

command -v vllm >/dev/null || { echo "!!! vllm not on PATH (activate vllm_dev)"; exit 1; }
mkdir -p "$OUT/json"
export VLLM_CPU_KVCACHE_SPACE=$KV_GB

# ---------- hardware / software info ----------
{
  echo "### date";   date -Is
  echo "### lscpu";  lscpu
  echo "### memory"; free -h
  echo "### versions"
  python3 -c "import vllm, torch; print('vllm', vllm.__version__); print('torch', torch.__version__)" 2>&1
  echo "### env";    env | grep -E '^(VLLM|OMP|HF_)' | sed 's/HF_TOKEN=.*/HF_TOKEN=<hidden>/'
  echo "### top at start"; top -bn1 -w 200 | head -15
} > "$OUT/hardware_info.txt" 2>&1

# ---------- which vllm build is this? (branch / commit of the source tree) ----------
VLLM_SRC=$(python3 -c "import os, vllm; print(os.path.dirname(os.path.dirname(vllm.__file__)))" 2>/dev/null)
{
  echo "label:        ${LABEL:-none}"
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
} > "$OUT/branch_info.txt" 2>&1
echo ">>> build under test:"; cat "$OUT/branch_info.txt"

if [[ "$MODEL" == *[Gg]emma-4* ]]; then
python3 - <<'PY' 2>/dev/null
# Gemma 4 recipe asks for vLLM 0.19.1+; warn only
import vllm
from packaging.version import parse
if parse(vllm.__version__) < parse("0.19.1"):
    print(f"!!! WARNING: vllm {vllm.__version__} is older than 0.19.1, which the Gemma 4 recipe lists as the minimum")
PY
fi

# ---------- refuse to start if something is already serving on the port ----------
if curl -sf localhost:$PORT/health >/dev/null 2>&1; then
  echo "!!! something is already serving on port $PORT (probably a server left over from an earlier run):"
  curl -s localhost:$PORT/v1/models | python3 -c "import sys,json; print('    models:', [m['id'] for m in json.load(sys.stdin)['data']])" 2>/dev/null
  echo "    Stop it first (e.g. pkill -f 'vllm serve'), or run with PORT=8001"
  exit 1
fi

# ---------- 2. server (output shows in terminal AND goes to server.log) ----------
vllm serve "$MODEL" --served-model-name "$NAME" --dtype bfloat16 \
  --tensor-parallel-size "$TP" --max-model-len $MAX_MODEL_LEN \
  --max-num-batched-tokens $MAX_BATCHED_TOKENS --max-num-seqs "$MAX_NUM_SEQS" \
  --no-enable-prefix-caching \
  --port $PORT ${SERVE_ARGS:-} > >(tee "$OUT/server.log") 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null' EXIT

while ! curl -sf localhost:$PORT/health >/dev/null; do
  kill -0 $SERVER_PID 2>/dev/null || { echo "!!! server died, see $OUT/server.log"; exit 1; }
  sleep 5
done
# make sure the server that answered is the one we started, and serves the expected name
kill -0 $SERVER_PID 2>/dev/null || { echo "!!! our server died but port $PORT answered: another server is using it. See $OUT/server.log"; exit 1; }
served=$(curl -s localhost:$PORT/v1/models | python3 -c "import sys,json; print(' '.join(m['id'] for m in json.load(sys.stdin)['data']))" 2>/dev/null)
echo ">>> server ready, serving: $served"
case " $served " in *" $NAME "*) ;; *) echo "!!! expected model '$NAME' is not being served"; exit 1;; esac

# ---------- 3. sanity check: one real chat request, print the output ----------
echo ">>> sanity check (curl)"
resp=$(curl -s -w '\n%{http_code}' localhost:$PORT/v1/chat/completions \
  -H 'Content-Type: application/json' -d @- <<EOF
{"model": "$NAME",
 "messages": [{"role": "user", "content": "Explain in two sentences what a mixture-of-experts model is."}],
 "max_tokens": 128, "temperature": 0}
EOF
)
code=${resp##*$'\n'}; body=${resp%$'\n'*}
echo "$body" > "$OUT/sanity_check.json"
echo "HTTP $code"
echo "$body" | python3 -m json.tool 2>/dev/null || echo "$body"
text=$(echo "$body" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'] or '')" 2>/dev/null)
echo "----- model output -----"; echo "$text"; echo "------------------------"
if [[ "$code" != "200" || -z "${text//[[:space:]]/}" ]]; then
  echo "!!! sanity check failed (HTTP $code or empty output), stopping before the benchmark"; exit 1
fi

# ---------- 4. bench helper, warmup ----------
bench() {  # in out concurrency num_prompts result_name
  vllm bench serve --backend openai-chat --endpoint /v1/chat/completions \
    --base-url http://localhost:$PORT --model "$NAME" --tokenizer "$MODEL" \
    --dataset-name random --random-input-len $1 --random-output-len $2 --ignore-eos \
    --max-concurrency $3 --num-prompts $4 --seed 0 --disable-tqdm \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,99 \
    --save-result --result-dir "$OUT/json" --result-filename "$5.json"
}
echo ">>> warmup"
bench 128 32 4 4 warmup > "$OUT/warmup.log" 2>&1 || { echo "!!! warmup failed, see $OUT/warmup.log"; exit 1; }
rm -f "$OUT/json/warmup.json"

# ---------- 5. sweep: num-prompts == concurrency ----------
for s in $SHAPES; do
  for n in $PROMPTS; do
    name=i${s%%:*}_o${s##*:}_c$n
    echo; echo ">>> $name  (num-prompts=$n, max-concurrency=$n)"
    bench ${s%%:*} ${s##*:} $n $n $name 2>&1 | tee "$OUT/$name.log"
  done
done

# ---------- 6. summary ----------
echo
if [[ -f "$(dirname "$0")/summarize.py" ]]; then
  python3 "$(dirname "$0")/summarize.py" "$OUT" | tee "$OUT/summary.txt"
fi
echo "Results in $OUT"
