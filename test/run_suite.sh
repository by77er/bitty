#!/usr/bin/env bash
# Regression suite: each mock asserts server-side, so any violation comes back
# as an ASSERTION 400 that the harness prints.
#
# Resolved before the cd below, or a relative $BASH_SOURCE (running as
# `bash run_suite.sh` from inside test/) points SCRATCH at the repo root and
# every mock dies at launch — which used to read as a wall of vacuous passes
# before the mock-liveness check existed.
SCRATCH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd /home/bit/Code/Bitty || exit 1
# Bash otherwise announces "Terminated" for every leftover process reaped
# below, which buries the actual results.
set +m
# Bracketed so the pattern cannot match the shell that is running this suite.
# Pids of harnesses this suite started, so cleanup never reaches beyond them.
STARTED=()
reap_started() {
  local pid
  for pid in "${STARTED[@]}"; do
    kill "$pid" 2>/dev/null
    wait "$pid" 2>/dev/null
  done
  STARTED=()
}
pass=0; fail=0

run() {
  local name=$1 port=$2 script=$3; shift 3
  # Only ever reap harnesses this suite started, tracked by pid. Matching on
  # the binary path would also kill a bitty someone is running for real, which
  # is not the suite's business.
  reap_started
  pkill -f "$script" >/dev/null 2>&1
  # Wait for the previous server to release the port, or the new one binds
  # nothing, dies silently, and the harness fails for an unrelated reason.
  for _ in $(seq 40); do
    (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null && exec 3<&- && sleep 0.1 || break
  done
  ( cd "$SCRATCH" && setsid nohup python3 "$script" >/dev/null 2>&1 & )
  for _ in $(seq 40); do
    (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null && exec 3<&- && break
    sleep 0.1
  done
  # A dead mock makes every turn fail with connection refused, which settles
  # and looks like a pass. Refuse to run the scenario instead.
  if ! (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
    printf 'FAIL  %s (mock never came up on :%s)\n' "$name" "$port"
    fail=$((fail+1))
    return
  fi
  exec 3<&-
  local out
  # Generous: spawning a script runs `deno check`, which costs real seconds.
  local start=$SECONDS
  local log; log=$(mktemp)
  ANTHROPIC_API_KEY=test ANTHROPIC_BASE_URL="http://127.0.0.1:$port" \
        timeout 180 ./target/debug/bitty "$@" > "$log" 2>&1 &
  local bpid=$!
  STARTED+=("$bpid")
  wait "$bpid" 2>/dev/null
  out=$(cat "$log"); rm -f "$log"
  local elapsed=$((SECONDS - start))
  reap_started
  pkill -f "$script" >/dev/null 2>&1
  if echo "$out" | grep -q "ASSERTION"; then
    printf 'FAIL  %s\n' "$name"; echo "$out" | grep -o "ASSERTION.*" | head -2 | sed 's/^/      /'
    fail=$((fail+1))
  elif echo "$out" | grep -q "settled"; then
    printf 'pass  %s (%ss)\n' "$name" "$elapsed"; pass=$((pass+1))
  else
    printf 'FAIL  %s (never settled)\n' "$name"; echo "$out" | tail -3 | sed 's/^/      /'
    fail=$((fail+1))
  fi
}

run "stop + self-stop + idempotent stop" 8734 mock_server.py    --once "compute 2+2"
run "topology + acl + inherited context" 8735 mock_topology.py  --once --role "coordinator" "draft and polish"
run "compaction round-trip + degrade"    8736 mock_compaction.py --once "write a haiku"
run "multicast + array stop + link signals"          8737 mock_notices.py    --once "coordinate three workers"

run "capabilities: attenuation + rejection"    8738 mock_caps.py      --once "delegate with least privilege"

run "script process (embedded deno)"          8739 mock_script.py    --once "use a script to count"

run "call_process + patch_script"                8740 mock_call.py      --once "compute with a script"

REPO=$(mktemp -d); SECRET=$(mktemp -d)
mkdir -p "$REPO/src"; echo 'fn main() {}' > "$REPO/src/main.rs"; echo TOPSECRET > "$SECRET/keys.txt"
export BITTY_TEST_REPO="$REPO" BITTY_TEST_SECRET="$SECRET"
run "filesystem capabilities"                   8741 mock_fs.py        --once --allow-read "$REPO" --allow-write "$REPO" "read the repo"

run "tool aliases (typed call to a process)"     8742 mock_alias.py     --once "wire a typed tool"

REPO2=$(mktemp -d); mkdir -p "$REPO2/src"; echo "fn main() {}" > "$REPO2/src/main.rs"
export BITTY_TEST_REPO="$REPO2"
run "typecheck at spawn + Deno API shim"       8746 mock_check.py     --once --allow-read "$REPO2" --allow-write "$REPO2" "build"

export BITTY_TEST_REPO="$REPO2"
run "inline typescript (run_script)"            8751 mock_inline.py    --once --allow-read "$REPO2" "compute inline"
run "tools follow capabilities"                8750 mock_tools.py     --once "spawn a leaf"
run "agent-to-agent call reply"                8749 mock_reply.py     --once "scout the env"
run "myopic worker (tools, no graph)"           8756 mock_myopic.py    --once "delegate with tools"
run "deno.serve in a script process"        8757 mock_serve.py     --once --allow-net 127.0.0.1:8899 "stand up a server"
run "reactive scripts (sleep + socket)"       8758 mock_reactive.py  --once --allow-net 127.0.0.1:8901 "be reactive"

# BITTY_COMPACT_ABOVE is in tokens; the harness estimates chars/4 when the
# mock reports tiny real usage, so 50000 tokens trips a couple of turns after
# the mock's ~180K chars of ballast, once the marker turns exist.
BITTY_COMPACT_ABOVE=50000 BITTY_COMPACTION=off \
  run "local compaction (summarise + restart)" 8760 mock_summarize.py --once "the original briefing"
unset BITTY_COMPACT_ABOVE BITTY_COMPACTION

run "overflow compacts and retries"          8761 mock_overflow.py  --once "grow then overflow"
run "artifact-backed long mailbox messages" 8793 mock_mailbox.py   --once "send a long report"

# The Codex provider needs the CLI's stored credentials just to start, so the
# scenario only runs where they exist.
if [ -f "$HOME/.codex/auth.json" ]; then
  BITTY_PROVIDER=codex BITTY_CODEX_URL=http://127.0.0.1:8770 \
    run "codex stream retry (mid-body EOF)"  8770 mock_codex_eof.py --once "say hi"
else
  echo "skip  codex stream retry (no ~/.codex/auth.json)"
fi

GATED=$(mktemp -d)
export BITTY_TEST_GATE_DIR="$GATED"
run "verification gate (fail, fix, pass)"    8762 mock_gate.py      --once --allow-read "$GATED" --allow-write "$GATED" --gate "test -f $GATED/marker" "do the work"

restart_arm() {
  local name="script survives a harness restart" port=8759 start=$SECONDS
  reap_started; pkill -f mock_restart.py >/dev/null 2>&1
  local dir markdir; dir=$(mktemp -d); markdir=$(mktemp -d)
  # Compact aggressively so a two-phase run actually exercises a checkpoint.
  export BITTY_COMPACT_FLOOR=1
  export BITTY_RESTART_MARK="$markdir/boots" BITTY_RESTART_PORT=$port
  local out=""
  for phase in 1 2; do
    ( cd "$SCRATCH" && BITTY_RESTART_PHASE=$phase setsid nohup python3 mock_restart.py >/dev/null 2>&1 & )
    for _ in $(seq 40); do
      (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null && exec 3<&- && break || sleep 0.1
    done
    if [ "$phase" = 1 ]; then
      out="$out$(ANTHROPIC_API_KEY=test ANTHROPIC_BASE_URL="http://127.0.0.1:$port" timeout 90 \
        ./target/debug/bitty --once --journal "$dir" --allow-read "$markdir" --allow-write "$markdir" "go" 2>&1)"
    else
      out="$out$(ANTHROPIC_API_KEY=test ANTHROPIC_BASE_URL="http://127.0.0.1:$port" timeout 90 \
        ./target/debug/bitty --once --journal "$dir" --resume "check" 2>&1)"
    fi
    pkill -f mock_restart.py >/dev/null 2>&1
    reap_started
    for _ in $(seq 40); do
      (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null && exec 3<&- && sleep 0.1 || break
    done
  done
  rm -rf "$dir" "$markdir"
  unset BITTY_COMPACT_FLOOR
  if echo "$out" | grep -q "ASSERTION"; then
    printf 'FAIL  %s\n' "$name"; echo "$out" | grep -o "ASSERTION.*" | head -2 | sed 's/^/      /'
    fail=$((fail+1))
  else
    printf 'pass  %s (%ss)\n' "$name" "$((SECONDS - start))"; pass=$((pass+1))
  fi
}
restart_arm


echo "---"
echo "$pass passed, $fail failed"
exit $((fail > 0))
