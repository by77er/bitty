#!/usr/bin/env bash
# Regression suite: each mock asserts server-side, so any violation comes back
# as an ASSERTION 400 that the harness prints.
cd /home/bit/Code/Bitty || exit 1
SCRATCH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pass=0; fail=0

run() {
  local name=$1 port=$2 script=$3; shift 3
  pkill -f "$script" >/dev/null 2>&1
  # Wait for the previous server to release the port, or the new one binds
  # nothing, dies silently, and the harness fails for an unrelated reason.
  for _ in $(seq 40); do
    (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null && exec 3<&- && sleep 0.1 || break
  done
  ( cd "$SCRATCH" && nohup python3 "$script" >/dev/null 2>&1 & )
  for _ in $(seq 40); do
    (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null && exec 3<&- && break
    sleep 0.1
  done
  local out
  out=$(ANTHROPIC_API_KEY=test ANTHROPIC_BASE_URL="http://127.0.0.1:$port" \
        timeout 60 ./target/debug/bitty "$@" 2>&1)
  pkill -f "$script" >/dev/null 2>&1
  if echo "$out" | grep -q "ASSERTION"; then
    printf 'FAIL  %s\n' "$name"; echo "$out" | grep -o "ASSERTION.*" | head -2 | sed 's/^/      /'
    fail=$((fail+1))
  elif echo "$out" | grep -q "settled"; then
    printf 'pass  %s\n' "$name"; pass=$((pass+1))
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

run "agent-to-agent call reply"                8749 mock_reply.py     --once "scout the env"

echo "---"
echo "$pass passed, $fail failed"
exit $((fail > 0))
