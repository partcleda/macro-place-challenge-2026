#!/usr/bin/env bash
# Decisive teardown of the verification batch: kill orchestrator + workers,
# then remove all daemon-side containers (the actual compute).
pkill -9 -f 'swag_verify/orchestrate.py' 2>/dev/null
ps -eo pid,cmd 2>/dev/null | grep -E '[s]wag_verify/run_one.sh|[t]imeout --signal=KILL 1800' | awk '{print $1}' | xargs -r kill -9 2>/dev/null
docker ps -a --filter "name=swag_" -q | xargs -r docker rm -f >/dev/null 2>&1
sleep 2
echo "workers: $(ps -eo cmd 2>/dev/null | grep -c '[s]wag_verify/run_one.sh')  containers: $(docker ps -a --filter name=swag_ -q 2>/dev/null | wc -l)"
