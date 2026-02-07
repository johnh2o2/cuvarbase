#!/bin/bash
# One-shot: create pod -> setup -> run tests -> stop pod.
#
# Usage:
#   ./scripts/gpu-test.sh                                          # Run all tests
#   ./scripts/gpu-test.sh cuvarbase/tests/test_tls_basic.py -v     # Specific tests
#   ./scripts/gpu-test.sh --keep cuvarbase/tests/test_tls_basic.py # Don't stop pod after

set -e

KEEP_POD=false
if [ "$1" = "--keep" ]; then
    KEEP_POD=true
    shift
fi

TEST_ARGS="${@:-cuvarbase/tests/test_tls_basic.py -v}"

echo "========================================"
echo "GPU Test: full lifecycle"
echo "========================================"
echo ""

# Step 1: Create pod (if not already running)
source .runpod.env 2>/dev/null || true

NEED_CREATE=true
if [ -n "${RUNPOD_POD_ID}" ] && [ -n "${RUNPOD_API_KEY}" ]; then
    # Check if existing pod is still running
    API_URL="https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}"
    STATUS=$(curl -s --request POST \
        --header 'content-type: application/json' \
        --url "${API_URL}" \
        --data "{\"query\": \"query { pod(input: {podId: \\\"${RUNPOD_POD_ID}\\\"}) { desiredStatus } }\"}" \
        | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    pod = data.get('data', {}).get('pod')
    print(pod['desiredStatus'] if pod else 'GONE')
except: print('GONE')
" 2>/dev/null)

    if [ "${STATUS}" = "RUNNING" ]; then
        echo "Reusing existing pod ${RUNPOD_POD_ID}"
        NEED_CREATE=false
    fi
fi

if [ "${NEED_CREATE}" = true ]; then
    echo "Step 1: Creating pod..."
    ./scripts/runpod-create.sh
    echo ""
    echo "Step 2: Setting up environment..."
    ./scripts/setup-remote.sh
else
    echo "Step 1: Pod already running, syncing code..."
    ./scripts/sync-to-runpod.sh
fi

echo ""
echo "Step 3: Running tests..."
echo "========================================"
./scripts/test-remote.sh ${TEST_ARGS}
TEST_EXIT=$?

echo ""
if [ "${KEEP_POD}" = true ]; then
    echo "Pod kept running (--keep flag). Stop with: ./scripts/runpod-stop.sh"
else
    echo "Step 4: Stopping pod..."
    ./scripts/runpod-stop.sh
fi

exit ${TEST_EXIT}
