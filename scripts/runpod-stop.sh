#!/bin/bash
# Stop (or terminate) the RunPod pod.
#
# Usage:
#   ./scripts/runpod-stop.sh            # Stop (can resume later, keeps volume)
#   ./scripts/runpod-stop.sh --terminate # Terminate (deletes everything)

set -e

if [ ! -f .runpod.env ]; then
    echo "Error: .runpod.env not found"
    exit 1
fi
source .runpod.env

if [ -z "${RUNPOD_API_KEY}" ]; then
    echo "Error: RUNPOD_API_KEY not set in .runpod.env"
    exit 1
fi

if [ -z "${RUNPOD_POD_ID}" ]; then
    echo "Error: RUNPOD_POD_ID not set in .runpod.env (no active pod?)"
    exit 1
fi

API_URL="https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}"

if [ "$1" = "--terminate" ]; then
    echo "Terminating pod ${RUNPOD_POD_ID}..."
    RESPONSE=$(curl -s --request POST \
        --header 'content-type: application/json' \
        --url "${API_URL}" \
        --data "{\"query\": \"mutation { podTerminate(input: {podId: \\\"${RUNPOD_POD_ID}\\\"}) }\"}")
    echo "Pod terminated."
else
    echo "Stopping pod ${RUNPOD_POD_ID}..."
    RESPONSE=$(curl -s --request POST \
        --header 'content-type: application/json' \
        --url "${API_URL}" \
        --data "{\"query\": \"mutation { podStop(input: {podId: \\\"${RUNPOD_POD_ID}\\\"}) { id desiredStatus } }\"}")
    echo "Pod stopped. Resume later from the RunPod dashboard, or re-run ./scripts/runpod-create.sh"
fi
