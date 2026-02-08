#!/bin/bash
# Create a RunPod GPU pod and configure .runpod.env for SSH access.
#
# Usage:
#   ./scripts/runpod-create.sh              # Default: cheapest available GPU
#   ./scripts/runpod-create.sh "NVIDIA RTX A4000"  # Specific GPU type
#
# Requires RUNPOD_API_KEY in .runpod.env

set -e

# Load config
if [ ! -f .runpod.env ]; then
    echo "Error: .runpod.env not found. Copy .runpod.env.template and add your RUNPOD_API_KEY."
    exit 1
fi
source .runpod.env

if [ -z "${RUNPOD_API_KEY}" ]; then
    echo "Error: RUNPOD_API_KEY not set in .runpod.env"
    echo "Get your key from https://www.runpod.io/console/user/settings"
    exit 1
fi

GPU_TYPE="${1:-NVIDIA RTX A4000}"
POD_NAME="cuvarbase-dev"
IMAGE="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
VOLUME_GB=20
DISK_GB=20
API_URL="https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}"

echo "Creating RunPod instance..."
echo "  GPU: ${GPU_TYPE}"
echo "  Image: ${IMAGE}"

# Create pod
RESPONSE=$(curl -s --request POST \
    --header 'content-type: application/json' \
    --url "${API_URL}" \
    --data "{\"query\": \"mutation { podFindAndDeployOnDemand(input: { cloudType: ALL, gpuCount: 1, volumeInGb: ${VOLUME_GB}, containerDiskInGb: ${DISK_GB}, minVcpuCount: 2, minMemoryInGb: 15, gpuTypeId: \\\"${GPU_TYPE}\\\", name: \\\"${POD_NAME}\\\", imageName: \\\"${IMAGE}\\\", ports: \\\"22/tcp\\\", volumeMountPath: \\\"/workspace\\\" }) { id costPerHr } }\"}")

# Extract pod ID
POD_ID=$(echo "${RESPONSE}" | python3 -c "
import sys, json
data = json.load(sys.stdin)
if 'errors' in data:
    print('ERROR: ' + data['errors'][0]['message'], file=sys.stderr)
    sys.exit(1)
pod = data['data']['podFindAndDeployOnDemand']
print(pod['id'])
" 2>&1)

if [[ "${POD_ID}" == ERROR:* ]]; then
    echo "${POD_ID}"
    echo ""
    echo "Full response: ${RESPONSE}"
    exit 1
fi

COST=$(echo "${RESPONSE}" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(data['data']['podFindAndDeployOnDemand']['costPerHr'])
")

echo "Pod created: ${POD_ID} (\$${COST}/hr)"
echo "Waiting for pod to start..."

# Poll until running and SSH is available
MAX_WAIT=180
WAITED=0
SSH_IP=""
SSH_PORT=""

while [ ${WAITED} -lt ${MAX_WAIT} ]; do
    sleep 5
    WAITED=$((WAITED + 5))

    STATUS_RESPONSE=$(curl -s --request POST \
        --header 'content-type: application/json' \
        --url "${API_URL}" \
        --data "{\"query\": \"query { pod(input: {podId: \\\"${POD_ID}\\\"}) { id desiredStatus runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort type } } } }\"}")

    # Parse status
    eval "$(echo "${STATUS_RESPONSE}" | python3 -c "
import sys, json
data = json.load(sys.stdin)
pod = data['data']['pod']
status = pod.get('desiredStatus', 'UNKNOWN')
print(f'POD_STATUS={status}')
runtime = pod.get('runtime')
if runtime and runtime.get('ports'):
    for port in runtime['ports']:
        if port['privatePort'] == 22 and port['isIpPublic']:
            print(f'SSH_IP={port[\"ip\"]}')
            print(f'SSH_PORT={port[\"publicPort\"]}')
")"

    printf "\r  Status: %-10s Waited: %ds" "${POD_STATUS}" "${WAITED}"

    if [ -n "${SSH_IP}" ] && [ -n "${SSH_PORT}" ]; then
        echo ""
        break
    fi
done

if [ -z "${SSH_IP}" ] || [ -z "${SSH_PORT}" ]; then
    echo ""
    echo "Error: Pod did not become SSH-ready within ${MAX_WAIT}s"
    echo "Pod ID: ${POD_ID} (check RunPod dashboard)"
    echo "Last status: ${POD_STATUS}"
    exit 1
fi

echo "SSH port reported: ${SSH_IP}:${SSH_PORT}"

SSH_KEY_OPT=""
if [ -f ~/.ssh/id_ed25519 ]; then
    SSH_KEY_OPT="-i ~/.ssh/id_ed25519"
fi

# Get podHostId for proxy SSH
echo "Getting proxy SSH credentials..."
POD_HOST_ID=$(curl -s --request POST \
    --header "content-type: application/json" \
    --url "${API_URL}" \
    --data "{\"query\": \"query { pod(input: {podId: \\\"${POD_ID}\\\"}) { machine { podHostId } } }\"}" \
    | python3 -c "import sys, json; print(json.load(sys.stdin)['data']['pod']['machine']['podHostId'])")

echo "Pod host ID: ${POD_HOST_ID}"

# Start SSHD via RunPod proxy (the image doesn't auto-start it)
echo "Starting SSH daemon via RunPod proxy..."
PROXY_SSH="ssh -tt -o ConnectTimeout=15 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ${SSH_KEY_OPT} ${POD_HOST_ID}@ssh.runpod.io"

echo 'ssh-keygen -A 2>/dev/null; service ssh start; mkdir -p /root/.ssh; chmod 700 /root/.ssh; echo "SSHD_SETUP_DONE"; exit' \
    | ${PROXY_SSH} 2>&1 | grep -q "SSHD_SETUP_DONE" && echo "SSHD started." || echo "Warning: SSHD setup may have failed."

# Add local SSH public key to authorized_keys
if [ -f ~/.ssh/id_ed25519.pub ]; then
    LOCAL_PUBKEY=$(cat ~/.ssh/id_ed25519.pub)
    echo "mkdir -p /root/.ssh && echo \"${LOCAL_PUBKEY}\" >> /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys && echo AUTH_OK; exit" \
        | ${PROXY_SSH} 2>&1 | grep -q "AUTH_OK" && echo "SSH key authorized." || echo "Warning: key setup may have failed."
fi

# Wait for direct SSH to accept connections
echo "Waiting for direct SSH..."
SSH_READY=false
SSH_WAIT=0
SSH_MAX_WAIT=30
while [ ${SSH_WAIT} -lt ${SSH_MAX_WAIT} ]; do
    if ssh -o ConnectTimeout=3 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes \
        ${SSH_KEY_OPT} -p ${SSH_PORT} root@${SSH_IP} "echo ok" >/dev/null 2>&1; then
        SSH_READY=true
        break
    fi
    sleep 3
    SSH_WAIT=$((SSH_WAIT + 3))
    printf "\r  SSH wait: %ds" "${SSH_WAIT}"
done
echo ""

if [ "${SSH_READY}" != true ]; then
    echo "Warning: Direct SSH not responding. Proxy SSH should still work."
fi

echo "SSH ready: ${SSH_IP}:${SSH_PORT}"

# Update .runpod.env with new connection details (preserve API key and other settings)
python3 -c "
import re

with open('.runpod.env', 'r') as f:
    content = f.read()

replacements = {
    'RUNPOD_SSH_HOST': '${SSH_IP}',
    'RUNPOD_SSH_PORT': '${SSH_PORT}',
    'RUNPOD_SSH_USER': 'root',
    'RUNPOD_POD_ID': '${POD_ID}',
}

for key, val in replacements.items():
    pattern = rf'^#?\s*{key}=.*$'
    replacement = f'{key}={val}'
    if re.search(pattern, content, re.MULTILINE):
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    else:
        content = content.rstrip() + f'\n{replacement}\n'

with open('.runpod.env', 'w') as f:
    f.write(content)
"

echo ""
echo "Updated .runpod.env with new connection details."
echo ""
echo "Pod ID:  ${POD_ID}"
echo "SSH:     ssh -i ~/.ssh/id_ed25519 -p ${SSH_PORT} root@${SSH_IP}"
echo "Cost:    \$${COST}/hr"
echo ""
echo "Next steps:"
echo "  ./scripts/setup-remote.sh                          # Install cuvarbase"
echo "  ./scripts/test-remote.sh cuvarbase/tests/test_tls_basic.py -v  # Run TLS tests"
echo "  ./scripts/runpod-stop.sh                           # Stop pod when done"
