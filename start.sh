#!/bin/bash
set -e

# Export RUNPOD_* env vars so they're available in SSH sessions
env | grep RUNPOD_ >> /etc/rp_environment 2>/dev/null || true
if [ -f /etc/rp_environment ]; then
    grep -qF 'source /etc/rp_environment' /root/.bashrc 2>/dev/null \
        || echo 'source /etc/rp_environment' >> /root/.bashrc
fi

# SSH setup
if [ -n "$PUBLIC_KEY" ]; then
    mkdir -p /root/.ssh
    chmod 700 /root/.ssh
    echo "$PUBLIC_KEY" > /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
    /usr/sbin/sshd
    echo "SSH server started"
fi

# Run user-provided pre_start hook if present (Runpod convention)
if [ -f /pre_start.sh ]; then
    echo "Running pre_start.sh..."
    bash /pre_start.sh
fi

echo "Pod is ready"

sleep infinity
