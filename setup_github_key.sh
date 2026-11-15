#!/bin/bash
set -u
KEY=/home/ubuntu/.ssh/id_ed25519_github
if [ ! -f "$KEY" ]; then
  ssh-keygen -t ed25519 -f "$KEY" -N "" -C "opencode-RKDD" >/dev/null
fi
chmod 600 "$KEY"
chmod 644 "$KEY.pub"
if ! grep -q "Host github.com" /home/ubuntu/.ssh/config 2>/dev/null; then
  cat >> /home/ubuntu/.ssh/config <<'EOF'
Host github.com
  HostName github.com
  User git
  IdentityFile ~/.ssh/id_ed25519_github
  IdentitiesOnly yes
EOF
fi
chmod 600 /home/ubuntu/.ssh/config
echo "--- public key ---"
cat "$KEY.pub"
