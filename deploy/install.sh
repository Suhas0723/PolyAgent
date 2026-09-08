#!/usr/bin/env bash
set -euo pipefail

APP_USER="polyagent"
APP_DIR="/opt/polyagent"
LOG_DIR="/var/log/polyagent"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $EUID -ne 0 ]]; then
    echo "run as root: sudo bash deploy/install.sh" >&2
    exit 1
fi

echo "==> installing system packages"
apt-get update -qq
apt-get install -y -qq python3 python3-venv python3-pip sqlite3 logrotate

echo "==> creating service account"
if ! id -u "$APP_USER" >/dev/null 2>&1; then
    useradd --system --create-home --home-dir "$APP_DIR" --shell /usr/sbin/nologin "$APP_USER"
fi

echo "==> laying out $APP_DIR"
mkdir -p "$APP_DIR" "$APP_DIR/data" "$LOG_DIR"
rsync -a --delete \
    --exclude '.git' --exclude '.venv' --exclude 'data' --exclude '__pycache__' \
    "$SRC_DIR/" "$APP_DIR/"

if [[ ! -f "$APP_DIR/.env" ]]; then
    cp "$APP_DIR/.env.example" "$APP_DIR/.env"
    echo "    wrote $APP_DIR/.env from template; edit it before starting the service"
fi
chmod 600 "$APP_DIR/.env"

echo "==> building virtualenv"
python3 -m venv "$APP_DIR/.venv"
"$APP_DIR/.venv/bin/pip" install --upgrade pip -q
"$APP_DIR/.venv/bin/pip" install -q -e "$APP_DIR"
if [[ "${POLYAGENT_INSTALL_LIVE:-0}" == "1" ]]; then
    "$APP_DIR/.venv/bin/pip" install -q -r "$APP_DIR/requirements-live.txt"
fi

chown -R "$APP_USER:$APP_USER" "$APP_DIR" "$LOG_DIR"

echo "==> initializing database"
sudo -u "$APP_USER" "$APP_DIR/.venv/bin/polyagent" --env-file "$APP_DIR/.env" init-db

echo "==> installing systemd units"
install -m 644 "$APP_DIR/deploy/polyagent.service" /etc/systemd/system/polyagent.service
install -m 644 "$APP_DIR/deploy/polyagent-poller.service" /etc/systemd/system/polyagent-poller.service
install -m 644 "$APP_DIR/deploy/polyagent.logrotate" /etc/logrotate.d/polyagent

systemctl daemon-reload
systemctl enable polyagent.service

cat <<'MSG'

==> install complete

next steps:
  1. edit /opt/polyagent/.env and set ANTHROPIC_API_KEY
  2. systemctl start polyagent
  3. journalctl -u polyagent -f
  4. optional pair poller: systemctl enable --now polyagent-poller

MSG
