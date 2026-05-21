# Push local source + processed data to the remote workspace.

set -euo pipefail
cd "$(dirname "$0")/.."

REMOTE="${REMOTE:-h100}"
REMOTE_DIR="${REMOTE_DIR:-/home/coder/yes-no}"

command -v tar >/dev/null || { echo "[sync] tar required on local"; exit 1; }
command -v ssh >/dev/null || { echo "[sync] ssh required on local"; exit 1; }

echo "[sync] $(pwd) -> ${REMOTE}:${REMOTE_DIR}"

COPYFILE_DISABLE=1 tar -C "$(pwd)" \
    --no-mac-metadata 2>/dev/null \
    --exclude='./results' \
    --exclude='./.venv' \
    --exclude='./__pycache__' \
    --exclude='*/__pycache__' \
    --exclude='./.git' \
    --exclude='./.DS_Store' \
    --exclude='*/.DS_Store' \
    --exclude='./._*' \
    --exclude='*/._*' \
    -cf - . 2>/dev/null \
    | ssh "$REMOTE" "mkdir -p '${REMOTE_DIR}' \
        && tar -C '${REMOTE_DIR}' --warning=no-unknown-keyword -xf - \
        && find '${REMOTE_DIR}' -name '._*' -type f -delete"

echo "[sync] done."
