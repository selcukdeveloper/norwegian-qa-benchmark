# Pull results/ from the remote workspace back to the local project.

set -euo pipefail
cd "$(dirname "$0")/.."

REMOTE="${REMOTE:-h100}"
REMOTE_DIR="${REMOTE_DIR:-/home/coder/yes-no}"

command -v tar >/dev/null || { echo "[sync] tar required on local"; exit 1; }
command -v ssh >/dev/null || { echo "[sync] ssh required on local"; exit 1; }

mkdir -p results
echo "[sync] ${REMOTE}:${REMOTE_DIR}/results -> $(pwd)/results"

ssh "$REMOTE" "tar -C '${REMOTE_DIR}' -cf - results" \
    | COPYFILE_DISABLE=1 tar -xf - -C "$(pwd)"

echo "[sync] done."
