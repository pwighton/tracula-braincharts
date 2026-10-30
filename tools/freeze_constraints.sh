#!/usr/bin/env bash
# Regenerate constraints.txt from the Python packages installed in a built image, so later builds resolve
# to the same versions:  tools/freeze_constraints.sh [IMAGE]   (default: pwighton/tracula-braincharts:latest)
# torch is installed separately (CPU-only index) and is pinned in the Dockerfile instead.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
image="${1:-pwighton/tracula-braincharts:latest}"

pins="$(docker run --rm --entrypoint pip "${image}" freeze --exclude tracula-braincharts --exclude torch)"

# Direct-reference pins ("name @ file:///...") point at paths that only existed where the package was built
# and break the next build; refuse to write them.
if grep -q ' @ ' <<< "${pins}"; then
    echo "error: pip freeze returned direct-reference pins; constraints.txt not changed:" >&2
    grep ' @ ' <<< "${pins}" >&2
    exit 1
fi

printf '%s\n' "${pins}" > "${here}/../constraints.txt"
echo "wrote $(wc -l < "${here}/../constraints.txt") pins to constraints.txt"
