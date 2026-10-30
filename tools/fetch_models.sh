#!/usr/bin/env bash
# Download the braincharts files listed in tracula_braincharts/data/model-checksums.sha256 from the
# pinned upstream commit, verify their checksums and unpack the FA models. Used by the Dockerfile; can
# also be run locally:  tools/fetch_models.sh /path/to/model/dir
#
# Resulting layout (same paths as the braincharts repo):
#   DEST/models/lifespan_fa_24K_19sites.zip
#   DEST/models/FA_50K_19sites/<tract>/Models/{NM_0_0_estimate.pkl,meta_data.md}
#   DEST/docs/{site_ids_fa_19sites.txt,phenotypes_fa.txt}
#   DEST/UPSTREAM_COMMIT
set -euo pipefail

UPSTREAM_REPO="predictive-clinical-neuroscience/braincharts"
UPSTREAM_COMMIT="c017a3e0beca4a62982ebf6563a19fa5c0b7284c"

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
checksums="${here}/../tracula_braincharts/data/model-checksums.sha256"
dest="${1:?usage: fetch_models.sh DEST_DIR}"

mkdir -p "${dest}"
while read -r _sha path; do
    [[ -z "${path}" ]] && continue
    url="https://raw.githubusercontent.com/${UPSTREAM_REPO}/${UPSTREAM_COMMIT}/${path}"
    echo "fetching ${path}"
    mkdir -p "${dest}/$(dirname "${path}")"
    curl -fsSL --retry 3 -o "${dest}/${path}" "${url}"
done < "${checksums}"

(cd "${dest}" && sha256sum -c "${checksums}")

# The zip also holds macOS resource forks (__MACOSX/); skip them.
unzip -q -o "${dest}/models/lifespan_fa_24K_19sites.zip" -x '__MACOSX/*' -d "${dest}/models"
# The zip stores FA_50K_19sites/ as drwx------, and unzip keeps that; make everything readable so the
# container works when run as the calling user.
chmod -R a+rX "${dest}"
echo "${UPSTREAM_COMMIT}" > "${dest}/UPSTREAM_COMMIT"
