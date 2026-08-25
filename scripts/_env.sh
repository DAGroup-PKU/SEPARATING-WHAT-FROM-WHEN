# Shared by infer.sh / train.sh. Source this file; do not run it.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="${REPO_ROOT}/packages/ltx-core/src:${REPO_ROOT}/packages/ltx-pipelines/src:${REPO_ROOT}/packages/ltx-trainer/src:${PYTHONPATH:-}"

SITE_PACKAGES="$("${PYTHON}" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")"
if [[ -d "${SITE_PACKAGES}/nvidia" ]]; then
  NV_LIB_PATHS="$(find "${SITE_PACKAGES}/nvidia" -maxdepth 3 -type d -name lib | tr '\n' ':')"
  export LD_LIBRARY_PATH="${NV_LIB_PATHS}${LD_LIBRARY_PATH:-}"
fi
TORCH_LIB="$("${PYTHON}" -c 'import torch, pathlib; print(pathlib.Path(torch.__file__).parent / "lib")' 2>/dev/null || true)"
if [[ -n "${TORCH_LIB}" ]]; then
  export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH:-}"
fi
