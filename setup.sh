#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

ENV_NAME="mbb-gs"
PYTHON_VERSION="3.10"
CUDA_VERSION="12.6"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu126"
CUDA_HOME_ARG=""

SKIP_ENV_CREATE=0
SKIP_PACKAGES=0
SKIP_CUDA_BUILD=0
SKIP_SMOKE_MEDIA=0
SKIP_TESTS=0


usage() {
    cat <<EOF
MBB-GS Linux setup

Options:
  --env-name NAME
  --python-version VERSION
  --cuda-version VERSION
  --cuda-home PATH
  --torch-index-url URL
  --skip-env-create
  --skip-packages
  --skip-cuda-build
  --skip-smoke-media
  --skip-tests
  -h, --help
EOF
}


step() {
    echo
    echo "============================================================"
    echo "$1"
    echo "============================================================"
}


fail() {
    echo "ERROR: $1" >&2
    exit 1
}


run() {
    echo ">> $*"
    "$@"
}


while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-name)
            ENV_NAME="$2"
            shift 2
            ;;
        --python-version)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --cuda-version)
            CUDA_VERSION="$2"
            shift 2
            ;;
        --cuda-home)
            CUDA_HOME_ARG="$2"
            shift 2
            ;;
        --torch-index-url)
            TORCH_INDEX_URL="$2"
            shift 2
            ;;
        --skip-env-create)
            SKIP_ENV_CREATE=1
            shift
            ;;
        --skip-packages)
            SKIP_PACKAGES=1
            shift
            ;;
        --skip-cuda-build)
            SKIP_CUDA_BUILD=1
            shift
            ;;
        --skip-smoke-media)
            SKIP_SMOKE_MEDIA=1
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 2
            ;;
    esac
done


find_conda() {
    if command -v conda >/dev/null 2>&1; then
        command -v conda
        return 0
    fi

    local candidates=(
        "$HOME/miniconda3/bin/conda"
        "$HOME/anaconda3/bin/conda"
        "$HOME/miniforge3/bin/conda"
    )

    local candidate
    for candidate in "${candidates[@]}"; do
        if [[ -x "$candidate" ]]; then
            echo "$candidate"
            return 0
        fi
    done

    return 1
}


echo
echo "MBB-GS Linux setup"
echo "Repository: $ROOT"
echo "Conda env:  $ENV_NAME"
echo "Python:     $PYTHON_VERSION"
echo "CUDA:       $CUDA_VERSION"


step "1. Detecting Conda"

CONDA_EXE="$(find_conda || true)"

if [[ -z "$CONDA_EXE" ]]; then
    fail "Conda was not found. Install Miniconda, Anaconda or Miniforge."
fi

echo "Conda: $CONDA_EXE"


step "2. Checking Conda environment"

if "$CONDA_EXE" run -n "$ENV_NAME" python --version >/dev/null 2>&1; then
    echo "Environment $ENV_NAME already exists."
else
    if [[ "$SKIP_ENV_CREATE" -eq 1 ]]; then
        fail "Environment $ENV_NAME does not exist and --skip-env-create was specified."
    fi

    run "$CONDA_EXE" create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
fi

run "$CONDA_EXE" run -n "$ENV_NAME" python --version


if [[ "$SKIP_PACKAGES" -eq 0 ]]; then
    step "3. Installing Python dependencies"

    run "$CONDA_EXE" run -n "$ENV_NAME" python -m pip install --upgrade pip setuptools wheel ninja

    run "$CONDA_EXE" run -n "$ENV_NAME" python -m pip install torch torchvision --index-url "$TORCH_INDEX_URL"

    if [[ -f "$ROOT/requirements.txt" ]]; then
        run "$CONDA_EXE" run -n "$ENV_NAME" python -m pip install -r "$ROOT/requirements.txt"
    fi

    run "$CONDA_EXE" run -n "$ENV_NAME" python -m pip install -e "$ROOT"
else
    step "3. Python dependency installation skipped"
fi


step "4. Detecting CUDA Toolkit"

if [[ -n "$CUDA_HOME_ARG" ]]; then
    CUDA_HOME="$CUDA_HOME_ARG"
elif [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME:-}/bin/nvcc" ]]; then
    CUDA_HOME="$CUDA_HOME"
elif [[ -x "/usr/local/cuda-${CUDA_VERSION}/bin/nvcc" ]]; then
    CUDA_HOME="/usr/local/cuda-${CUDA_VERSION}"
elif [[ -x "/usr/local/cuda/bin/nvcc" ]]; then
    CUDA_HOME="/usr/local/cuda"
elif command -v nvcc >/dev/null 2>&1; then
    NVCC_FOUND="$(command -v nvcc)"
    CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$NVCC_FOUND")")")"
else
    fail "CUDA Toolkit was not found. nvcc is required to build the CUDA extensions."
fi

if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
    fail "nvcc was not found at $CUDA_HOME/bin/nvcc"
fi

export CUDA_HOME
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"

if [[ -d "$CUDA_HOME/lib64" ]]; then
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi

echo "CUDA_HOME: $CUDA_HOME"
run "$CUDA_HOME/bin/nvcc" --version


step "5. Checking system tools"

if ! command -v g++ >/dev/null 2>&1; then
    fail "g++ was not found. Install the C++ build tools for your Linux distribution."
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
    fail "ffmpeg was not found in PATH."
fi

if ! command -v ffprobe >/dev/null 2>&1; then
    fail "ffprobe was not found in PATH."
fi

echo "G++:     $(command -v g++)"
echo "FFmpeg:  $(command -v ffmpeg)"
echo "FFprobe: $(command -v ffprobe)"

run g++ --version


if [[ "$SKIP_CUDA_BUILD" -eq 0 ]]; then
    step "6. Building raster_cuda"

    pushd "$ROOT/cuda/raster_cuda" >/dev/null
    run "$CONDA_EXE" run -n "$ENV_NAME" python setup.py build_ext --inplace
    popd >/dev/null


    step "7. Building gabor_audio_cuda"

    pushd "$ROOT/cuda/gabor_audio_cuda" >/dev/null
    run "$CONDA_EXE" run -n "$ENV_NAME" python setup.py build_ext --inplace
    popd >/dev/null
else
    step "6-7. CUDA build skipped"
fi


if [[ "$SKIP_SMOKE_MEDIA" -eq 0 ]]; then
    step "8. Generating synthetic smoke media"

    run "$CONDA_EXE" run -n "$ENV_NAME" python "$ROOT/scripts/generate_smoke_media.py" --force
else
    step "8. Smoke media generation skipped"
fi


step "9. Running environment doctor"

DOCTOR_ARGS=(
    run
    -n "$ENV_NAME"
    python
    "$ROOT/scripts/doctor.py"
)

if [[ "$SKIP_TESTS" -eq 0 ]]; then
    DOCTOR_ARGS+=(--tests)
fi

run "$CONDA_EXE" "${DOCTOR_ARGS[@]}"


echo
echo "============================================================"
echo "MBB-GS setup completed successfully"
echo "============================================================"
echo
echo "Activate the environment with:"
echo "conda activate $ENV_NAME"
echo
echo "Run the doctor with:"
echo "python scripts/doctor.py --tests"
echo
echo "Run the smoke pipeline with:"
echo "python scripts/pipeline/run_pipeline_video_audio.py --config configs/examples/smoke_20frames/pipeline.json"
echo
