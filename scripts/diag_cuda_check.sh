#!/usr/bin/env bash
#SBATCH --job-name=diag_cuda
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00

ROOT_DIR="/home/r84368868/lerobot-so101-data"
cd "$ROOT_DIR"

CONDA_BIN_DIR="/home/r84368868/miniconda3/bin"
CONDA_ENV_PATH="/home/r84368868/envs/lerobot/"
source "$CONDA_BIN_DIR/../etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_PATH"
export PATH="${CONDA_ENV_PATH%/}/bin:$PATH"

echo "host: $(hostname)"
echo "which python: $(command -v python)"
nvidia-smi --query-gpu=name,driver_version --format=csv
echo "which lerobot-train: $(command -v lerobot-train)"

python - <<'PYEOF'
import torch, torchvision
print("torch:", torch.__version__, "| cuda build:", torch.version.cuda)
print("torchvision:", torchvision.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    x = torch.randn(2048, 2048, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print("matmul OK on", torch.cuda.get_device_name(0), "| mean:", y.mean().item())
else:
    raise SystemExit("CUDA still not available — fix did not work")
PYEOF
