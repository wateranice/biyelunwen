# PyTorch + CUDA 12.4（Windows，Python 3.12 对应 cp312 轮子）
# 在「Anaconda Prompt」或 PowerShell 中执行： .\scripts\install_pytorch_cuda_cu124.ps1
# 若 pip 仍报哈希错误，请改用脚本末尾的 conda 命令。

$ErrorActionPreference = "Stop"
Write-Host ">>> 卸载 CPU/CUDA 旧版 torch / torchvision（若失败请先关闭占用 Python 的程序）..."
pip uninstall -y torch torchvision
if ($LASTEXITCODE -ne 0) {
    Write-Host "!!! pip uninstall 返回非零，若提示文件被占用，请关闭 IDE/Jupyter 后重试。"
}

Write-Host ">>> 清理 pip 缓存（避免损坏包导致 HASH 不匹配）..."
pip cache purge

Write-Host ">>> 安装 CUDA 12.4 版（约 2.5GB+，请保持网络稳定）..."
pip install torch torchvision `
  --index-url https://download.pytorch.org/whl/cu124 `
  --no-cache-dir `
  --default-timeout 600 `
  --retries 15

Write-Host ">>> 验证 CUDA ..."
python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('device_count', torch.cuda.device_count())"

Write-Host @"

若 pip 反复失败，可在已安装 NVIDIA 驱动的前提下用 conda（推荐）：
  conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia

装好后把 configs/default.yaml 里 server.device 改为 cuda。
"@
