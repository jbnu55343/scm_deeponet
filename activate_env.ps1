# PowerShell 脚本用于激活 pytorch_gpu 环境并运行命令
# 使用方式: .\activate_env.ps1 "python scripts/run_spatial_comparison.py"

param(
    [string]$Command = ""
)

# 设置执行策略
Set-ExecutionPolicy -ExecutionPolicy ByPass -Scope Process -Force

# 初始化 conda
$condaPath = "D:\DL\Scripts\conda.exe"
& $condaPath "shell.powershell" "hook" | Out-String | Invoke-Expression

# 激活虚拟环境
conda activate pytorch_gpu

# 执行命令
if ($Command) {
    Write-Host "执行命令: $Command" -ForegroundColor Green
    Invoke-Expression $Command
} else {
    Write-Host "pytorch_gpu 环境已激活！" -ForegroundColor Green
    Write-Host "Python 版本: $(python --version)" -ForegroundColor Green
    Write-Host "PyTorch 可用: $(python -c 'import torch; print(torch.__version__)')" -ForegroundColor Green
}
