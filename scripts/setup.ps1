$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $pythonPath)) {
    py -3.12 -m venv (Join-Path $projectRoot ".venv")
}

& $pythonPath -m pip install --upgrade pip
& $pythonPath -m pip install -e "${projectRoot}[dev,research]"

Write-Host "Workspace ready. In VS Code, select: $pythonPath"
