param([string]$Python = "python")
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
& $Python -c "import sys; assert sys.version_info >= (3,10); print(sys.executable)"
if ($LASTEXITCODE -ne 0) { throw "Provide a real Python 3.10+ executable with -Python; the WindowsApps alias is not an installed interpreter." }
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    & $Python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Could not create environment" }
}
& '.venv/Scripts/python.exe' -m pip install -e '.[dev]'
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed" }
Write-Host "Environment ready. No experiments started. Run .venv/Scripts/python.exe -m mbdnn plan"
