<#
.SYNOPSIS
    Windows (PowerShell) equivalent of the Makefile.

.DESCRIPTION
    Run from anywhere; the script switches to the repo root. Uses .venv\Scripts\python.exe
    when the virtual environment exists, otherwise the first `python` on PATH.

.EXAMPLE
    .\scripts\tasks.ps1 install
    .\scripts\tasks.ps1 test
    .\scripts\tasks.ps1 eval
    $env:RAG_API_KEY = "<key>"; .\scripts\tasks.ps1 load-test -Url https://<service>.run.app
#>
param(
    [Parameter(Position = 0)]
    [ValidateSet("help", "install", "index", "run", "test", "lint", "eval-retrieval", "ablate",
                 "eval-smoke", "eval", "eval-no-reflection", "readme", "review", "failures",
                 "calibrate-export", "calibrate-score", "docker-build", "docker-run",
                 "monitoring-up", "load-test", "deploy")]
    [string]$Task = "help",
    [string]$Url = ""
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

# UTF-8 for every file and console write, whatever the Windows code page (usually cp1252).
$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

$Py = "python"
if (Test-Path ".venv\Scripts\python.exe") { $Py = ".venv\Scripts\python.exe" }

function Invoke-Step {
    param([string]$Exe, [string[]]$Arguments)
    Write-Host "> $Exe $($Arguments -join ' ')" -ForegroundColor Cyan
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code ${LASTEXITCODE}: $Exe" }
}

switch ($Task) {
    "help" {
        Get-Help $PSCommandPath -Detailed
        Write-Host "Tasks: install index run test lint eval-retrieval ablate eval-smoke eval eval-no-reflection"
        Write-Host "       readme review failures calibrate-export calibrate-score docker-build docker-run"
        Write-Host "       monitoring-up load-test deploy"
    }
    "install" {
        Invoke-Step $Py @("-m", "pip", "install", "torch==2.14.0", "--index-url", "https://download.pytorch.org/whl/cpu")
        Invoke-Step $Py @("-m", "pip", "install", "-r", "requirements-dev.txt")
    }
    "index"              { Invoke-Step $Py @("-m", "app.rag.build_index") }
    "run"                { Invoke-Step $Py @("-m", "uvicorn", "app.main:create_app", "--factory", "--reload", "--port", "8081") }
    "test"               { Invoke-Step $Py @("-m", "pytest") }
    "lint"               { Invoke-Step $Py @("-m", "ruff", "check", ".") }
    "eval-retrieval" {
        Invoke-Step $Py @("-m", "eval.retrieval_eval", "--split", "test")
        Invoke-Step $Py @("-m", "eval.retrieval_eval", "--split", "dev")
    }
    "ablate"             { Invoke-Step $Py @("-m", "eval.ablate_retrieval") }
    "eval-smoke"         { Invoke-Step $Py @("-m", "eval.run_eval", "--split", "test", "--limit", "10", "--tag", "smoke", "--yes") }
    "eval"               { Invoke-Step $Py @("-m", "eval.run_eval", "--split", "test") }
    "eval-no-reflection" { Invoke-Step $Py @("-m", "eval.run_eval", "--split", "test", "--max-refinements", "0", "--tag", "no-reflection") }
    "readme"             { Invoke-Step $Py @("-m", "eval.update_readme") }
    "review"             { Invoke-Step $Py @("-m", "eval.review_golden", "--split", "test", "--out", "review_test_split.md") }
    "failures"           { Invoke-Step $Py @("-m", "eval.inspect_failures", "--out", "eval_failures.md") }
    "calibrate-export"   { Invoke-Step $Py @("-m", "eval.judge_calibration", "export") }
    "calibrate-score"    { Invoke-Step $Py @("-m", "eval.judge_calibration", "score") }
    "docker-build"       { Invoke-Step "docker" @("build", "-t", "agent-rag-api", ".") }
    # The container listens on 8080 (Cloud Run's default $PORT); expose it on localhost:8081.
    "docker-run"         { Invoke-Step "docker" @("run", "--rm", "-p", "8081:8080", "--env-file", ".env", "agent-rag-api") }
    "monitoring-up"      { Invoke-Step "docker" @("compose", "-f", "monitoring/docker-compose.yml", "up", "--build") }
    "load-test" {
        if (-not $Url) { throw "Pass the service URL: .\scripts\tasks.ps1 load-test -Url https://<service>.run.app" }
        if (-not $env:RAG_API_KEY) { throw 'Set the API key first: $env:RAG_API_KEY = "<key>"' }
        Invoke-Step $Py @("-m", "eval.load_test", "--url", $Url, "--n", "30", "--concurrency", "2")
    }
    "deploy" {
        Write-Host "Deployment uses bash + gcloud. Run it from Google Cloud Shell (shell.cloud.google.com):"
        Write-Host "  git clone https://github.com/twktan/agent-rag-api.git && cd agent-rag-api"
        Write-Host "  PROJECT_ID=<your-project-id> ./scripts/deploy.sh"
    }
}
