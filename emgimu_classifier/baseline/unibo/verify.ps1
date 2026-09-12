param(
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$SourceRoot = Join-Path $RepositoryRoot "src"
$BenchmarkRoot = Join-Path $PSScriptRoot "benchmark"
$OfficialDataRoot = Join-Path $PSScriptRoot "official\data"

Push-Location $RepositoryRoot
try {
    $env:PYTHONPATH = $SourceRoot
    & $PythonExe -m emgimu.cli benchmark-check $BenchmarkRoot
    if ($LASTEXITCODE -ne 0) {
        throw "UniBo benchmark integrity check failed."
    }
    $OfficialSample = Get-ChildItem -LiteralPath $OfficialDataRoot -Filter "*.mat" -File |
        Select-Object -First 1
    if ($null -ne $OfficialSample) {
        $env:UNIBO_INAIL_SAMPLE = $OfficialSample.FullName
    }
    & $PythonExe -m unittest discover -s tests -p test_unibo_adapter.py -v
    if ($LASTEXITCODE -ne 0) {
        throw "UniBo adapter tests failed."
    }
}
finally {
    Pop-Location
}
