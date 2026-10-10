param(
    [string]$PythonwPath = 'D:\miniconda\envs\emgforce\pythonw.exe',
    [string]$DesktopPath = [Environment]::GetFolderPath('Desktop'),
    [string]$ShortcutName = 'EMG-IMU 项目版（Song 8通道）.lnk'
)

$appRoot = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $appRoot '../../..')).Path
$entry = Join-Path $appRoot 'main_decision_v2.py'
$classifierRoot = Join-Path $repoRoot 'emgimu_classifier'
foreach ($required in @($PythonwPath, $entry,
    (Join-Path $classifierRoot 'benchmarks/song_real8/song_extended_window_v1/source_bank.pkl'),
    (Join-Path $classifierRoot 'feature_bank/SONG_PERSONAL_SESSION_ACCEPTANCE_V1.json'),
    (Join-Path $classifierRoot 'benchmarks/song_real8/song_extended_window_v1/policy.json'),
    (Join-Path $classifierRoot 'benchmarks/song_real8/SONG_EXTENDED_WINDOW_V1_RESULTS.json'),
    (Join-Path $classifierRoot 'benchmarks/song_real8/song_raw_quality_v1/source_gate.pkl'),
    (Join-Path $classifierRoot 'benchmarks/song_real8/SONG_RAW_QUALITY_V1_RESULTS.json'))) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required decision application file not found: $required"
    }
}
if (-not (Test-Path -LiteralPath $DesktopPath -PathType Container)) {
    throw "Desktop directory not found: $DesktopPath"
}
if ([IO.Path]::GetFileName($ShortcutName) -ne $ShortcutName -or
    [IO.Path]::GetExtension($ShortcutName) -ne '.lnk') {
    throw 'ShortcutName must be a .lnk filename without a directory'
}
$link = Join-Path $DesktopPath $ShortcutName
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($link)
$shortcut.TargetPath = (Resolve-Path -LiteralPath $PythonwPath).Path
$shortcut.Arguments = '"' + $entry + '"'
$shortcut.WorkingDirectory = $appRoot
$shortcut.Description = 'Open the current eight-channel personal/session calibration and realtime decision page'
$shortcut.Save()
$saved = $shell.CreateShortcut($link)
if ($saved.TargetPath -ne $shortcut.TargetPath -or
    $saved.Arguments -ne $shortcut.Arguments -or
    $saved.WorkingDirectory -ne $appRoot) {
    throw "Shortcut verification failed: $link"
}
Write-Output $link
