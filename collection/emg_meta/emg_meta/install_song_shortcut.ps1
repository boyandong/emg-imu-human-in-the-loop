param(
    [string]$PythonwPath = 'D:\miniconda\envs\emgforce\pythonw.exe',
    [string]$DesktopPath = [Environment]::GetFolderPath('Desktop')
)

$appRoot = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$entry = Join-Path $appRoot 'main.py'
$bundle = Join-Path $appRoot 'models/song_real8_f0_spd'
if (-not (Test-Path -LiteralPath $PythonwPath -PathType Leaf)) {
    throw "Python GUI interpreter not found: $PythonwPath"
}
foreach ($required in @($entry, (Join-Path $bundle 'song_manifest.json'),
                       (Join-Path $bundle 'song_f0_model.json'))) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Song application/model file not found: $required"
    }
}
if (-not (Test-Path -LiteralPath $DesktopPath -PathType Container)) {
    throw "Desktop directory not found: $DesktopPath"
}

$link = Join-Path $DesktopPath 'EMG-IMU 项目版（Song 8通道）.lnk'
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($link)
$shortcut.TargetPath = (Resolve-Path -LiteralPath $PythonwPath).Path
$shortcut.Arguments = '"' + $entry + '" --song-realtime'
$shortcut.WorkingDirectory = $appRoot
$shortcut.Description = 'Open Song eight-channel realtime recognition with the F0+SPD model selected'
$shortcut.Save()

$saved = $shell.CreateShortcut($link)
if ($saved.Arguments -ne $shortcut.Arguments -or
    $saved.TargetPath -ne $shortcut.TargetPath -or
    $saved.WorkingDirectory -ne $appRoot) {
    throw "Shortcut verification failed: $link"
}
Write-Output $link
