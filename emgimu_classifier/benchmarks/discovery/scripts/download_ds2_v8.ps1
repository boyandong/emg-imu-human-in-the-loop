param(
    [Parameter(Mandatory = $true)] [string] $OutputDirectory,
    [int] $ChunkBytes = 8388608
)

$ErrorActionPreference = 'Stop'
if ($ChunkBytes -lt 65536) { throw 'ChunkBytes must be at least 65536' }

$datasetApi = 'https://www.kaggle.com/api/v1/datasets/download/cinthyazuniga/ds2-emg-signals-three-force-type?datasetVersionNumber=8'
$expectedHost = 'storage.googleapis.com'
$expectedPath = '/kaggle-data-sets/8185127/13291049/bundle/archive.zip'

function Get-StorageUrl {
    try {
        $response = Invoke-WebRequest -Uri $datasetApi -MaximumRedirection 0 -TimeoutSec 30 -UseBasicParsing
        $location = $response.Headers['Location']
    } catch {
        if (-not $_.Exception.Response) { throw }
        $location = $_.Exception.Response.Headers.Location
    }
    if (-not $location) { throw 'Kaggle did not return an archive redirect' }
    $uri = [uri]$location
    if ($uri.Scheme -ne 'https' -or $uri.Host -ne $expectedHost -or $uri.AbsolutePath -ne $expectedPath) {
        throw 'Kaggle archive identity/redirect changed; inspect before continuing'
    }
    return $uri.AbsoluteUri
}

New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$directory = (Resolve-Path -LiteralPath $OutputDirectory).Path
$archive = Join-Path $directory 'ds2_kaggle_v8.zip'
$partial = Join-Path $directory 'ds2_kaggle_v8.zip.part'
$chunk = Join-Path $directory 'ds2_kaggle_v8.chunk'
$probe = Join-Path $directory 'ds2_kaggle_v8.probe'

$url = Get-StorageUrl
$head = Invoke-WebRequest -Uri $url -Headers @{ Range = 'bytes=0-0' } -TimeoutSec 60 -OutFile $probe -PassThru
$range = [string]@($head.Headers['Content-Range'])[0]
if ($head.StatusCode -ne 206 -or $range -notmatch '^bytes 0-0/(\d+)$') {
    throw "Cannot determine archive length: HTTP $($head.StatusCode), $range"
}
$total = [long]$Matches[1]
Remove-Item -LiteralPath $probe -Force
if (Test-Path -LiteralPath $archive) {
    if ((Get-Item -LiteralPath $archive).Length -ne $total) {
        throw 'Existing archive has an unexpected size; refusing to overwrite it'
    }
    Write-Host "Already downloaded: $archive ($total bytes)"
    exit 0
}
if (-not (Test-Path -LiteralPath $partial)) {
    [System.IO.File]::WriteAllBytes($partial, [byte[]]::new(0))
}
$position = [long](Get-Item -LiteralPath $partial).Length
if ($position -gt $total) { throw 'Partial archive exceeds expected length' }
Write-Host "DS2 v8 archive: $total bytes; resuming at $position bytes"
$lastPercent = -1
$chunksSinceRefresh = 0

while ($position -lt $total) {
    $end = [Math]::Min($position + $ChunkBytes - 1, $total - 1)
    $expectedLength = $end - $position + 1
    $downloaded = $false
    for ($attempt = 1; $attempt -le 4 -and -not $downloaded; $attempt++) {
        try {
            if ($chunksSinceRefresh -ge 40 -or $attempt -gt 1) {
                $url = Get-StorageUrl
                $chunksSinceRefresh = 0
            }
            $response = Invoke-WebRequest -Uri $url -Headers @{ Range = "bytes=$position-$end" } `
                -TimeoutSec 120 -OutFile $chunk -PassThru
            $returned = [string]@($response.Headers['Content-Range'])[0]
            if ($response.StatusCode -ne 206 -or
                $returned -ne "bytes $position-$end/$total" -or
                (Get-Item -LiteralPath $chunk).Length -ne $expectedLength) {
                throw "Unexpected chunk response: HTTP $($response.StatusCode), $returned"
            }
            $source = [System.IO.File]::OpenRead($chunk)
            $destination = [System.IO.File]::Open($partial, [System.IO.FileMode]::Open,
                [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
            try {
                if ($destination.Length -ne $position) { throw 'Partial archive changed during download' }
                $destination.Seek($position, [System.IO.SeekOrigin]::Begin) | Out-Null
                $source.CopyTo($destination)
                $destination.Flush()
            } finally {
                $source.Dispose()
                $destination.Dispose()
            }
            $position = $end + 1
            $chunksSinceRefresh++
            $downloaded = $true
            Remove-Item -LiteralPath $chunk -Force
            $percent = [int][Math]::Floor(100.0 * $position / $total)
            if ($percent -ge $lastPercent + 2 -or $position -eq $total) {
                Write-Host ("Download {0}% ({1}/{2} bytes)" -f $percent, $position, $total)
                $lastPercent = $percent
            }
        } catch {
            if (Test-Path -LiteralPath $chunk) { Remove-Item -LiteralPath $chunk -Force }
            if (Test-Path -LiteralPath $partial) {
                $partialFile = [System.IO.File]::Open($partial, [System.IO.FileMode]::Open,
                    [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
                try {
                    if ($partialFile.Length -lt $position) { throw 'Partial archive shrank unexpectedly' }
                    $partialFile.SetLength($position)
                } finally {
                    $partialFile.Dispose()
                }
            }
            if ($attempt -eq 4) { throw }
            Write-Host "Chunk at $position failed (attempt $attempt); retrying"
            Start-Sleep -Seconds 3
        }
    }
}

if ((Get-Item -LiteralPath $partial).Length -ne $total) { throw 'Final archive size mismatch' }
$digest = (Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash.ToLowerInvariant()
Move-Item -LiteralPath $partial -Destination $archive
Write-Host "DS2 v8 archive complete: $archive"
Write-Host "SHA-256: $digest"
