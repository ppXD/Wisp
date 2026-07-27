param(
    [Parameter(Mandatory = $true)]
    [string]$Installer
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $Installer -PathType Leaf)) {
    throw "Windows installer not found: $Installer"
}

$extractDirectory = Join-Path ([System.IO.Path]::GetTempPath()) "wisp-windows-smoke-$([guid]::NewGuid())"
$process = $null

try {
    New-Item -ItemType Directory -Path $extractDirectory | Out-Null
    & 7z x -y "-o$extractDirectory" $Installer | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "7z failed to extract $Installer"
    }

    $executable = Join-Path $extractDirectory "Wisp.exe"
    $process = Start-Process -FilePath $executable -WorkingDirectory $extractDirectory -PassThru
    if ($process.WaitForExit(15_000)) {
        throw "Packaged Wisp.exe exited during startup with code $($process.ExitCode)"
    }

    Write-Host "Packaged Wisp.exe remained running for 15 seconds"
}
finally {
    if ($null -ne $process -and -not $process.HasExited) {
        Stop-Process -Id $process.Id -Force
        $process.WaitForExit()
    }
    if (Test-Path -LiteralPath $extractDirectory) {
        Remove-Item -LiteralPath $extractDirectory -Recurse -Force
    }
}
