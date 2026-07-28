param(
    [Parameter(Mandatory = $true)]
    [string]$Installer
)

$ErrorActionPreference = "Stop"
$readyTitle = "Wisp [ready]"

if (-not (Test-Path -LiteralPath $Installer -PathType Leaf)) {
    throw "Windows installer not found: $Installer"
}

$extractDirectory = Join-Path ([System.IO.Path]::GetTempPath()) "wisp-windows-smoke-$([guid]::NewGuid())"
$process = $null
$previousSmokeTest = $env:WISP_SMOKE_TEST

try {
    New-Item -ItemType Directory -Path $extractDirectory | Out-Null
    & 7z x -y "-o$extractDirectory" $Installer | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "7z failed to extract $Installer"
    }

    $executable = Join-Path $extractDirectory "Wisp.exe"
    $env:WISP_SMOKE_TEST = "1"
    $process = Start-Process -FilePath $executable -WorkingDirectory $extractDirectory -PassThru

    $readyDeadline = [DateTime]::UtcNow.AddSeconds(30)
    do {
        if ($process.HasExited) {
            throw "Packaged Wisp.exe exited during startup with code $($process.ExitCode)"
        }
        $process.Refresh()
        if ($process.MainWindowTitle -eq $readyTitle -and $process.MainWindowHandle -ne [IntPtr]::Zero) {
            break
        }
        Start-Sleep -Milliseconds 250
    } while ([DateTime]::UtcNow -lt $readyDeadline)

    if ($process.MainWindowTitle -ne $readyTitle -or $process.MainWindowHandle -eq [IntPtr]::Zero) {
        throw "Packaged Wisp.exe did not render its frontend within 30 seconds"
    }

    if (-not $process.CloseMainWindow()) {
        throw "Packaged Wisp.exe did not accept a close request"
    }
    if (-not $process.WaitForExit(10000)) {
        throw "Packaged Wisp.exe rendered but its window became unresponsive while closing"
    }

    Write-Host "Packaged Wisp.exe rendered its frontend and closed responsively"
}
finally {
    $env:WISP_SMOKE_TEST = $previousSmokeTest
    if ($null -ne $process -and -not $process.HasExited) {
        Stop-Process -Id $process.Id -Force
        $process.WaitForExit()
    }
    if (Test-Path -LiteralPath $extractDirectory) {
        Remove-Item -LiteralPath $extractDirectory -Recurse -Force
    }
}
