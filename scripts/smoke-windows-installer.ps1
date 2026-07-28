param(
    [Parameter(Mandatory = $true)]
    [string]$Installer
)

$ErrorActionPreference = "Stop"
$readyTitle = "Wisp [ready]"

if (-not (Test-Path -LiteralPath $Installer -PathType Leaf)) {
    throw "Windows installer not found: $Installer"
}

$installDirectory = Join-Path ([System.IO.Path]::GetTempPath()) "wisp-windows-smoke-$([guid]::NewGuid())"
$process = $null
$previousSmokeTest = $env:WISP_SMOKE_TEST

try {
    # Exercise the actual NSIS install path. Extracting the archive directly can miss install-time
    # placement/renaming mistakes and does not reproduce the way users launch Wisp.
    $installerProcess = Start-Process `
        -FilePath $Installer `
        -ArgumentList @("/S", "/D=`"$installDirectory`"") `
        -PassThru `
        -Wait `
        -WindowStyle Hidden
    if ($installerProcess.ExitCode -ne 0) {
        throw "Wisp installer failed with code $($installerProcess.ExitCode)"
    }

    $executable = Join-Path $installDirectory "Wisp.exe"
    if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
        throw "Installed Wisp.exe not found: $executable"
    }

    $env:WISP_SMOKE_TEST = "1"
    $process = Start-Process -FilePath $executable -WorkingDirectory $installDirectory -PassThru

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

    # Do not let a current GitHub runner hide a missing redistributable. The process must use the
    # app-local MSVC runtime that was built alongside whisper.cpp, not System32's possibly older copy.
    $msvcp = $process.Modules |
        Where-Object { $_.ModuleName -ieq "msvcp140.dll" } |
        Select-Object -First 1
    $expectedMsvcp = Join-Path $installDirectory "msvcp140.dll"
    if ($null -eq $msvcp) {
        throw "Packaged Wisp.exe did not load msvcp140.dll"
    }
    if (-not [System.IO.Path]::GetFullPath($msvcp.FileName).Equals(
        [System.IO.Path]::GetFullPath($expectedMsvcp),
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        throw "Packaged Wisp.exe loaded the machine-wide VC++ runtime: $($msvcp.FileName)"
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

    $uninstaller = Join-Path $installDirectory "uninstall.exe"
    if (Test-Path -LiteralPath $uninstaller -PathType Leaf) {
        $uninstallProcess = Start-Process `
            -FilePath $uninstaller `
            -ArgumentList "/S" `
            -PassThru `
            -Wait `
            -WindowStyle Hidden
        if ($uninstallProcess.ExitCode -ne 0) {
            Write-Warning "Wisp uninstaller failed with code $($uninstallProcess.ExitCode)"
        }
    }

    if (Test-Path -LiteralPath $installDirectory) {
        $resolvedInstall = [System.IO.Path]::GetFullPath($installDirectory)
        $resolvedTemp = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath())
        if (-not $resolvedInstall.StartsWith(
            $resolvedTemp,
            [System.StringComparison]::OrdinalIgnoreCase
        ) -or [System.IO.Path]::GetFileName($resolvedInstall) -notlike "wisp-windows-smoke-*") {
            throw "Refusing to remove unexpected smoke-test directory: $resolvedInstall"
        }
        Remove-Item -LiteralPath $resolvedInstall -Recurse -Force
    }
}
