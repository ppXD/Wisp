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
    # NSIS requires /D=<path> to be the final argument and does not accept quotes around it. The
    # generated smoke path contains no spaces, so pass the switch verbatim.
    $installerProcess = Start-Process `
        -FilePath $Installer `
        -ArgumentList @("/S", "/D=$installDirectory") `
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

    # Do not let a current GitHub runner hide a missing redistributable. Every VC++ runtime module
    # that is bundled for the current toolset and loaded at startup must come from the app
    # directory. Match against the discovered package set instead of a broad prefix: Windows itself
    # supplies msvcp_win.dll as part of the UCRT, and it must remain machine-wide.
    $bundledRuntimeNames = @{}
    Get-ChildItem -LiteralPath $installDirectory -File -Filter "*.dll" |
        Where-Object {
            $_.Name -match "^(concrt|msvcp|vcamp|vccorlib|vcomp|vcruntime).*\.dll$"
        } |
        ForEach-Object {
            $bundledRuntimeNames[$_.Name] = $true
        }
    $runtimeModules = @($process.Modules | Where-Object {
        $bundledRuntimeNames.ContainsKey($_.ModuleName)
    })
    if (-not ($runtimeModules | Where-Object { $_.ModuleName -ieq "msvcp140.dll" })) {
        throw "Packaged Wisp.exe did not load msvcp140.dll"
    }
    foreach ($module in $runtimeModules) {
        $expected = Join-Path $installDirectory $module.ModuleName
        if (-not [System.IO.Path]::GetFullPath($module.FileName).Equals(
            [System.IO.Path]::GetFullPath($expected),
            [System.StringComparison]::OrdinalIgnoreCase
        )) {
            throw "Packaged Wisp.exe loaded a machine-wide VC++ runtime: $($module.FileName)"
        }
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
