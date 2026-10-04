<#
.SYNOPSIS
    Downloads the trained AIGVDet export from Google Drive into AIGVDET\models
    and verifies every file against SHA256SUMS.txt.

.EXAMPLE
    .\tools\pull_export.ps1
    .\tools\pull_export.ps1 -RcloneExe "C:\tools\rclone.exe"
#>
param(
    [string]$RcloneExe = "D:\rclone-v1.75.1-windows-amd64\rclone-v1.75.1-windows-amd64\rclone.exe",
    [string]$Remote = "gdrive:SPL3/Video-AIGVDet/export"
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$models = Join-Path $projectRoot "models"

if (-not (Test-Path $RcloneExe)) {
    $found = Get-Command rclone -ErrorAction SilentlyContinue
    if ($null -eq $found) { throw "rclone not found. Pass -RcloneExe <path to rclone.exe>." }
    $RcloneExe = $found.Source
}

New-Item -ItemType Directory -Force -Path $models | Out-Null
& $RcloneExe copy $Remote $models --progress
if ($LASTEXITCODE -ne 0) { throw "rclone copy failed (exit code $LASTEXITCODE)" }

$sums = Join-Path $models "SHA256SUMS.txt"
if (-not (Test-Path $sums)) { throw "SHA256SUMS.txt is missing from the download" }

$failures = 0
foreach ($line in Get-Content $sums) {
    if ([string]::IsNullOrWhiteSpace($line)) { continue }
    $parts = $line -split "  ", 2
    $expected = $parts[0].Trim().ToLower()
    $relative = $parts[1].Trim()
    $path = Join-Path $models ($relative -replace "/", "\")
    if (-not (Test-Path $path)) {
        Write-Host "MISSING   $relative"
        $failures++
        continue
    }
    $actual = (Get-FileHash -Algorithm SHA256 -Path $path).Hash.ToLower()
    if ($actual -ne $expected) {
        Write-Host "BAD HASH  $relative"
        $failures++
    } else {
        Write-Host "ok        $relative"
    }
}

if ($failures -gt 0) { throw "$failures file(s) failed verification. Re-run this script." }
Write-Host "Export verified in $models"
