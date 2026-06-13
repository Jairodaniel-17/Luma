# Luma DB — Windows PowerShell install script
# Downloads the latest release binary from GitHub and installs it.
# Usage:
#   irm https://raw.githubusercontent.com/Jairodaniel-17/rust-kiss-vdb/main/install.ps1 | iex
#   # or with a specific version / destination:
#   & ([scriptblock]::Create((irm 'https://.../install.ps1'))) -Version v3.1.0 -InstallDir "$env:LOCALAPPDATA\Programs\luma"

param(
    [string]$Version   = "latest",
    [string]$InstallDir = "$env:LOCALAPPDATA\Programs\luma"
)

$ErrorActionPreference = "Stop"
$Repo   = "Jairodaniel-17/rust-kiss-vdb"
$Binary = "luma.exe"

# ----- resolve version -----
if ($Version -eq "latest") {
    Write-Host "Fetching latest release tag..."
    $rel     = Invoke-RestMethod "https://api.github.com/repos/$Repo/releases/latest"
    $Version = $rel.tag_name
    Write-Host "Latest version: $Version"
}

# ----- build URL -----
$Target  = "windows-x86_64"
$Archive = "luma-$Version-$Target.zip"
$Url     = "https://github.com/$Repo/releases/download/$Version/$Archive"

# ----- download -----
$Tmp = Join-Path $env:TEMP "luma-install-$([System.IO.Path]::GetRandomFileName())"
New-Item -ItemType Directory -Force -Path $Tmp | Out-Null
$ZipPath = Join-Path $Tmp $Archive

Write-Host "Downloading $Url ..."
Invoke-WebRequest -Uri $Url -OutFile $ZipPath -UseBasicParsing

# ----- verify checksum (best-effort) -----
try {
    $SumsUrl  = "https://github.com/$Repo/releases/download/$Version/SHA256SUMS.txt"
    $SumsPath = Join-Path $Tmp "SHA256SUMS.txt"
    Invoke-WebRequest -Uri $SumsUrl -OutFile $SumsPath -UseBasicParsing
    $Expected = (Get-Content $SumsPath | Select-String $Archive) -replace '.* ', ''
    $Actual   = (Get-FileHash $ZipPath -Algorithm SHA256).Hash.ToLower()
    if ($Expected -and $Actual -ne $Expected) {
        Write-Error "Checksum mismatch!`n  expected: $Expected`n  actual:   $Actual"
    }
    Write-Host "Checksum OK."
} catch {
    Write-Warning "Could not verify checksum: $_"
}

# ----- extract -----
Expand-Archive -Path $ZipPath -DestinationPath $Tmp -Force
$ExePath = Get-ChildItem -Path $Tmp -Filter $Binary -Recurse | Select-Object -First 1 -ExpandProperty FullName

# ----- install -----
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
Copy-Item $ExePath -Destination (Join-Path $InstallDir $Binary) -Force

# ----- add to PATH for current user (persistent) -----
$CurrentPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($CurrentPath -notlike "*$InstallDir*") {
    [Environment]::SetEnvironmentVariable("Path", "$CurrentPath;$InstallDir", "User")
    Write-Host "Added $InstallDir to your PATH (restart your shell to take effect)."
}

# ----- cleanup -----
Remove-Item -Recurse -Force $Tmp

Write-Host ""
Write-Host "Luma DB $Version installed to $InstallDir\$Binary"
Write-Host ""
Write-Host "Quick start:"
Write-Host '  luma serve                          # start on port 8080'
Write-Host '  $env:LUMA_API_KEY="secret"; luma serve --port 1234'
Write-Host ""
Write-Host "Docs: https://github.com/$Repo#readme"
