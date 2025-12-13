$ErrorActionPreference = "Stop"

param(
  [int]$N = 2000,
  [string]$Base,
  [string]$Prefix = "load:"
)

if (-not $PSBoundParameters.ContainsKey('Base') -or [string]::IsNullOrWhiteSpace($Base)) {
  $port = $env:PORT_RUST_KISS_VDB
  if ([string]::IsNullOrWhiteSpace($port)) {
    $port = "9917"
  }
  $Base = "http://localhost:$port"
}

Write-Host "Writing $N state keys..."
for ($i = 0; $i -lt $N; $i++) {
  $key = "$Prefix$i"
  $body = "{`"value`":{`"i`":$i}}"
  curl.exe -sS -X PUT "$Base/v1/state/$key" -H "Content-Type: application/json" -d $body | Out-Null
}

Write-Host "Done"
