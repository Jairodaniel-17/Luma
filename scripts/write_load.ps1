$ErrorActionPreference = "Stop"

param(
  [int]$N = 2000,
  [string]$Base = "http://localhost:8080",
  [string]$Prefix = "load:"
)

Write-Host "Writing $N state keys..."
for ($i = 0; $i -lt $N; $i++) {
  $key = "$Prefix$i"
  $body = "{`"value`":{`"i`":$i}}"
  curl.exe -sS -X PUT "$Base/v1/state/$key" -H "Content-Type: application/json" -d $body | Out-Null
}

Write-Host "Done"
