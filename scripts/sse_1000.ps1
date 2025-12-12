$ErrorActionPreference = "Stop"

param(
  [int]$N = 1000,
  [string]$Base = "http://localhost:8080",
  [string]$ApiKey = "dev",
  [string]$Since = "0",
  [string]$Types = "state_updated,state_deleted,vector_added,vector_upserted,vector_updated,vector_deleted,gap"
)

$auth = "Authorization: Bearer $ApiKey"
$url = "$Base/v1/stream?since=$Since&types=$Types"

Write-Host "Starting $N SSE clients -> $url"
Write-Host "Stop: Get-Job | Stop-Job; Get-Job | Remove-Job"

1..$N | ForEach-Object {
  Start-Job -ScriptBlock {
    param($u, $h)
    curl.exe -N $u -H $h | Out-Null
  } -ArgumentList $url, $auth | Out-Null
}

Write-Host "Started: $(Get-Job | Measure-Object | Select-Object -ExpandProperty Count) jobs"

