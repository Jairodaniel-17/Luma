$ErrorActionPreference = "Stop"

$base = "http://localhost:8080"
$auth = "Authorization: Bearer dev"

Write-Host "Create collection docs"
curl.exe -sS -X POST "$base/v1/vector/docs" -H $auth -H "Content-Type: application/json" -d "{\"dim\":3,\"metric\":\"cosine\"}" | Out-Host

Write-Host "Upsert vectors"
curl.exe -sS -X POST "$base/v1/vector/docs/upsert" -H $auth -H "Content-Type: application/json" -d "{\"id\":\"a\",\"vector\":[1,0,0],\"meta\":{\"tag\":\"x\",\"job_id\":\"j1\"}}" | Out-Host
curl.exe -sS -X POST "$base/v1/vector/docs/upsert" -H $auth -H "Content-Type: application/json" -d "{\"id\":\"b\",\"vector\":[0,1,0],\"meta\":{\"tag\":\"y\",\"job_id\":\"j2\"}}" | Out-Host

Write-Host "Put state job:123"
curl.exe -sS -X PUT "$base/v1/state/job:123" -H $auth -H "Content-Type: application/json" -d "{\"value\":{\"progress\":42},\"ttl_ms\":60000}" | Out-Host

Write-Host "Search"
curl.exe -sS -X POST "$base/v1/vector/docs/search" -H $auth -H "Content-Type: application/json" -d "{\"vector\":[0.9,0.1,0],\"k\":3,\"filters\":{\"tag\":\"x\"},\"include_meta\":true}" | Out-Host

