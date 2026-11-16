<#
.SYNOPSIS
  Build and run the API and UI together with docker compose, secrets passed as environment
  variables exactly as ECS does.

.DESCRIPTION
  Reads the secret from AWS Secrets Manager with the 'aero' profile (read-only). The five values
  are held only in this PowerShell process's environment while `docker compose up` runs; compose
  copies them into the API container by name (docker-compose.yml lists names, not values). They
  are never printed, never written to disk, and removed from this process when the script ends.
  The UI container never receives them.

.EXAMPLE
  aws sso login --profile aero
  .\scripts\compose_up.ps1
  .\scripts\compose_up.ps1 -Down
#>
param(
    [string]$SecretName = "akinsds1126-aero-rag/app",
    [string]$Profile = "aero",
    [string]$Region = "eu-west-2",
    [switch]$Down
)
$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")

if ($Down) {
    $ErrorActionPreference = "Continue"
    docker compose --project-directory $Root down
    exit $LASTEXITCODE
}

$SecretKeys = @(
    "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY",
    "OPENSEARCH_ENDPOINT", "OPENSEARCH_USERNAME", "OPENSEARCH_PASSWORD"
)

$json = aws secretsmanager get-secret-value --secret-id $SecretName --query SecretString `
    --output text --profile $Profile --region $Region
if ($LASTEXITCODE -ne 0 -or -not $json) {
    throw "Could not read the secret. Run: aws sso login --profile $Profile"
}
$secret = $json | ConvertFrom-Json
$json = $null

try {
    foreach ($key in $SecretKeys) {
        $value = $secret.$key
        if (-not $value) { throw "The secret has no value for $key." }
        Set-Item -Path "Env:$key" -Value $value
    }
    # --wait returns once both containers report healthy (or fails if one does not).
    # Docker writes progress to stderr; judge success by the exit code, not by stderr output.
    $ErrorActionPreference = "Continue"
    docker compose --project-directory $Root up --build --detach --wait
    $code = $LASTEXITCODE
    $ErrorActionPreference = "Stop"
    if ($code -ne 0) { throw "docker compose up failed. See: docker compose logs" }
}
finally {
    foreach ($key in $SecretKeys) { Remove-Item -Path "Env:$key" -ErrorAction SilentlyContinue }
    $secret = $null
}

Write-Host "UI:   http://localhost:8501"
Write-Host "API:  http://localhost:8000/docs"
Write-Host "Logs: docker compose logs -f      Stop: .\scripts\compose_up.ps1 -Down"
