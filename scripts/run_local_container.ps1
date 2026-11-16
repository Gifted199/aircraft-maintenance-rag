<#
.SYNOPSIS
  Run the API image locally the way ECS will: secrets arrive as environment variables.

.DESCRIPTION
  Reads the secret from AWS Secrets Manager with the 'aero' profile (read-only), then starts the
  container. The five secret values are:
    - held only in this PowerShell process's memory,
    - passed to docker by NAME (-e NAME), so docker copies the value from this process's
      environment and the value never appears on a command line,
    - never printed and never written to disk,
    - removed from this process's environment when the script ends.
  Inside the container the app sees all five as environment variables and therefore never calls
  Secrets Manager, exactly as on ECS.

.EXAMPLE
  aws sso login --profile aero
  .\scripts\run_local_container.ps1
  .\scripts\run_local_container.ps1 -Stop
#>
param(
    [string]$Image = "akinsds1126-aero-rag-api:0.2.0",
    [string]$Name = "akinsds1126-aero-rag-api",
    [int]$Port = 8000,
    [string]$SecretName = "akinsds1126-aero-rag/app",
    [string]$Profile = "aero",
    [string]$Region = "eu-west-2",
    [switch]$Stop
)
$ErrorActionPreference = "Stop"

$SecretKeys = @(
    "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY",
    "OPENSEARCH_ENDPOINT", "OPENSEARCH_USERNAME", "OPENSEARCH_PASSWORD"
)
$OptionalKeys = @("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST")
# Non-secret settings come from .env, except the two that only make sense on a laptop.
$LocalOnly = @("AWS_PROFILE", "APP_SECRET_NAME")

docker rm -f $Name 2>$null | Out-Null
if ($Stop) { Write-Host "Stopped $Name."; exit 0 }

$json = aws secretsmanager get-secret-value --secret-id $SecretName --query SecretString `
    --output text --profile $Profile --region $Region
if ($LASTEXITCODE -ne 0 -or -not $json) {
    throw "Could not read the secret. Run: aws sso login --profile $Profile"
}
$secret = $json | ConvertFrom-Json
$json = $null

$dockerArgs = @("run", "--detach", "--name", $Name, "--publish", "${Port}:8000")
try {
    foreach ($key in $SecretKeys) {
        $value = $secret.$key
        if (-not $value) { throw "The secret has no value for $key." }
        Set-Item -Path "Env:$key" -Value $value
        $dockerArgs += @("--env", $key)          # name only: docker reads the value itself
    }
    foreach ($key in $OptionalKeys) {            # tracing: off if absent
        if ($secret.$key) {
            Set-Item -Path "Env:$key" -Value $secret.$key
            $dockerArgs += @("--env", $key)
        }
    }
    $envFile = Join-Path $PSScriptRoot "..\.env"
    if (Test-Path $envFile) {
        foreach ($line in Get-Content $envFile) {
            if ($line -match '^\s*([A-Z0-9_]+)\s*=\s*(.*)$' -and $LocalOnly -notcontains $Matches[1]) {
                $dockerArgs += @("--env", "$($Matches[1])=$($Matches[2].Trim())")
            }
        }
    }
    $dockerArgs += $Image
    $containerId = docker @dockerArgs
    if ($LASTEXITCODE -ne 0) { throw "docker run failed." }
}
finally {
    foreach ($key in $SecretKeys + $OptionalKeys) {
        Remove-Item -Path "Env:$key" -ErrorAction SilentlyContinue
    }
    $secret = $null
}

Write-Host "Started $Name ($($containerId.Substring(0, 12))) on http://localhost:$Port"
Write-Host "Logs:  docker logs $Name      Stop:  .\scripts\run_local_container.ps1 -Stop"
