# Deploying to ECS

Every command here changes AWS and is run by you (CLAUDE.md rule 2). PowerShell syntax. Start
every session with:

```powershell
aws sso login --profile aero
$P = @("--profile", "aero", "--region", "eu-west-2")
$Registry = "666127452756.dkr.ecr.eu-west-2.amazonaws.com"
$Cluster = "akinsds1126-aero-rag-cluster"
$Service = "akinsds1126-aero-rag-api-service-jk8marb6"
$Sg = "sg-02fa227fd6bdc5931"
```

| Revision | API image | UI image | What changed |
|---|---|---|---|
| 1 | 0.1.0 | none | API only, port 8000 public |
| 2 | 0.2.0 | 0.2.0 | chat UI joins the task, Langfuse tracing, 8501 public, 8000 closed |
| **3** | 0.2.0 | **0.2.1** | USWDS-aligned UI, small-talk replies, aircraft chip follows detection |

# Release 0.2.1: UI only (task definition revision 3)

Only the UI container's image changes; the API stays on `0.2.0`, the secrets, ports, sizes and
security group are unchanged (checked against the live revision 2 on 16 November 2026). The file
is [deploy/ecs/task-definition-rev3.json](../deploy/ecs/task-definition-rev3.json); unlike the
revision 2 file it already has the secret ARN filled in, so it can be registered as is.

## 1. Check the local image is the one you tested

`.\scripts\compose_up.ps1` built `akinsds1126-aero-rag-ui:0.2.1` and is serving it at
http://localhost:8501. ECR tags are immutable: once `0.2.1` is pushed it can never be replaced,
so push only when you are happy with what you see there.

```powershell
docker compose ps --format "table {{.Name}}\t{{.Image}}\t{{.Status}}"   # ui: ...-ui:0.2.1 (healthy)
```

## 2. Tag and push the UI image

```powershell
aws ecr get-login-password @P | docker login --username AWS --password-stdin $Registry
docker tag akinsds1126-aero-rag-ui:0.2.1 "$Registry/akinsds1126-aero-rag-ui:0.2.1"
docker push "$Registry/akinsds1126-aero-rag-ui:0.2.1"
aws ecr describe-images @P --repository-name akinsds1126-aero-rag-ui `
  --query "imageDetails[].imageTags" --output text      # expect 0.2.0 and 0.2.1
```

## 3. Register revision 3 and deploy it

```powershell
aws ecs register-task-definition @P --cli-input-json file://deploy/ecs/task-definition-rev3.json `
  --query "taskDefinition.[family,revision]" --output text           # expect: ... 3
aws ecs update-service @P --cluster $Cluster --service $Service `
  --task-definition akinsds1126-aero-rag-api:3 --query "service.deployments[0].rolloutState"
aws ecs wait services-stable @P --cluster $Cluster --services $Service
```

The new task starts and must pass both health checks before the old one stops; if it never
becomes healthy, the deployment circuit breaker rolls back to revision 2 on its own.

## 4. Find the new address (the public IP changes on every deployment)

```powershell
$Task = aws ecs list-tasks @P --cluster $Cluster --service-name $Service --query "taskArns[0]" --output text
$Eni = aws ecs describe-tasks @P --cluster $Cluster --tasks $Task `
  --query "tasks[0].attachments[0].details[?name=='networkInterfaceId'].value" --output text
$Ip = aws ec2 describe-network-interfaces @P --network-interface-ids $Eni `
  --query "NetworkInterfaces[0].Association.PublicIp" --output text
"UI: http://${Ip}:8501"
aws ecs describe-tasks @P --cluster $Cluster --tasks $Task `
  --query "tasks[0].containers[].[name,lastStatus,healthStatus,image]" --output table
```

Expected: `api` on `...-api:0.2.0` and `ui` on `...-ui:0.2.1`, both `RUNNING` / `HEALTHY`; the
sidebar of the UI ends with "UI version 0.2.1".

**Rollback:** `aws ecs update-service @P --cluster $Cluster --service $Service --task-definition akinsds1126-aero-rag-api:2`,
then step 4 again for the new IP.

# Release 0.2.0: phases 6 + 7a (task definition revision 2, done 16 November 2026)

Kept as the record of how revision 2 was deployed.

What changes: the API image goes from 0.1.0 to 0.2.0 (evaluation v2 fixes and Langfuse tracing);
a second container, the Streamlit chat UI 0.2.0, joins the same task; the task gets the three
Langfuse keys from the same secret and `LANGFUSE_CAPTURE_TEXT=true`; the public port moves from
8000 (API) to 8501 (UI).

**`LANGFUSE_CAPTURE_TEXT=true`** sends each question and final answer to Langfuse (redacted for
secrets and host names). It is on only because the manuals are synthetic; it must be `false` for
real, proprietary manuals. JSON has no comments, so the task definition carries this warning as a
Docker label on the API container (`aero-rag.note.LANGFUSE_CAPTURE_TEXT`).

State before this release: service `akinsds1126-aero-rag-api-service-jk8marb6` on
`akinsds1126-aero-rag-api:1`, security group `sg-02fa227fd6bdc5931` (ingress: 8000 from
0.0.0.0/0 and ::/0). Uses the variables set at the top of this page.

## 1. ECR repository for the UI (same settings as the API repository)

```powershell
aws ecr create-repository @P --repository-name akinsds1126-aero-rag-ui `
  --image-tag-mutability IMMUTABLE --image-scanning-configuration scanOnPush=true `
  --encryption-configuration encryptionType=AES256 `
  --tags Key=project,Value=aero-rag Key=env,Value=dev Key=owner,Value=akinlolu
```

## 2. Push both images (the ones built and tested locally by `scripts/compose_up.ps1`)

The ECR repositories use immutable tags, so a pushed `0.2.0` can never be replaced. Push only after
the last local rebuild (`.\scripts\compose_up.ps1`) of the version you want to demo, and check
both containers are up on `0.2.0` with `docker compose ps`.

```powershell
aws ecr get-login-password @P | docker login --username AWS --password-stdin $Registry
foreach ($name in "akinsds1126-aero-rag-api", "akinsds1126-aero-rag-ui") {
  docker tag "${name}:0.2.0" "$Registry/${name}:0.2.0"
  docker push "$Registry/${name}:0.2.0"
}
```

## 3. Register task definition revision 2

`deploy/ecs/task-definition-rev2.json` has `SECRET_ARN_HERE` for the three Langfuse keys. It is the
same secret as the other five: `arn:aws:secretsmanager:eu-west-2:666127452756:secret:akinsds1126-aero-rag/app-uUSUsb`.
The execution role already reads `akinsds1126-aero-rag/app-*`, so no IAM change is needed.

```powershell
$SecretArn = "arn:aws:secretsmanager:eu-west-2:666127452756:secret:akinsds1126-aero-rag/app-uUSUsb"
(Get-Content deploy\ecs\task-definition-rev2.json -Raw).Replace("SECRET_ARN_HERE", $SecretArn) |
  Set-Content -Encoding ascii "$env:TEMP\taskdef-rev2.json"
aws ecs register-task-definition @P --cli-input-json "file://$env:TEMP\taskdef-rev2.json" `
  --query "taskDefinition.[family,revision]" --output text
```

## 4. Open 8501 (the UI) before deploying

```powershell
aws ec2 authorize-security-group-ingress @P --group-id $Sg `
  --ip-permissions "IpProtocol=tcp,FromPort=8501,ToPort=8501,IpRanges=[{CidrIp=0.0.0.0/0,Description=StreamlitUI}]" `
  --tag-specifications "ResourceType=security-group-rule,Tags=[{Key=project,Value=aero-rag},{Key=env,Value=dev},{Key=owner,Value=akinlolu}]"
```

For a tighter demo, replace `0.0.0.0/0` with your own address (`<your-ip>/32`).

## 5. Deploy revision 2 and wait

```powershell
aws ecs update-service @P --cluster $Cluster --service $Service `
  --task-definition akinsds1126-aero-rag-api:2 --query "service.deployments[0].rolloutState"
aws ecs wait services-stable @P --cluster $Cluster --services $Service
```

The service keeps 100% healthy during the rollout (the new task starts before the old one stops),
and the deployment circuit breaker rolls back automatically if the new task never becomes healthy.

## 6. Find the public address and check

```powershell
$Task = aws ecs list-tasks @P --cluster $Cluster --service-name $Service --query "taskArns[0]" --output text
$Eni = aws ecs describe-tasks @P --cluster $Cluster --tasks $Task `
  --query "tasks[0].attachments[0].details[?name=='networkInterfaceId'].value" --output text
$Ip = aws ec2 describe-network-interfaces @P --network-interface-ids $Eni `
  --query "NetworkInterfaces[0].Association.PublicIp" --output text
"UI: http://${Ip}:8501"
aws ecs describe-tasks @P --cluster $Cluster --tasks $Task `
  --query "tasks[0].containers[].[name,lastStatus,healthStatus,image]" --output table
aws logs tail /ecs/akinsds1126-aero-rag-api @P --since 10m | Select-String "tracing|startup complete"
```

Expected: both containers `RUNNING` / `HEALTHY`, a log line `tracing enabled`, the UI answering at
`http://<ip>:8501`, and the query visible in Langfuse.

## 7. Close 8000 to the internet (after step 6 works)

The UI reaches the API inside the task at `localhost:8000`, which a security group does not
affect, so the API no longer needs to be public.

```powershell
aws ec2 revoke-security-group-ingress @P --group-id $Sg `
  --security-group-rule-ids sgr-044359ed9341f95e3 sgr-0371be041e2fc8ff8
Test-NetConnection $Ip -Port 8000 -InformationLevel Quiet   # expect False
Test-NetConnection $Ip -Port 8501 -InformationLevel Quiet   # expect True
```

## 8. Tag the security group (it was created by the console without our tags)

```powershell
aws ec2 create-tags @P --resources $Sg --tags Key=project,Value=aero-rag Key=env,Value=dev Key=owner,Value=akinlolu
```

## Notes

- **Size and cost.** The task grows from 0.5 vCPU / 1 GB to 1 vCPU / 2 GB because Streamlit runs
  next to the API: about $0.057 an hour in eu-west-2 (was about $0.028).
- **The security group belongs to the console's CloudFormation stack.** Editing its rules is
  normal but shows as drift on that stack; deleting the service in the console deletes the stack
  and the group with it.
- **The public IP changes on every deployment** (no load balancer). Re-run step 6 after each.
- **Ingestion** (one-off task): revision 2 would also start the UI. Keep using revision 1 for it
  (`--task-definition akinsds1126-aero-rag-api:1`); the ingestion code has not changed.
- **Rollback.** `aws ecs update-service @P --cluster $Cluster --service $Service --task-definition akinsds1126-aero-rag-api:1`,
  and if step 7 was done, re-open 8000 with `authorize-security-group-ingress` as in step 4.
