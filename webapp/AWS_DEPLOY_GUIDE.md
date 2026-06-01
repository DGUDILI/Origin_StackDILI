# DGUDILI AWS 배포 가이드

> ECS Fargate(백엔드) + S3/CloudFront(프론트엔드) 기반 서버리스 아키텍처

---

## 전체 아키텍처

```
사용자 브라우저
    │
    ├─ 정적 파일 ──► CloudFront ──► S3 (React 빌드 결과)
    │
    └─ API 요청 ──► CloudFront (또는 직접) ──► ALB ──► ECS Fargate (FastAPI)
                                                              │
                                                    GraphMACCSEncoder
                                                    (CPU-only 컨테이너)
                                                              │
                                                    S3 (배치 결과 CSV 저장, 선택)
```

**AWS 서비스 목록**

| 서비스 | 역할 |
|--------|------|
| ECR | Docker 이미지 레지스트리 |
| ECS Fargate | 백엔드 컨테이너 실행 (서버 관리 불필요) |
| ALB | ECS 앞단 로드 밸런서 (헬스 체크, HTTPS) |
| ACM | SSL/TLS 인증서 |
| S3 | 프론트엔드 정적 호스팅 + 배치 결과 저장 |
| CloudFront | 프론트엔드 CDN + 캐시 (전 세계 빠른 응답) |
| IAM | 최소 권한 역할 |

---

## 사전 준비

```bash
# AWS CLI 설치 및 인증
aws configure
# AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, 리전(ap-northeast-2) 입력

# 리전 변수 설정
export AWS_REGION=ap-northeast-2
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
```

---

## Step 1: ECR 레지스트리 생성 및 이미지 푸시

### 1-1. ECR 레포지토리 생성

```bash
aws ecr create-repository \
  --repository-name dgudili-backend \
  --region $AWS_REGION

# 출력에서 repositoryUri 확인 (이후 단계에서 사용)
# 예: 123456789012.dkr.ecr.ap-northeast-2.amazonaws.com/dgudili-backend
export ECR_URI=$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/dgudili-backend
```

### 1-2. Docker 이미지 빌드 및 푸시

```bash
cd webapp/backend

# ECR 로그인
aws ecr get-login-password --region $AWS_REGION \
  | docker login --username AWS --password-stdin $ECR_URI

# 이미지 빌드 (첫 빌드는 torch 다운로드로 15~20분 소요)
docker build -t dgudili-backend .

# 태그 및 푸시
docker tag dgudili-backend:latest $ECR_URI:latest
docker push $ECR_URI:latest
```

> **주의**: `pretrained_graph_encoder.pt` 가중치 파일이 `weights/` 폴더에 있어야 빌드 성공.

---

## Step 2: IAM 역할 생성 (ECS Task Role)

ECS 컨테이너가 S3에 접근하기 위한 최소 권한 역할.

```bash
# 신뢰 정책 파일 생성
cat > ecs-trust-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": { "Service": "ecs-tasks.amazonaws.com" },
    "Action": "sts:AssumeRole"
  }]
}
EOF

# IAM 역할 생성
aws iam create-role \
  --role-name dgudili-ecs-task-role \
  --assume-role-policy-document file://ecs-trust-policy.json

# S3 버킷 접근 정책 (배치 결과 저장용, 선택)
cat > s3-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": ["s3:PutObject", "s3:GetObject"],
    "Resource": "arn:aws:s3:::dgudili-batch-results/*"
  }]
}
EOF

aws iam put-role-policy \
  --role-name dgudili-ecs-task-role \
  --policy-name S3BatchResults \
  --policy-document file://s3-policy.json

# ECS Task Execution Role (이미지 Pull, CloudWatch 로그)
aws iam attach-role-policy \
  --role-name dgudili-ecs-task-role \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
```

---

## Step 3: ECS 클러스터 및 Task Definition

### 3-1. 클러스터 생성

```bash
aws ecs create-cluster \
  --cluster-name dgudili-cluster \
  --capacity-providers FARGATE
```

### 3-2. CloudWatch 로그 그룹 생성

```bash
aws logs create-log-group \
  --log-group-name /ecs/dgudili-backend \
  --region $AWS_REGION
```

### 3-3. Task Definition 등록

```bash
cat > task-definition.json << EOF
{
  "family": "dgudili-backend",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "8192",
  "taskRoleArn": "arn:aws:iam::${AWS_ACCOUNT_ID}:role/dgudili-ecs-task-role",
  "executionRoleArn": "arn:aws:iam::${AWS_ACCOUNT_ID}:role/dgudili-ecs-task-role",
  "containerDefinitions": [{
    "name": "dgudili-backend",
    "image": "${ECR_URI}:latest",
    "portMappings": [{"containerPort": 8000, "protocol": "tcp"}],
    "environment": [
      {"name": "DEVICE",         "value": "cpu"},
      {"name": "DILI_THRESHOLD", "value": "0.45"},
      {"name": "S3_ENABLED",     "value": "true"},
      {"name": "S3_BUCKET_NAME", "value": "dgudili-batch-results"},
      {"name": "AWS_REGION",     "value": "${AWS_REGION}"},
      {"name": "ALLOWED_ORIGINS","value": "https://your-cloudfront-domain.cloudfront.net"},
      {"name": "LOG_LEVEL",      "value": "INFO"}
    ],
    "logConfiguration": {
      "logDriver": "awslogs",
      "options": {
        "awslogs-group":  "/ecs/dgudili-backend",
        "awslogs-region": "${AWS_REGION}",
        "awslogs-stream-prefix": "ecs"
      }
    },
    "healthCheck": {
      "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
      "interval": 30,
      "timeout": 10,
      "retries": 3,
      "startPeriod": 120
    }
  }]
}
EOF

aws ecs register-task-definition \
  --cli-input-json file://task-definition.json
```

> **리소스 권장값**
> - CPU 2 vCPU (2048), 메모리 8 GB (8192): ChemBERTa(~3.5M) + 모델 + RDKit 추론 여유 확보
> - 트래픽 낮으면 CPU 1 vCPU / 메모리 4 GB 도 가능

---

## Step 4: ALB + ECS 서비스

### 4-1. VPC / 서브넷 / 보안 그룹 확인

```bash
# 기본 VPC ID 확인
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" \
  --query 'Vpcs[0].VpcId' --output text)

# 서브넷 ID 목록
SUBNET_IDS=$(aws ec2 describe-subnets \
  --filters "Name=vpc-id,Values=$VPC_ID" \
  --query 'Subnets[].SubnetId' --output text | tr '\t' ',')

# ALB용 보안 그룹 (80, 443 인바운드)
ALB_SG=$(aws ec2 create-security-group \
  --group-name dgudili-alb-sg \
  --description "ALB security group" \
  --vpc-id $VPC_ID \
  --query 'GroupId' --output text)

aws ec2 authorize-security-group-ingress \
  --group-id $ALB_SG --protocol tcp --port 80  --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress \
  --group-id $ALB_SG --protocol tcp --port 443 --cidr 0.0.0.0/0

# ECS용 보안 그룹 (ALB에서만 8000 허용)
ECS_SG=$(aws ec2 create-security-group \
  --group-name dgudili-ecs-sg \
  --description "ECS task security group" \
  --vpc-id $VPC_ID \
  --query 'GroupId' --output text)

aws ec2 authorize-security-group-ingress \
  --group-id $ECS_SG --protocol tcp --port 8000 --source-group $ALB_SG
```

### 4-2. ALB 생성

```bash
ALB_ARN=$(aws elbv2 create-load-balancer \
  --name dgudili-alb \
  --subnets $(echo $SUBNET_IDS | tr ',' ' ') \
  --security-groups $ALB_SG \
  --query 'LoadBalancers[0].LoadBalancerArn' --output text)

# 타깃 그룹 (헬스 체크: /health)
TG_ARN=$(aws elbv2 create-target-group \
  --name dgudili-tg \
  --protocol HTTP --port 8000 \
  --vpc-id $VPC_ID \
  --target-type ip \
  --health-check-path /health \
  --health-check-interval-seconds 30 \
  --healthy-threshold-count 2 \
  --unhealthy-threshold-count 3 \
  --query 'TargetGroups[0].TargetGroupArn' --output text)

# HTTP 리스너 (HTTPS 없이 테스트 시)
aws elbv2 create-listener \
  --load-balancer-arn $ALB_ARN \
  --protocol HTTP --port 80 \
  --default-actions Type=forward,TargetGroupArn=$TG_ARN

# ALB DNS 이름 확인
ALB_DNS=$(aws elbv2 describe-load-balancers \
  --load-balancer-arns $ALB_ARN \
  --query 'LoadBalancers[0].DNSName' --output text)
echo "ALB DNS: $ALB_DNS"
```

### 4-3. ECS 서비스 생성

```bash
aws ecs create-service \
  --cluster dgudili-cluster \
  --service-name dgudili-backend \
  --task-definition dgudili-backend \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={
    subnets=[$SUBNET_IDS],
    securityGroups=[$ECS_SG],
    assignPublicIp=ENABLED
  }" \
  --load-balancers "targetGroupArn=$TG_ARN,containerName=dgudili-backend,containerPort=8000"
```

> 서비스 시작 후 `http://$ALB_DNS/health`에서 `{"status":"ready"}` 확인까지 약 3~5분 소요.

---

## Step 5: HTTPS 설정 (ACM + ALB)

```bash
# ACM 인증서 요청 (소유 도메인 필요)
CERT_ARN=$(aws acm request-certificate \
  --domain-name api.your-domain.com \
  --validation-method DNS \
  --query 'CertificateArn' --output text)

# 콘솔에서 DNS 검증 후:
# HTTPS 리스너 추가
aws elbv2 create-listener \
  --load-balancer-arn $ALB_ARN \
  --protocol HTTPS --port 443 \
  --certificates CertificateArn=$CERT_ARN \
  --default-actions Type=forward,TargetGroupArn=$TG_ARN

# HTTP → HTTPS 리다이렉트
aws elbv2 modify-listener \
  --listener-arn <HTTP_LISTENER_ARN> \
  --default-actions Type=redirect,RedirectConfig="{Protocol=HTTPS,Port=443,StatusCode=HTTP_301}"
```

---

## Step 6: 프론트엔드 — S3 + CloudFront

### 6-1. S3 버킷 생성 및 정적 호스팅

```bash
FRONTEND_BUCKET=dgudili-frontend-$(date +%s)
aws s3 mb s3://$FRONTEND_BUCKET --region $AWS_REGION

# 정적 호스팅 활성화
aws s3 website s3://$FRONTEND_BUCKET \
  --index-document index.html \
  --error-document index.html   # SPA fallback
```

### 6-2. 프론트엔드 빌드 및 업로드

```bash
cd webapp/frontend

# 실제 ALB 도메인(또는 HTTPS 도메인)으로 빌드
VITE_API_BASE_URL=https://api.your-domain.com npm run build

# S3 업로드
aws s3 sync dist/ s3://$FRONTEND_BUCKET \
  --delete \
  --cache-control "max-age=31536000,public,immutable" \
  --exclude "index.html"

# index.html은 캐시 없이
aws s3 cp dist/index.html s3://$FRONTEND_BUCKET/index.html \
  --cache-control "no-cache,no-store,must-revalidate"
```

### 6-3. CloudFront 배포 생성

```bash
aws cloudfront create-distribution \
  --distribution-config '{
    "CallerReference": "dgudili-'$(date +%s)'",
    "Comment": "DGUDILI Frontend",
    "DefaultRootObject": "index.html",
    "Origins": {
      "Quantity": 1,
      "Items": [{
        "Id": "s3-origin",
        "DomainName": "'$FRONTEND_BUCKET'.s3-website.'$AWS_REGION'.amazonaws.com",
        "CustomOriginConfig": {
          "HTTPPort": 80,
          "HTTPSPort": 443,
          "OriginProtocolPolicy": "http-only"
        }
      }]
    },
    "DefaultCacheBehavior": {
      "TargetOriginId": "s3-origin",
      "ViewerProtocolPolicy": "redirect-to-https",
      "CachePolicyId": "658327ea-f89d-4fab-a63d-7e88639e58f6",
      "Compress": true,
      "ForwardedValues": {
        "QueryString": false,
        "Cookies": {"Forward": "none"}
      },
      "MinTTL": 0,
      "TrustedSigners": {"Enabled": false, "Quantity": 0}
    },
    "CustomErrorResponses": {
      "Quantity": 1,
      "Items": [{
        "ErrorCode": 404,
        "ResponsePagePath": "/index.html",
        "ResponseCode": "200",
        "ErrorCachingMinTTL": 300
      }]
    },
    "Enabled": true
  }'

# CloudFront 도메인 확인 (xxxxx.cloudfront.net)
```

### 6-4. ALLOWED_ORIGINS 업데이트

CloudFront 도메인 확인 후 ECS Task Definition의 `ALLOWED_ORIGINS` 환경 변수를 업데이트하고 서비스를 재배포합니다.

```bash
# 새 Task Definition 등록 (ALLOWED_ORIGINS 값만 변경)
# task-definition.json의 ALLOWED_ORIGINS를 실제 CloudFront 도메인으로 수정 후:
aws ecs register-task-definition --cli-input-json file://task-definition.json

# 서비스 업데이트 (무중단 롤링 배포)
aws ecs update-service \
  --cluster dgudili-cluster \
  --service dgudili-backend \
  --task-definition dgudili-backend  # 최신 revision 자동 사용
```

---

## Step 7: 배치 결과 S3 버킷 (선택)

```bash
aws s3 mb s3://dgudili-batch-results --region $AWS_REGION

# 결과 파일 1시간 후 자동 삭제 (비용 절감)
aws s3api put-bucket-lifecycle-configuration \
  --bucket dgudili-batch-results \
  --lifecycle-configuration '{
    "Rules": [{
      "ID": "delete-old-results",
      "Status": "Enabled",
      "Expiration": {"Days": 1},
      "Prefix": "batch-results/"
    }]
  }'
```

---

## Step 8: 배포 업데이트 (이후)

```bash
# 1. 이미지 재빌드 및 푸시
cd webapp/backend
docker build -t dgudili-backend .
docker tag dgudili-backend:latest $ECR_URI:latest
docker push $ECR_URI:latest

# 2. 서비스 강제 재배포 (새 이미지 사용)
aws ecs update-service \
  --cluster dgudili-cluster \
  --service dgudili-backend \
  --force-new-deployment

# 3. 프론트엔드 재빌드 및 S3 동기화
cd webapp/frontend
VITE_API_BASE_URL=https://api.your-domain.com npm run build
aws s3 sync dist/ s3://$FRONTEND_BUCKET --delete
aws cloudfront create-invalidation \
  --distribution-id <DISTRIBUTION_ID> \
  --paths "/*"
```

---

## 비용 추정 (서울 리전, 소규모 트래픽)

| 서비스 | 스펙 | 월 비용 (참고) |
|--------|------|---------------|
| ECS Fargate | 0.5 vCPU / 4GB, 730h | ~$15 |
| ALB | 1 LCU/시간 기준 | ~$20 |
| S3 (프론트) | 수 MB 저장 | < $1 |
| CloudFront | 소량 요청 | < $5 |
| ECR | 1 GB 이미지 | ~$1 |
| **합계** | | **~$40/월** |

> GPU 인스턴스 필요 없음 — ChemBERTa 추론은 CPU에서 분자당 3~10초 수준.

---

## 트러블슈팅

### 백엔드 서비스가 UNHEALTHY

1. ECS 태스크 로그 확인:
   ```bash
   aws logs tail /ecs/dgudili-backend --follow
   ```
2. `STARTUP COMPLETE` 로그 없으면 → 가중치 파일 또는 ChemBERTa 다운로드 실패
3. `startPeriod: 120`초 이내에 헬스 체크 통과해야 함

### CORS 오류

- `ALLOWED_ORIGINS` 환경 변수가 실제 CloudFront 도메인과 일치하는지 확인
- `*` 와일드카드 사용 시 `ALLOW_CREDENTIALS=false` 필수 (이미 기본값)

### 모델 로드 OOM

- Fargate 메모리를 8192 MB 이상으로 증가
- ChemBERTa (~700 MB) + 모델 가중치 + RDKit 처리 여유 필요

### ChemBERTa 다운로드 실패 (인터넷 없는 환경)

- HuggingFace 모델을 빌드 시점에 이미지에 포함:
  ```dockerfile
  RUN python -c "from transformers import AutoTokenizer, AutoModel; \
    AutoTokenizer.from_pretrained('DeepChem/ChemBERTa-77M-MLM'); \
    AutoModel.from_pretrained('DeepChem/ChemBERTa-77M-MLM')"
  ENV TRANSFORMERS_OFFLINE=1
  ```
