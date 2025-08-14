# 🚀 AWS ECS Deployment Guide

This guide walks you through deploying the Canvas Reel Optimizer to AWS ECS Fargate.

## 📋 Prerequisites

### AWS Requirements
- AWS CLI installed and configured
- Docker installed and running
- AWS account with appropriate permissions
- Access to Amazon Bedrock models (Nova Canvas, Nova Reel, Claude Sonnet 4)
- S3 bucket for video storage

### Required AWS Permissions
Your AWS user/role needs permissions for:
- ECS (Elastic Container Service)
- ECR (Elastic Container Registry)
- CloudFormation
- IAM (for creating roles)
- EC2 (for VPC, subnets, security groups)
- Elastic Load Balancing
- CloudWatch Logs
- Bedrock (for AI models)
- S3 (for file storage)

## 🚀 Quick Deployment

### Option 1: Automated Deployment (Recommended)

```bash
# Clone the repository
git clone https://github.com/buckylee2019/canvas_reel_optimizer.git
cd canvas_reel_optimizer

# Run the deployment script
./deploy-ecs.sh
```

The script will:
1. ✅ Create ECR repository
2. ✅ Build and push Docker image
3. ✅ Deploy CloudFormation stack
4. ✅ Set up load balancer and ECS service
5. ✅ Provide application URL

### Option 2: Manual Deployment

#### Step 1: Create ECR Repository
```bash
aws ecr create-repository --repository-name canvas-reel-optimizer --region us-east-1
```

#### Step 2: Build and Push Docker Image
```bash
# Get your AWS account ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

# Build image
docker build -t canvas-reel-optimizer .

# Tag and push
docker tag canvas-reel-optimizer:latest $AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/canvas-reel-optimizer:latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/canvas-reel-optimizer:latest
```

#### Step 3: Deploy CloudFormation Stack
```bash
# Update the CloudFormation template with your values
# Replace YOUR_ACCOUNT_ID, YOUR_REGION, YOUR_BUCKET_NAME in cloudformation-ecs-simple.yaml

# Deploy the stack
aws cloudformation deploy \
    --template-file cloudformation-ecs-simple.yaml \
    --stack-name canvas-reel-optimizer-ecs \
    --parameter-overrides \
        VpcId=vpc-xxxxxxxxx \
        SubnetIds=subnet-xxxxxxxx,subnet-yyyyyyyy \
        PublicSubnetIds=subnet-aaaaaaaa,subnet-bbbbbbbb \
        ImageUri=$AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/canvas-reel-optimizer:latest \
    --capabilities CAPABILITY_NAMED_IAM \
    --region us-east-1
```

## ⚙️ Configuration

### Environment Variables
The application uses these environment variables:
- `AWS_DEFAULT_REGION`: AWS region (default: us-east-1)
- `STREAMLIT_SERVER_PORT`: Port for Streamlit (default: 8501)
- `STREAMLIT_SERVER_ADDRESS`: Server address (default: 0.0.0.0)
- `STREAMLIT_SERVER_HEADLESS`: Run in headless mode (default: true)

### S3 Bucket Configuration
1. Create an S3 bucket for video storage
2. Update the bucket name in `config.py`
3. Update the CloudFormation template with your bucket name

### Bedrock Model Access
Ensure your AWS account has access to:
- `amazon.nova-canvas-v1:0`
- `amazon.nova-reel-v1:0`
- `amazon.nova-reel-v1:1`
- `us.anthropic.claude-sonnet-4-20250514-v1:0`
- `amazon.nova-pro-v1:0`
- `amazon.nova-lite-v1:0`

## 🏗️ Architecture

```
Internet → ALB → ECS Fargate → Bedrock Models
                     ↓
                 S3 Bucket
```

### Components Created:
- **ECS Cluster**: Fargate cluster for container orchestration
- **ECS Service**: Manages container instances
- **Application Load Balancer**: Distributes traffic
- **Target Group**: Health checks and routing
- **Security Groups**: Network security
- **IAM Roles**: Task execution and application permissions
- **CloudWatch Logs**: Application logging

## 📊 Monitoring

### CloudWatch Logs
- Log Group: `/ecs/canvas-reel-optimizer`
- Stream Prefix: `ecs`

### Health Checks
- **ALB Health Check**: `/_stcore/health`
- **Container Health Check**: Built-in Streamlit health endpoint

### Metrics
Monitor these CloudWatch metrics:
- ECS Service CPU/Memory utilization
- ALB request count and latency
- Target group healthy/unhealthy hosts

## 🔧 Maintenance

### Updating the Application
```bash
# Make your changes, then redeploy
./deploy-ecs.sh
```

### Scaling
```bash
# Update desired count in ECS service
aws ecs update-service \
    --cluster canvas-reel-optimizer-cluster \
    --service canvas-reel-optimizer-service \
    --desired-count 2
```

### Viewing Logs
```bash
# View recent logs
aws logs tail /ecs/canvas-reel-optimizer --follow
```

## 💰 Cost Optimization

### Fargate Pricing
- **CPU**: $0.04048 per vCPU per hour
- **Memory**: $0.004445 per GB per hour
- **Current config**: ~$30-40/month for 1 task running 24/7

### Cost Reduction Tips
1. **Use Fargate Spot**: 70% cost reduction
2. **Scale to zero**: Stop service when not in use
3. **Right-size resources**: Adjust CPU/memory based on usage
4. **Use CloudWatch alarms**: Auto-scale based on demand

### Fargate Spot Configuration
```yaml
# In CloudFormation template
DefaultCapacityProviderStrategy:
  - CapacityProvider: FARGATE_SPOT
    Weight: 1
```

## 🔒 Security

### Network Security
- ECS tasks in private subnets
- ALB in public subnets
- Security groups restrict access

### IAM Security
- Least privilege IAM roles
- Separate execution and task roles
- Bedrock access limited to required models

### Data Security
- S3 bucket with appropriate permissions
- CloudWatch logs encrypted
- Container images scanned by ECR

## 🚨 Troubleshooting

### Common Issues

#### Service Won't Start
```bash
# Check service events
aws ecs describe-services \
    --cluster canvas-reel-optimizer-cluster \
    --services canvas-reel-optimizer-service
```

#### Health Check Failures
```bash
# Check container logs
aws logs tail /ecs/canvas-reel-optimizer --follow
```

#### Image Pull Errors
```bash
# Verify ECR permissions
aws ecr describe-repositories --repository-names canvas-reel-optimizer
```

### Debug Commands
```bash
# Check task status
aws ecs list-tasks --cluster canvas-reel-optimizer-cluster

# Describe task
aws ecs describe-tasks --cluster canvas-reel-optimizer-cluster --tasks TASK_ARN

# Check ALB targets
aws elbv2 describe-target-health --target-group-arn TARGET_GROUP_ARN
```

## 🗑️ Cleanup

### Delete Everything
```bash
# Delete CloudFormation stack
aws cloudformation delete-stack --stack-name canvas-reel-optimizer-ecs --region us-east-1

# Delete ECR repository
aws ecr delete-repository --repository-name canvas-reel-optimizer --force --region us-east-1
```

## 📞 Support

For deployment issues:
1. Check CloudFormation events in AWS Console
2. Review ECS service events
3. Check CloudWatch logs
4. Verify IAM permissions
5. Ensure Bedrock model access

---

**🎉 Once deployed, your Canvas Reel Optimizer will be available at the ALB DNS name provided by the deployment script!**
