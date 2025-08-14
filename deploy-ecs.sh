#!/bin/bash

# Canvas Reel Optimizer - ECS Deployment Script
# This script builds and deploys the application to AWS ECS Fargate

set -e

# Load environment variables from .env file if it exists
if [ -f ".env" ]; then
    echo "Loading environment variables from .env file..."
    export $(grep -v '^#' .env | xargs)
fi

# Configuration (with fallbacks to defaults)
AWS_REGION="${AWS_REGION:-us-east-1}"
AWS_PROFILE="${AWS_PROFILE:-default}"
AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID}"
ECR_REPOSITORY="${ECR_REPOSITORY:-canvas-reel-optimizer}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
STACK_NAME="${STACK_NAME:-canvas-reel-optimizer-ecs}"

# Validate required environment variables
if [ -z "$AWS_ACCOUNT_ID" ]; then
    echo "❌ Error: AWS_ACCOUNT_ID is required. Please set it in .env file or environment."
    echo "Example: AWS_ACCOUNT_ID=123456789012"
    exit 1
fi

if [ -z "$AWS_PROFILE" ] || [ "$AWS_PROFILE" = "default" ]; then
    echo "⚠️  Warning: Using default AWS profile. Set AWS_PROFILE in .env for specific profile."
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Canvas Reel Optimizer - ECS Deployment${NC}"
echo "=================================================="

# Check if AWS CLI is configured for the specific profile
if ! AWS_PROFILE=$AWS_PROFILE aws sts get-caller-identity > /dev/null 2>&1; then
    echo -e "${RED}❌ AWS CLI not configured for profile: $AWS_PROFILE${NC}"
    exit 1
fi

echo -e "${GREEN}✅ AWS CLI configured${NC}"
echo "Profile: $AWS_PROFILE"
echo "Account ID: $AWS_ACCOUNT_ID"
echo "Region: $AWS_REGION"

# Step 1: Create ECR repository if it doesn't exist
echo -e "\n${YELLOW}📦 Step 1: Setting up ECR repository...${NC}"
if ! AWS_PROFILE=$AWS_PROFILE aws ecr describe-repositories --repository-names $ECR_REPOSITORY --region $AWS_REGION > /dev/null 2>&1; then
    echo "Creating ECR repository..."
    AWS_PROFILE=$AWS_PROFILE aws ecr create-repository --repository-name $ECR_REPOSITORY --region $AWS_REGION
    echo -e "${GREEN}✅ ECR repository created${NC}"
else
    echo -e "${GREEN}✅ ECR repository already exists${NC}"
fi

# Step 2: Build and push Docker image
echo -e "\n${YELLOW}🐳 Step 2: Building and pushing Docker image...${NC}"

# Get ECR login token
AWS_PROFILE=$AWS_PROFILE aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Build Docker image
echo "Building Docker image..."
docker build -t $ECR_REPOSITORY:$IMAGE_TAG .

# Tag image for ECR
docker tag $ECR_REPOSITORY:$IMAGE_TAG $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY:$IMAGE_TAG

# Push image to ECR
echo "Pushing image to ECR..."
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY:$IMAGE_TAG

echo -e "${GREEN}✅ Docker image pushed to ECR${NC}"

# Step 3: Update CloudFormation template with actual values
echo -e "\n${YELLOW}⚙️  Step 3: Preparing CloudFormation template...${NC}"

echo -e "${GREEN}✅ CloudFormation template prepared${NC}"

# Step 4: Get VPC and Subnet information
echo -e "\n${YELLOW}🌐 Step 4: Getting VPC and subnet information...${NC}"

# Get default VPC
DEFAULT_VPC=$(AWS_PROFILE=$AWS_PROFILE aws ec2 describe-vpcs --filters "Name=is-default,Values=true" --query "Vpcs[0].VpcId" --output text --region $AWS_REGION)

if [ "$DEFAULT_VPC" = "None" ] || [ -z "$DEFAULT_VPC" ]; then
    echo -e "${RED}❌ No default VPC found. Please specify VPC and subnet IDs manually.${NC}"
    echo "Edit the deploy command below with your VPC and subnet IDs."
    exit 1
fi

# Get subnets in the default VPC
SUBNETS=$(AWS_PROFILE=$AWS_PROFILE aws ec2 describe-subnets --filters "Name=vpc-id,Values=$DEFAULT_VPC" --query "Subnets[*].SubnetId" --output text --region $AWS_REGION)
SUBNET_ARRAY=($SUBNETS)

if [ ${#SUBNET_ARRAY[@]} -lt 2 ]; then
    echo -e "${RED}❌ Need at least 2 subnets for ALB. Found: ${#SUBNET_ARRAY[@]}${NC}"
    exit 1
fi

# Use all subnets for both public and private (simplified deployment)
# Use public subnets for both ALB and ECS (simplified approach)
PUBLIC_SUBNETS="${SUBNET_ARRAY[@]}"
PRIVATE_SUBNETS="${SUBNET_ARRAY[@]}"  # Using same subnets for now

echo "VPC ID: $DEFAULT_VPC"
echo "Public Subnets: $PUBLIC_SUBNETS"
echo "ECS Subnets: $PUBLIC_SUBNETS"

# Step 5: Deploy CloudFormation stack
echo -e "\n${YELLOW}☁️  Step 5: Deploying CloudFormation stack...${NC}"

AWS_PROFILE=$AWS_PROFILE aws cloudformation deploy \
    --template-file cloudformation-ecs-simple.yaml \
    --stack-name $STACK_NAME \
    --parameter-overrides \
        VpcId=$DEFAULT_VPC \
        SubnetIds="${PUBLIC_SUBNETS// /,}" \
        PublicSubnetIds="${PUBLIC_SUBNETS// /,}" \
        ImageUri=$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY:$IMAGE_TAG \
    --capabilities CAPABILITY_NAMED_IAM \
    --region $AWS_REGION

echo -e "${GREEN}✅ CloudFormation stack deployed${NC}"

# Step 6: Get the application URL
echo -e "\n${YELLOW}🌍 Step 6: Getting application URL...${NC}"

ALB_DNS=$(AWS_PROFILE=$AWS_PROFILE aws cloudformation describe-stacks \
    --stack-name $STACK_NAME \
    --query "Stacks[0].Outputs[?OutputKey=='LoadBalancerDNS'].OutputValue" \
    --output text \
    --region $AWS_REGION)

if [ -n "$ALB_DNS" ] && [ "$ALB_DNS" != "None" ]; then
    echo -e "\n${GREEN}🎉 Deployment completed successfully!${NC}"
    echo "=================================================="
    echo -e "${BLUE}Application URL: http://$ALB_DNS${NC}"
    echo "=================================================="
    echo ""
    echo "📝 Next steps:"
    echo "1. Wait 2-3 minutes for the service to fully start"
    echo "2. The application is accessible directly via the ALB (no CloudFront)"
    echo "3. All Streamlit features including WebSockets work perfectly"
    echo "4. Ensure your AWS account has access to Bedrock models"
    echo "5. Configure your application settings in the UI"
    echo ""
    echo "🔧 To update the application:"
    echo "1. Make your changes"
    echo "2. Run this script again"
    echo ""
    echo "🗑️  To delete the deployment:"
    echo "AWS_PROFILE=$AWS_PROFILE aws cloudformation delete-stack --stack-name $STACK_NAME --region $AWS_REGION"
else
    echo -e "${RED}❌ Could not retrieve application URL${NC}"
fi

# Cleanup

echo -e "\n${GREEN}✅ Deployment script completed${NC}"
