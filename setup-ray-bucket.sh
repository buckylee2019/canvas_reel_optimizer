#!/bin/bash

# Setup S3 bucket for Ray model in us-west-2
# This script creates the required S3 bucket for Luma Ray model

set -e

# Configuration
AWS_PROFILE="buckylee+test_224425919845"
AWS_ACCOUNT_ID="224425919845"
BUCKET_NAME="canvas-reel-optimizer-west2-${AWS_ACCOUNT_ID}"
REGION="us-west-2"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🪣 Setting up S3 bucket for Ray model in us-west-2${NC}"
echo "=================================================="

# Check if AWS CLI is configured for the specific profile
if ! AWS_PROFILE=$AWS_PROFILE aws sts get-caller-identity > /dev/null 2>&1; then
    echo -e "${RED}❌ AWS CLI not configured for profile: $AWS_PROFILE${NC}"
    exit 1
fi

echo -e "${GREEN}✅ AWS CLI configured${NC}"
echo "Profile: $AWS_PROFILE"
echo "Account ID: $AWS_ACCOUNT_ID"
echo "Region: $REGION"
echo "Bucket: $BUCKET_NAME"

# Check if bucket already exists
echo -e "\n${YELLOW}🔍 Checking if bucket exists...${NC}"
if AWS_PROFILE=$AWS_PROFILE aws s3api head-bucket --bucket $BUCKET_NAME --region $REGION 2>/dev/null; then
    echo -e "${GREEN}✅ Bucket already exists: $BUCKET_NAME${NC}"
else
    echo -e "${YELLOW}📦 Creating S3 bucket in us-west-2...${NC}"
    
    # Create bucket with location constraint for us-west-2
    AWS_PROFILE=$AWS_PROFILE aws s3api create-bucket \
        --bucket $BUCKET_NAME \
        --region $REGION \
        --create-bucket-configuration LocationConstraint=$REGION
    
    echo -e "${GREEN}✅ S3 bucket created: $BUCKET_NAME${NC}"
    
    # Set bucket versioning (optional but recommended)
    echo -e "${YELLOW}🔄 Enabling versioning...${NC}"
    AWS_PROFILE=$AWS_PROFILE aws s3api put-bucket-versioning \
        --bucket $BUCKET_NAME \
        --versioning-configuration Status=Enabled \
        --region $REGION
    
    # Set bucket lifecycle policy to clean up old versions (optional)
    echo -e "${YELLOW}🗑️ Setting lifecycle policy...${NC}"
    cat > /tmp/lifecycle-policy.json << EOF
{
    "Rules": [
        {
            "ID": "DeleteOldVersions",
            "Status": "Enabled",
            "NoncurrentVersionExpiration": {
                "NoncurrentDays": 30
            },
            "AbortIncompleteMultipartUpload": {
                "DaysAfterInitiation": 7
            }
        }
    ]
}
EOF
    
    AWS_PROFILE=$AWS_PROFILE aws s3api put-bucket-lifecycle-configuration \
        --bucket $BUCKET_NAME \
        --lifecycle-configuration file:///tmp/lifecycle-policy.json \
        --region $REGION
    
    # Clean up temp file
    rm /tmp/lifecycle-policy.json
    
    echo -e "${GREEN}✅ Lifecycle policy configured${NC}"
fi

# Test bucket access
echo -e "\n${YELLOW}🧪 Testing bucket access...${NC}"
if AWS_PROFILE=$AWS_PROFILE aws s3 ls s3://$BUCKET_NAME --region $REGION > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Bucket access confirmed${NC}"
else
    echo -e "${RED}❌ Cannot access bucket${NC}"
    exit 1
fi

echo -e "\n${GREEN}🎉 Ray model S3 bucket setup completed!${NC}"
echo "=================================================="
echo -e "${BLUE}Bucket Name: s3://$BUCKET_NAME${NC}"
echo -e "${BLUE}Region: $REGION${NC}"
echo -e "${BLUE}Purpose: Luma Ray model video output${NC}"
echo ""
echo "📝 Next steps:"
echo "1. The Ray model will now use this bucket automatically"
echo "2. Videos generated with Ray model will be stored in us-west-2"
echo "3. Nova Reel models will continue using the us-east-1 bucket"
echo ""
echo "🔧 To delete this bucket later:"
echo "AWS_PROFILE=$AWS_PROFILE aws s3 rb s3://$BUCKET_NAME --force --region $REGION"
