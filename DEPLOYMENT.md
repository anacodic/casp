# Production Deployment Guide

This guide explains how to configure environment variables for production deployment.

## Quick Start

1. **Copy the example environment file:**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` with your actual credentials:**
   - Replace all `your_*` placeholders with real values
   - The `.env` file is in `.gitignore` and will NOT be committed

3. **Check your environment configuration:**
   ```bash
   python check_env.py
   ```

4. **Run the application:**
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

**⚠️ Important:** Never commit your `.env` file with real credentials. Only `.env.example` (with placeholders) should be in the repository.

## Required Environment Variables

### AWS Bedrock Configuration (REQUIRED for LLM features)

The system uses AWS Bedrock for LLM orchestration. You must configure AWS credentials.

#### Option 1: Environment Variables (Recommended for Production)

**Basic AWS Credentials:**
```bash
export AWS_ACCESS_KEY_ID=your_aws_access_key_id
export AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
export AWS_REGION=us-east-1  # or your preferred region
export BEDROCK_MODEL_ID=us.anthropic.claude-3-7-sonnet-20250219-v1:0  # Optional, has default
```

#### Option 2: AWS Credentials File

Create `~/.aws/credentials`:
```ini
[default]
aws_access_key_id = your_access_key
aws_secret_access_key = your_secret_key
```

Create `~/.aws/config`:
```ini
[default]
region = us-east-1
```

#### AWS IAM Permissions Required

Your AWS credentials must have permissions to invoke Bedrock models:
- `bedrock:InvokeModel`
- `bedrock:InvokeModelWithResponseStream` (optional)

Example IAM policy:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream"
      ],
      "Resource": "arn:aws:bedrock:*::foundation-model/*"
    }
  ]
}
```

## Optional Environment Variables

### External API Keys

These APIs enhance the system but are not required (fallback methods are used if not set):

```bash
export OPENWEATHERMAP_API_KEY=your_openweathermap_api_key
export NEWSAPI_API_KEY=your_newsapi_api_key
export OPENROUTESERVICE_API_KEY=your_openrouteservice_api_key
```

### Feature Flags

```bash
export FALLBACK_ENABLED=true  # Enable fallback methods when APIs fail (default: true)
export WEB_SEARCH_ENABLED=true  # Enable web search for risk/sourcing agents (default: true)
```

## Docker Deployment

If deploying with Docker, set environment variables in your `docker-compose.yml` or Dockerfile:

```yaml
# docker-compose.yml
services:
  app:
    build: .
    environment:
      - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
      - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
      - AWS_REGION=${AWS_REGION:-us-east-1}
      - BEDROCK_MODEL_ID=${BEDROCK_MODEL_ID}
      - OPENWEATHERMAP_API_KEY=${OPENWEATHERMAP_API_KEY}
      - NEWSAPI_API_KEY=${NEWSAPI_API_KEY}
      - OPENROUTESERVICE_API_KEY=${OPENROUTESERVICE_API_KEY}
```

Or use a `.env` file:
```bash
# .env
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_REGION=us-east-1
```

## Cloud Platform Deployment

### AWS (EC2, ECS, Lambda)

Use IAM roles instead of access keys when possible:
- **EC2/ECS**: Attach IAM role to instance/task
- **Lambda**: Attach IAM role to function
- **EKS**: Use IRSA (IAM Roles for Service Accounts)

Set only `AWS_REGION` environment variable; credentials come from the role.

### Other Cloud Platforms

Set all AWS credentials as environment variables:
- **Heroku**: `heroku config:set AWS_ACCESS_KEY_ID=...`
- **Railway**: Set in dashboard
- **Render**: Set in dashboard
- **Fly.io**: `fly secrets set AWS_ACCESS_KEY_ID=...`

## Troubleshooting

### LLM Not Working

1. **Check AWS credentials:**
   ```bash
   python check_env.py
   ```

2. **Test Bedrock access:**
   ```python
   import boto3
   client = boto3.client("bedrock-runtime", region_name="us-east-1")
   # This should not raise an error
   ```

3. **Verify IAM permissions:**
   - Check that your IAM user/role has `bedrock:InvokeModel` permission
   - Verify Bedrock is available in your region

4. **Check region:**
   - Ensure `AWS_REGION` matches a region where Bedrock is available
   - Default regions: `us-east-1`, `us-west-2`, `eu-west-1`, `ap-southeast-1`

### Common Errors

- **"NoCredentialsError"**: AWS credentials not found
  - Solution: Set `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` or configure `~/.aws/credentials`

- **"AccessDeniedException"**: Missing IAM permissions
  - Solution: Add `bedrock:InvokeModel` permission to your IAM user/role

- **"ValidationException"**: Invalid model ID
  - Solution: Check `BEDROCK_MODEL_ID` is correct (default should work)

- **"Region not available"**: Bedrock not available in region
  - Solution: Change `AWS_REGION` to a supported region (e.g., `us-east-1`)

## Verification

After setting environment variables, verify everything works:

```bash
# Run the environment checker
python check_env.py

# Start the application
uvicorn app:app --host 0.0.0.0 --port 8000

# Test the API
curl -X POST http://localhost:8000/api/extract \
  -H "Content-Type: application/json" \
  -d '{"query": "insulin Mumbai to Delhi"}'
```

If the LLM extraction works, you should see extracted features in the response.
