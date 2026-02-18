# Environment Variables Reference

Complete list of all environment variables **actually used** in the supply chain system.

## 🔴 Required for Core Functionality

### AWS Credentials
- `AWS_ACCESS_KEY_ID` - AWS access key ID (required)
- `AWS_SECRET_ACCESS_KEY` - AWS secret access key (required)
- `AWS_REGION` - AWS region (default: `us-east-1`)
- `AWS_SESSION_TOKEN` - Optional, only if using temporary credentials

### Bedrock Configuration
- `BEDROCK_MODEL_ID` - Bedrock model ID (default: `us.anthropic.claude-3-7-sonnet-20250219-v1:0`)

## 🟢 Optional - External APIs

### Weather & News APIs
- `OPENWEATHERMAP_API_KEY` - OpenWeatherMap API key
- `NEWSAPI_API_KEY` - NewsAPI key

### Distance/Routing APIs
- `OPENROUTESERVICE_API_KEY` - OpenRouteService API key

## ⚙️ Feature Flags

- `FALLBACK_ENABLED` - Enable fallback methods when APIs fail (default: `true`)
- `WEB_SEARCH_ENABLED` - Enable web search for risk/sourcing agents (default: `true`)

## Usage Examples

### Local Development
```bash
# Load from .env file
source .env
# or
export $(cat .env | xargs)
```

### Production (Docker)
```yaml
environment:
  - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
  - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
  - AWS_REGION=${AWS_REGION}
```

### Production (Cloud Platform)
Set environment variables in your platform's dashboard or use secrets management.

## See Also

- `.env.example` - Template with all variables
- `DEPLOYMENT.md` - Detailed deployment instructions
- `SECURITY_NOTICE.md` - Security best practices
