# ⚠️ SECURITY NOTICE - Environment Variables

## Important: Never Commit Real Credentials

**DO NOT** commit actual credentials to the repository. The `.env.example` file contains placeholders only.

## Setting Up Your Environment

1. **Copy the example file:**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` with your actual credentials:**
   - Replace all `your_*` placeholders with real values
   - The `.env` file is already in `.gitignore` and will NOT be committed

3. **For production deployment:**
   - Set environment variables directly in your deployment platform
   - Use secrets management (AWS Secrets Manager, etc.)
   - Never hardcode credentials in code

## Current Credentials Status

If you see actual credentials in any file:
- ✅ `.env.example` - Contains placeholders only (safe to commit)
- ❌ `.env` - Should contain real credentials (already in `.gitignore`)
- ❌ Any other files - Should NOT contain real credentials

## If Credentials Are Exposed

If you accidentally commit credentials:

1. **Immediately rotate/revoke the exposed credentials**
2. **Remove from git history:**
   ```bash
   git filter-branch --force --index-filter \
     "git rm --cached --ignore-unmatch .env" \
     --prune-empty --tag-name-filter cat -- --all
   ```
3. **Force push** (coordinate with team first)
4. **Update `.gitignore`** to ensure it's excluded

## Best Practices

- ✅ Use `.env.example` for documentation
- ✅ Keep `.env` in `.gitignore`
- ✅ Use environment variables in production
- ✅ Rotate credentials regularly
- ❌ Never commit `.env` files
- ❌ Never hardcode credentials in code
- ❌ Never share credentials in chat/email
