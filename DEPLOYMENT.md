# Deploying WHALE-RE to Streamlit Community Cloud

## Prerequisites
- GitHub account
- OpenAI API key
- Streamlit Community Cloud account (free at [share.streamlit.io](https://share.streamlit.io))

## Step 1: Prepare Your Repository

### Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit for Streamlit deployment"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/whale-re.git
git push -u origin main
```

**Note:** The `.gitignore` file is already configured to exclude sensitive files like `.env` and `auth_config.yaml`.

## Step 2: Configure Streamlit Secrets

When deploying to Streamlit Cloud, you'll need to add your secrets in the dashboard.

### In Streamlit Cloud Dashboard:

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click **"New app"**
4. Select your repository, branch (`main`), and main file (`app.py`)
5. Click **"Advanced settings"** before deploying
6. In the **Secrets** section, add this TOML configuration:

```toml
# OpenAI API Configuration
OPENAI_API_KEY = "sk-your-actual-api-key-here"
OPENAI_MODEL = "gpt-4o"

# Authentication Configuration
[credentials.usernames.alice]
name = "Alice"
password = "$2b$12$DJtK4F3kGyVSd0ieDMHmx.C4RCJeN7TaS6aWYutbOQWU75C9mmNMO"
email = "alice@example.com"
role = "analyst"

[credentials.usernames.bob]
name = "Bob Reviewer"
password = "$2b$12$W/GQPLRA/ObQFKp99OeEC.cWfjuvGLPhZWVLLCP66c8Lc9PPjTdNG"
email = "bob@example.com"
role = "reviewer"

[credentials.usernames.manager]
name = "Project Manager"
password = "$2b$12$WNYa77dKgAAGffqacJddxuDn/sPzWZUou4uEz6RDPiqR1Y.pCygua"
email = "manager@example.com"
role = "manager"

# Cookie configuration for authentication
[cookie]
name = "whale_auth"
key = "CHANGE-THIS-TO-A-RANDOM-SECRET-KEY"
expiry_days = 0.000694
```

### Important Security Notes:
- Replace `OPENAI_API_KEY` with your actual OpenAI API key
- **Change the `cookie.key`** to a random secret (use a password generator for 32+ characters)
- To change user passwords, generate new bcrypt hashes using Python:
  ```python
  import bcrypt
  password = "new_password"
  hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
  print(hashed.decode('utf-8'))
  ```

## Step 3: Deploy

1. After adding secrets, click **"Deploy"**
2. Wait for the build to complete (3-5 minutes)
3. Your app will be live at `https://your-app-name.streamlit.app`

## Step 4: Post-Deployment

### Update Users and Passwords
To add/modify users after deployment:
1. Go to your app dashboard on Streamlit Cloud
2. Click **"Settings"** → **"Secrets"**
3. Edit the TOML configuration
4. Save changes (app will automatically restart)

### Monitor Usage
- Check app logs in the Streamlit Cloud dashboard
- Monitor OpenAI API usage in your OpenAI dashboard
- Set up billing alerts for API costs

## Local Development vs Cloud

The app automatically detects its environment:
- **Local:** Uses `auth_config.yaml` and `.env` files
- **Cloud:** Uses Streamlit secrets (configured in dashboard)

This means you can develop locally without changing code for deployment.

## Troubleshooting

### App won't start
- Check the logs in Streamlit Cloud dashboard
- Verify all required secrets are configured
- Ensure `requirements.txt` includes all dependencies

### Authentication not working
- Verify the TOML format in secrets is correct (indentation matters)
- Check that bcrypt password hashes are properly formatted
- Ensure cookie key is set and unique

### OpenAI API errors
- Verify your API key is valid and has credits
- Check API usage limits in OpenAI dashboard
- Ensure `OPENAI_MODEL` matches an available model

## Sample Login Credentials

Default test users (change passwords in production):
- **alice** / alice123 (Analyst role)
- **bob** / bob12345 (Reviewer role)
- **manager** / manage! (Manager role)

## Cost Considerations

- **Streamlit Cloud:** Free for public repos
- **OpenAI API:** Pay per token used
  - Set up usage limits in OpenAI dashboard
  - Monitor costs regularly
  - Consider caching responses for repeated queries

## Support

For issues:
1. Check [Streamlit Community Forum](https://discuss.streamlit.io)
2. Review [Streamlit Deployment Docs](https://docs.streamlit.io/streamlit-community-cloud)
3. Check OpenAI API status at [status.openai.com](https://status.openai.com)
