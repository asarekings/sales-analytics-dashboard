import subprocess

# User config
GITHUB_USER = "asarekings"
REPO_NAME = "sales-analytics-dashboard"
REMOTE_URL = f"https://github.com/{GITHUB_USER}/{REPO_NAME}.git"

def run(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print("Error running:", cmd)
        exit(result.returncode)

# 1. Initialize git repo if not already
run("git init")

# 2. Add all files
run("git add .")

# 3. Commit changes
run('git commit -m "Initial project structure for docs and deployment"')

# 4. Add remote if not already present
remotes = subprocess.getoutput("git remote")
if "origin" not in remotes:
    run(f"git remote add origin {REMOTE_URL}")

# 5. Push to GitHub (main branch)
run("git branch -M main")
run("git push -u origin main")

print("✅ Project pushed to GitHub!")