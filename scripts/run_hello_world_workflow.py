import os

from dotenv import load_dotenv
from github import Github
from github.GithubException import GithubException

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = os.getenv("GITHUB_REPO")

if not GITHUB_TOKEN:
    print("Environment variable GITHUB_TOKEN is not defined.")
    exit(1)

if not GITHUB_REPO:
    print("Environment variable GITHUB_REPO is not defined.")
    exit(1)

try:
    gh = Github(GITHUB_TOKEN)
    repo = gh.get_repo(GITHUB_REPO)

    workflows = repo.get_workflows()
    for w in workflows:
        print(f"Found workflow: {w.name}")

    workflow = repo.get_workflow("hello_world_workflow.yml")

    if not workflow:
        raise ValueError("Could not find hello world workflow.")

    workflow.create_dispatch('main')
    print("Successfully triggered hello world workflow")

except GithubException as e:
    print(f"GitHub API error: {e.status} - {e.data.get('message', 'Unknown error')}")
except Exception as e:
    print(f"Error: {str(e)}")
finally:
    if "gh" in locals():
        gh.close()
