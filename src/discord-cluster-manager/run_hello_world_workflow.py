from consts import GITHUB_REPO, GITHUB_TOKEN
from github import Github
from github.GithubException import GithubException
from utils import get_github_branch_name, setup_logging

logger = setup_logging()


def trigger_workflow():
    try:
        gh = Github(GITHUB_TOKEN)
        repo = gh.get_repo(GITHUB_REPO)

        workflows = repo.get_workflows()
        for w in workflows:
            logger.info(f"Found workflow: {w.name}")

        workflow = repo.get_workflow("Hello World Job")

        if not workflow:
            raise ValueError("Could not find hello world workflow")

        branch = get_github_branch_name()
        workflow.create_dispatch(branch)
        logger.info("Successfully triggered hello world workflow")

    except GithubException as e:
        logger.error(f"GitHub API error: {e.status} - {e.data.get('message', 'Unknown error')}")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
    finally:
        if "gh" in locals():
            gh.close()


if __name__ == "__main__":
    trigger_workflow()
