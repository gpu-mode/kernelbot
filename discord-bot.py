from dotenv import load_dotenv
from github import Github
import os
import time
from datetime import datetime, timezone
import requests
import discord
import asyncio
import logging
import zipfile
import subprocess
from leaderboard import Leaderboard, LeaderboardEntry
from discord import app_commands

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
logger.info("Environment variables loaded")

def get_github_branch_name():
    """
    Runs a git command to determine the remote branch name, to be used in the GitHub Workflow
    """
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', '--symbolic-full-name', '@{u}'],
            capture_output=True,
            text=True,
            check=True
        )
        logging.info(f"Remote branch found: {result.stdout.strip().split('/', 1)[1]}")
        return result.stdout.strip().split('/', 1)[1]
    except subprocess.CalledProcessError:
        logging.warning("Could not determine remote branch, falling back to 'main'")
        return 'main'

# Validate environment variables
if not os.getenv('DISCORD_TOKEN'):
    logger.error("DISCORD_TOKEN not found in environment variables")
    raise ValueError("DISCORD_TOKEN not found")
if not os.getenv('GITHUB_TOKEN'):
    logger.error("GITHUB_TOKEN not found in environment variables")
    raise ValueError("GITHUB_TOKEN not found")
if not os.getenv('GITHUB_REPO'):
    logger.error("GITHUB_REPO not found in environment variables")
    raise ValueError("GITHUB_REPO not found")
if os.getenv("DEBUG") and not os.getenv('DISCORD_DEBUG_TOKEN'):
    logger.error("DISCORD_DEBUG_TOKEN not found in environment variables for debug mode")
    raise ValueError("DISCORD_DEBUG_TOKEN not found")

logger.info(f"Using GitHub repo: {os.getenv('GITHUB_REPO')}")

# Bot setup with minimal intents
class ClusterBot(discord.Client):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self):
        await self.tree.sync()

client = ClusterBot()

# Initialize leaderboard
leaderboard = Leaderboard(title="LLM Performance Rankings")

async def trigger_github_action(script_content):
    """
    Triggers the GitHub action with custom train.py contents
    """
    logger.info("Attempting to trigger GitHub action")
    gh = Github(os.getenv('GITHUB_TOKEN'))
    repo = gh.get_repo(os.getenv('GITHUB_REPO'))
    
    try:
        trigger_time = datetime.now(timezone.utc)
        logger.info(f"Looking for workflow 'train_workflow.yml' in repo {os.getenv('GITHUB_REPO')}")
        
        workflow = repo.get_workflow("train_workflow.yml")
        logger.info("Found workflow, attempting to dispatch")
        
        success = workflow.create_dispatch(get_github_branch_name(), {'script_content': script_content})
        logger.info(f"Workflow dispatch result: {success}")
        
        if success:
            await asyncio.sleep(2)
            runs = list(workflow.get_runs())
            logger.info(f"Found {len(runs)} total runs")
            
            for run in runs:
                logger.info(f"Checking run {run.id} created at {run.created_at}")
                if run.created_at.replace(tzinfo=timezone.utc) > trigger_time:
                    logger.info(f"Found matching run with ID: {run.id}")
                    return run.id
            
            logger.warning("No matching runs found after trigger")
            return None
            
    except Exception as e:
        logger.error(f"Error in trigger_github_action: {str(e)}", exc_info=True)
        return None

async def download_artifact(run_id):
    """
    Downloads the training log artifact from the workflow run
    """
    logger.info(f"Attempting to download artifacts for run {run_id}")
    gh = Github(os.getenv('GITHUB_TOKEN'))
    repo = gh.get_repo(os.getenv('GITHUB_REPO'))
    
    try:
        run = repo.get_workflow_run(run_id)
        artifacts = run.get_artifacts()
        logger.info(f"Found {artifacts.totalCount} artifacts")
        
        for artifact in artifacts:
            logger.info(f"Found artifact: {artifact.name}")
            if artifact.name == 'training-logs':
                url = artifact.archive_download_url
                headers = {'Authorization': f'token {os.getenv("GITHUB_TOKEN")}'}
                response = requests.get(url, headers=headers)
                
                if response.status_code == 200:
                    logger.info("Successfully downloaded artifact")
                    with open('training.log.zip', 'wb') as f:
                        f.write(response.content)
                    
                    with zipfile.ZipFile('training.log.zip') as z:
                        with z.open('training.log') as f:
                            logs = f.read().decode('utf-8')
                    
                    os.remove('training.log.zip')
                    return logs
                else:
                    logger.error(f"Failed to download artifact. Status code: {response.status_code}")
        
        logger.warning("No training-logs artifact found")
        return "No training logs found in artifacts"
    except Exception as e:
        logger.error(f"Error in download_artifact: {str(e)}", exc_info=True)
        return f"Error downloading artifacts: {str(e)}"

async def check_workflow_status(run_id, thread):
    """
    Monitors the GitHub Action workflow status and updates Discord thread
    """
    logger.info(f"Starting to monitor workflow status for run {run_id}")
    gh = Github(os.getenv('GITHUB_TOKEN'))
    repo = gh.get_repo(os.getenv('GITHUB_REPO'))
    
    while True:
        try:
            run = repo.get_workflow_run(run_id)
            logger.info(f"Current status: {run.status}")
            
            if run.status == "completed":
                logger.info("Workflow completed, downloading artifacts")
                logs = await download_artifact(run_id)
                return run.conclusion, logs, run.html_url
            
            await thread.send(f"Workflow still running... Status: {run.status}\nLive view: {run.html_url}")
            await asyncio.sleep(30)
        except Exception as e:
            logger.error(f"Error in check_workflow_status: {str(e)}", exc_info=True)
            return "error", str(e), None

@client.event
async def on_ready():
    logger.info(f'Logged in as {client.user}')
    await client.tree.sync()

@client.tree.command(name="mockadd", description="Add a mock entry to the leaderboard")
async def mockadd(interaction: discord.Interaction):
    logger.info("mockadd command received")
    mock_metrics = {
        "Score": 0.95,
        "MMLU": 0.71,
        "CNN/DailyMail": 0.19,
        "TruthfulQA": 0.76,
        "BBQ": 0.90,
        "GPT4All": 0.85,
        "WizardCoder": 0.82,
        "HumanEval": 0.88
    }
    
    try:
        leaderboard.add_entry(
            username=interaction.user.name,
            metrics=mock_metrics,
            run_id=f"mock_{len(leaderboard.entries)}"
        )
        await interaction.response.send_message("Added mock entry to leaderboard! Use `/leaderboard` to view it")
    except Exception as e:
        logger.error(f"Error adding mock entry: {str(e)}")
        await interaction.response.send_message(f"Error adding mock entry: {str(e)}")

@client.tree.command(name="leaderboard", description="Show the current leaderboard")
async def show_leaderboard(interaction: discord.Interaction):
    logger.info("leaderboard command received")
    formatted_board = await leaderboard.format_discord_message()
    await interaction.response.send_message(formatted_board)

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if client.user in message.mentions:
        content = message.content.lower()
        logger.info(f"Bot mentioned with message: {content}")

        if 'mockadd' in content:
            logger.info("mockadd command received")
            mock_metrics = {
                "Score": 0.95,
                "MMLU": 0.71,
                "CNN/DailyMail": 0.19,
                "TruthfulQA": 0.76,
                "BBQ": 0.90,
                "GPT4All": 0.85,
                "WizardCoder": 0.82,
                "HumanEval": 0.88
            }
            
            try:
                leaderboard.add_entry(
                    username=message.author.name,
                    metrics=mock_metrics,
                    run_id=f"mock_{len(leaderboard.entries)}"
                )
                await message.channel.send("Added mock entry to leaderboard! Type '@Cluster-Bot leaderboard' to view it")
            except Exception as e:
                logger.error(f"Error adding mock entry: {str(e)}")
                await message.channel.send(f"Error adding mock entry: {str(e)}")
            return

        if 'leaderboard' in content:
            logger.info("leaderboard command received")
            formatted_board = await leaderboard.format_discord_message()
            await message.channel.send(formatted_board)
            return

        await message.channel.send("Available commands:\n- @Cluster-Bot mockadd\n- @Cluster-Bot leaderboard")

# Run the bot
if __name__ == "__main__":
    logger.info("Starting bot...")
    if debug_mode := os.getenv("DEBUG"):
        logger.info("Running in debug mode")
    client.run(os.getenv('DISCORD_DEBUG_TOKEN') if debug_mode else os.getenv('DISCORD_TOKEN'))
