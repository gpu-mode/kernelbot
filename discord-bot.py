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
from leaderboard import Leaderboard, LeaderboardEntry, HardwareInfo, KernelInfo
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
leaderboard = Leaderboard()

def create_mock_entries():
    # First user - optimized implementations
    fast_user = "cuda_wizard"
    fast_hardware = HardwareInfo(
        name="4090",
        count=1,
        memory=24,
        provider="Local"
    )

    try:
        # GEMM Entry
        leaderboard.add_entry(
            username=fast_user,
            hardware=fast_hardware,
            kernel=KernelInfo(
                name="GEMM FP16",
                language="CUDA",
                problem_size="M=8192, N=8192, K=8192",
                description="Mixed-precision matrix multiplication optimized with tensor cores and shared memory tiling",
                runtime=0.85,
                framework="CUDA",
                version="12.1"
            ),
            metrics={
                "Throughput in TFLOPS": 90.5,
                "Memory Bandwidth in GB/s": 850.2,
                "Score": 0.98
            },
            run_id=f"mock_{len(leaderboard.entries)}"
        )

        # Softmax Entry
        leaderboard.add_entry(
            username=fast_user,
            hardware=fast_hardware,
            kernel=KernelInfo(
                name="Fused Softmax",
                language="CUDA",
                problem_size="batch=128, seq=2048",
                description="Fused softmax with shared memory and warp-level optimizations",
                runtime=0.15,
                framework="CUDA",
                version="12.1"
            ),
            metrics={
                "Throughput in TFLOPS": 45.2,
                "Memory Bandwidth in GB/s": 789.5,
                "Score": 0.95
            },
            run_id=f"mock_{len(leaderboard.entries)}"
        )

        # Conv2D Entry
        leaderboard.add_entry(
            username=fast_user,
            hardware=fast_hardware,
            kernel=KernelInfo(
                name="Conv2D",
                language="CUDA",
                problem_size="batch=32, c_in=256, h=112, w=112, k=512, r=3, s=3",
                description="Im2Col + GEMM based convolution with tensor cores",
                runtime=0.42,
                framework="CUDA",
                version="12.1"
            ),
            metrics={
                "Throughput in TFLOPS": 75.8,
                "Memory Bandwidth in GB/s": 820.4,
                "Score": 0.92
            },
            run_id=f"mock_{len(leaderboard.entries)}"
        )

        # Flash Attention Entry
        leaderboard.add_entry(
            username=fast_user,
            hardware=fast_hardware,
            kernel=KernelInfo(
                name="Flash Attention",
                language="CUDA",
                problem_size="batch=32, seq=2048, heads=32",
                description="Optimized attention implementation with tiling and recomputation",
                runtime=0.28,
                framework="CUDA",
                version="12.1"
            ),
            metrics={
                "Throughput in TFLOPS": 68.4,
                "Memory Bandwidth in GB/s": 795.6,
                "Score": 0.90
            },
            run_id=f"mock_{len(leaderboard.entries)}"
        )

        # Add slower implementations for second user here...
        # (Similar structure but with slower runtimes)

        return "Added a mock entry to the RTX 4090 leaderboard! Type '@Cluster-Bot leaderboards' to view all leaderboards"
    except Exception as e:
        logger.error(f"Error adding mock entries: {str(e)}")
        return f"Error adding mock entries: {str(e)}"

@client.event
async def on_ready():
    logger.info(f'Logged in as {client.user}')
    await client.tree.sync()

@client.tree.command(name="mockadd", description="Add a mock entry to the leaderboard")
async def mockadd(interaction: discord.Interaction):
    logger.info("mockadd command received")
    mock_hardware = HardwareInfo(
        name="4090",
        count=1,
        memory=24,
        provider="Local"
    )
    
    mock_kernel = KernelInfo(
        name="GEMM 1024x1024",
        language="CUDA",
        problem_size="M=1024, N=1024, K=1024",
        runtime=0.42,
        framework="CUDA",
        version="12.1"
    )
    
    mock_metrics = {
        "Score": 0.95,
        "Throughput in TFLOPS": 45.2,
        "Memory Bandwidth in GB/s": 850.2,
        "Occupancy": 0.98,
        "Achieved Memory BW %": 86.4
    }
    
    try:
        leaderboard.add_entry(
            username=interaction.user.name,
            hardware=mock_hardware,
            kernel=mock_kernel,
            metrics=mock_metrics,
            run_id=f"mock_{len(leaderboard.entries)}"
        )
        await interaction.response.send_message("Added mock entry to leaderboard! Use `/leaderboard` to view it")
    except Exception as e:
        logger.error(f"Error adding mock entry: {str(e)}")
        await interaction.response.send_message(f"Error adding mock entry: {str(e)}")

@client.tree.command(name="leaderboards", description="Show available leaderboards")
async def show_leaderboards(interaction: discord.Interaction):
    hardware_list = leaderboard.get_available_hardware()
    if not hardware_list:
        await interaction.response.send_message("No leaderboards available yet!")
        return
        
    output = ["```md", "# Available Leaderboards", ""]
    for idx, hw in enumerate(hardware_list, 1):
        output.append(f"{idx}. {hw} CUDA Kernel Benchmarks")
    output.append("```")
    await interaction.response.send_message("\n".join(output))

@client.tree.command(name="leaderboard_show", description="Show specific leaderboard")
async def show_specific_leaderboard(interaction: discord.Interaction, number: int):
    hardware_list = leaderboard.get_available_hardware()
    if not hardware_list:
        await interaction.response.send_message("No leaderboards available yet!")
        return
        
    if number < 1 or number > len(hardware_list):
        await interaction.response.send_message(f"Please select a number between 1 and {len(hardware_list)}")
        return
        
    hardware_name = hardware_list[number - 1]
    formatted_board = await leaderboard.format_discord_message(hardware_name)
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
            response = create_mock_entries()
            await message.channel.send(response)
            return
        elif 'leaderboards' in content:
            logger.info("leaderboards command received")
            hardware_list = leaderboard.get_available_hardware()
            if not hardware_list:
                await message.channel.send("```md\n# No leaderboards available yet!\n```")
                return
            
            output = [
                "```md",
                "# Available Leaderboards",
                "===============================",
                ""
            ]
            for idx, hw in enumerate(hardware_list, 1):
                output.append(f"{idx}. {hw} CUDA Kernel Benchmarks")
            output.append("")
            output.append("```")
            await message.channel.send("\n".join(output))
            return
        elif 'leaderboard' in content:
            logger.info("leaderboard command received")
            try:
                parts = content.split('leaderboard')
                if len(parts) > 1 and parts[1].strip():
                    number = int(parts[1].strip())
                    hardware_list = leaderboard.get_available_hardware()
                    
                    if not hardware_list:
                        await message.channel.send("No leaderboards available yet!")
                        return
                    
                    if number < 1 or number > len(hardware_list):
                        await message.channel.send(f"Please select a number between 1 and {len(hardware_list)}")
                        return
                    
                    hardware_name = hardware_list[number - 1]
                    formatted_board = await leaderboard.format_discord_message(hardware_name)
                    await message.channel.send(formatted_board)
                else:
                    await message.channel.send("Please specify a leaderboard number. Use '@bot leaderboards' to see available options.")
            except ValueError:
                await message.channel.send("Please provide a valid number after 'leaderboard'")
            return

# Run the bot
if __name__ == "__main__":
    logger.info("Starting bot...")
    if debug_mode := os.getenv("DEBUG"):
        logger.info("Running in debug mode")
    client.run(os.getenv('DISCORD_DEBUG_TOKEN') if debug_mode else os.getenv('DISCORD_TOKEN'))
