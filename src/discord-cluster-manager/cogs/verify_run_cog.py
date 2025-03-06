import asyncio
import datetime
import re
import subprocess
import tempfile
import uuid
from pathlib import Path
from unittest.mock import AsyncMock

import discord
import env
from cogs import admin_cog
from cogs.github_cog import GitHubCog
from cogs.leaderboard_cog import LeaderboardSubmitCog
from cogs.modal_cog import ModalCog
from consts import CUDA_FLAGS, GPU_TO_SM, SubmissionMode
from discord import app_commands
from discord.app_commands import Choice
from discord.ext import commands
from task import make_task
from utils import RunItem, SubmissionItem, send_discord_message, setup_logging, with_error_handling

logger = setup_logging()


def create_mock_attachment(file_name: str, content: str):
    "Create an AsyncMock to simulate discord.Attachment"

    mock_attachment = AsyncMock(spec=discord.Attachment)
    mock_attachment.filename = file_name
    mock_attachment.content_type = "text/plain"
    mock_attachment.read = AsyncMock(return_value=content.encode("utf-8"))
    return mock_attachment


class VerifyRunCog(commands.Cog):
    """
    A Discord cog for verifying the success of training runs.

    A cog that verifies training runs across different platforms and GPU types.
    Runs test scripts on GitHub (NVIDIA and AMD) and Modal to validate that the
    runs complete successfully. Each run is monitored for expected output
    messages.
    """

    def __init__(self, bot):
        self.bot = bot

    async def verify_github_run(
        self,
        github_cog: GitHubCog,
        choice: app_commands.Choice,
        interaction: discord.Interaction,
        lang: str,
    ) -> bool:
        # async_submit_cog_job
        github_command = github_cog.submit_leaderboard
        if lang == "py":
            sub_code = create_mock_attachment(
                "submission.py", Path("examples/softmax_py/submission.py").read_text()
            )
            task = make_task("examples/softmax_py")
        else:
            sub_code = create_mock_attachment(
                "test.cu", Path("examples/identity_cuda/submission.cu").read_text()
            )
            task = make_task("examples/identity_cuda")

        github_thread, _ = await github_command(
            interaction, sub_code, choice, task=task, mode=SubmissionMode.TEST
        )

        message_contents = [msg.content async for msg in github_thread.history(limit=None)]

        required_patterns = ["Running on GitHub...", "Passed 5/5 tests"]

        all_patterns_found = all(
            any(re.search(pattern, content, re.DOTALL) is not None for content in message_contents)
            for pattern in required_patterns
        )

        if all_patterns_found:
            await send_discord_message(
                interaction,
                f"✅ GitHub run ({choice.name}) for {lang} completed successfully - "
                "all expected messages found!",
            )
            return True
        else:
            missing_patterns = [
                pattern
                for pattern in required_patterns
                if not any(re.search(pattern, content, re.DOTALL) for content in message_contents)
            ]
            await send_discord_message(
                interaction,
                f"❌ GitHub run ({choice.name}) for {lang} verification failed. "
                + "Missing expected messages:\n"
                + "\n".join(f"- {pattern}" for pattern in missing_patterns),
            )
            return False

    async def verify_modal_run(
        self, modal_cog: ModalCog, interaction: discord.Interaction, lang: str
    ) -> bool:
        t4 = app_commands.Choice(name="T4", value="t4")
        modal_command = modal_cog.submit_leaderboard

        if lang == "py":
            sub_code = create_mock_attachment(
                "submission.py", Path("examples/identity_py/submission.py").read_text()
            )
            task = make_task("examples/identity_py")
        else:
            sub_code = create_mock_attachment(
                "test.cu", Path("examples/identity_cuda/submission.cu").read_text()
            )
            task = make_task("examples/identity_cuda")

        modal_thread, _ = await modal_command(
            interaction, sub_code, t4, task=task, mode=SubmissionMode.TEST
        )

        message_contents = [msg.content async for msg in modal_thread.history(limit=None)]

        required_patterns = ["Running on Modal...", "Passed 5/5 tests"]

        all_patterns_found = all(
            any(re.search(pattern, content, re.DOTALL) is not None for content in message_contents)
            for pattern in required_patterns
        )

        if all_patterns_found:
            await send_discord_message(
                interaction,
                f"✅ Modal run for {lang} completed successfully - all expected messages found!",
            )
            return True
        else:
            missing_patterns = [
                pattern
                for pattern in required_patterns
                if not any(re.search(pattern, content, re.DOTALL) for content in message_contents)
            ]
            await send_discord_message(
                interaction,
                f"❌ Modal run verification for {lang} failed. Missing expected messages:\n"
                + "\n".join(f"- {pattern}" for pattern in missing_patterns),
            )
            return False

    @app_commands.command(name="verify-task")
    @app_commands.autocomplete(task=admin_cog.leaderboard_dir_autocomplete)
    @app_commands.choices(
        mode=[
            Choice(name=SubmissionMode.TEST.name, value=SubmissionMode.TEST.value),
            Choice(name=SubmissionMode.BENCHMARK.name, value=SubmissionMode.BENCHMARK.value),
            Choice(name=SubmissionMode.LEADERBOARD.name, value=SubmissionMode.LEADERBOARD.value),
            Choice(name="All", value="all"),
        ]
    )
    @with_error_handling
    async def verify_task(
        self, interaction: discord.Interaction, task: str, mode: Choice[str] = None
    ):
        directory = Path(env.PROBLEM_DEV_DIR) / task
        if not directory.resolve().is_relative_to(Path.cwd() / env.PROBLEM_DEV_DIR):
            await send_discord_message(interaction, f"Invalid path {directory.resolve()}")
            return
        try:
            task = make_task(directory)
        except Exception as E:
            logger.exception("Could not make task", exc_info=E)
            await send_discord_message(interaction, f"Invalid task {directory}")
            return
        await send_discord_message(interaction, f"Testing {directory}")

        modes = []
        if mode is None:
            modes = [SubmissionMode.LEADERBOARD]
        elif mode.value == "all":
            modes = [SubmissionMode.TEST, SubmissionMode.BENCHMARK, SubmissionMode.LEADERBOARD]
        else:
            modes = [SubmissionMode(mode.value)]

        lb_name = f"test.{uuid.uuid4().hex}"
        # create the dummy leaderboard
        with self.bot.leaderboard_db as db:  # type: LeaderboardDB
            db.create_leaderboard(
                {
                    "name": lb_name,
                    "deadline": datetime.datetime.now() + datetime.timedelta(days=1),
                    "task": task,
                    "gpu_types": "T4",
                    "creator_id": interaction.user.id,
                }
            )
        try:
            # make submissions
            submissions = []
            reports = []
            for sub in directory.glob("solutions/*/*"):
                for mode in modes:
                    submissions.append(
                        self.verify_submission(interaction, lb_name, sub, mode, reports)
                    )
            await asyncio.gather(*submissions)
        except Exception as E:
            logger.exception("Error in LB test", exc_info=E)
            await send_discord_message(interaction, str(E), ephemeral=True)
            return
        finally:
            with self.bot.leaderboard_db as db:
                db.delete_leaderboard(lb_name, force=True)

        report = f"Report:```{'\n'.join(sorted(reports))}```"

        await send_discord_message(interaction, report)

    async def verify_submission(  # noqa: C901
        self,
        interaction: discord.Interaction,
        lb_name: str,
        sub: Path,
        mode: SubmissionMode,
        reports: list[str],
    ):
        lb_cog = LeaderboardSubmitCog(self.bot)
        script = create_mock_attachment(sub.name, sub.read_text())
        sub_id = await lb_cog.on_submit_hook(interaction, lb_name, script, mode, cmd_gpus=["T4"])

        run_id = f"{sub.parent.name}/{sub.name}:"

        if sub_id == -1:
            reports.append(f"❌ {run_id:20} submitting failed")
            return

        report_success = True

        # verify results
        with self.bot.leaderboard_db as db:
            sub_data: SubmissionItem = db.get_submission_by_id(sub_id)
        if sub_data is None:
            reports.append(f"❌ {run_id:20} cannot find in db")
            return

        if sub_data["done"] is not True:
            reports.append(f"❌ {run_id:20} is unfinished")
            return

        if sub.parent.name == "correct":
            run: RunItem
            for run in sub_data["runs"]:
                if run["passed"] is not True:
                    reports.append(f"❌ {run_id:20} run {run['mode']} failed")
                    report_success = False
        elif sub.parent.name == "wrong":
            for run in sub_data["runs"]:
                if run["passed"] is True:
                    reports.append(f"❌ {run_id:20} run {run['mode']} passed")
                    report_success = False

        if report_success:
            reports.append(f"✅ {run_id:20} {mode.name} behaved as expected")

    async def generate_ptx_code(self, source_code: str, gpu_type: str, include_sass: bool = False) -> tuple[bool, str]:
        """
        Generate PTX code for a CUDA submission.
        
        Args:
            source_code (str): The CUDA source code
            gpu_type (str): The GPU architecture to target
            include_sass (bool): Whether to include SASS assembly code
            
        Returns:
            tuple[bool, str]: Success status and the PTX output or error message
        """
        # Get the SM architecture code for the specified GPU type
        arch = GPU_TO_SM.get(gpu_type)
        if not arch:
            return False, f"Unknown GPU type: {gpu_type}. Available types: {', '.join(GPU_TO_SM.keys())}"
            
        try:
            # Create a temporary directory for the compilation
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                source_file = temp_path / "submission.cu"
                
                # Write the source code to a file
                source_file.write_text(source_code)
                
                # Prepare the compilation command with PTX output flag
                ptx_flags = CUDA_FLAGS.copy() + ["-ptx"]
                
                # Add sass generation flag if requested
                if include_sass:
                    ptx_flags.append("-Xptxas=-v")  # Verbose output with sass info
                    
                arch_flag = f"-gencode=arch=compute_{arch},code=compute_{arch}"
                
                command = ["nvcc"] + ptx_flags + [str(source_file), arch_flag, "-o", str(temp_path / "output.ptx")]
                
                # Check if nvcc is available
                nvcc_check = subprocess.run(["which", "nvcc"], capture_output=True, text=True)
                if nvcc_check.returncode != 0:
                    return False, "NVCC (CUDA compiler) not found. Is CUDA installed?"
                
                # Run the compilation
                process = subprocess.run(command, capture_output=True, text=True)
                
                # Prepare the output with both stderr (for SASS if requested) and the PTX file
                result = ""
                
                # Include compilation output which contains SASS information
                if include_sass and process.stderr:
                    result += "SASS Assembly Information:\n"
                    result += "-" * 40 + "\n"
                    result += process.stderr + "\n"
                    result += "-" * 40 + "\n\n"
                
                if process.returncode != 0:
                    # Compilation failed
                    return False, f"PTX generation failed:\n{process.stderr}"
                
                # Read the PTX file
                ptx_file = temp_path / "output.ptx"
                if ptx_file.exists():
                    result += "PTX Code:\n"
                    result += "-" * 40 + "\n"
                    result += ptx_file.read_text()
                    return True, result
                else:
                    return False, "PTX file was not generated"
        except Exception as e:
            return False, f"Error generating PTX: {str(e)}"

    @app_commands.command(name="ptx")
    @app_commands.describe(
        submission="The CUDA submission file (.cu extension)",
        gpu_type="The GPU architecture to target",
        include_sass="Whether to include SASS/assembly output",
        as_file="Return the PTX code as a downloadable file instead of text messages"
    )
    @app_commands.choices(
        gpu_type=[
            Choice(name=gpu, value=gpu) for gpu in GPU_TO_SM.keys()
        ]
    )
    @with_error_handling
    async def ptx_command(self, interaction: discord.Interaction, 
                          submission: discord.Attachment, 
                          gpu_type: Choice[str] = None,
                          include_sass: bool = False,
                          as_file: bool = False):
        """
        Generate PTX code from a CUDA submission.
        
        Parameters
        ------------
        submission: File
            The CUDA submission file (.cu extension)
        gpu_type: Choice[str]
            The GPU architecture to target
        include_sass: bool
            Whether to include SASS assembly code in the output
        as_file: bool
            Return the PTX code as a downloadable file instead of text messages
        """
        if not interaction.response.is_done():
            await interaction.response.defer()
            
        # Validate the file extension
        if not submission.filename.endswith('.cu'):
            await send_discord_message(interaction, "❌ Only .cu file extensions are supported for PTX generation")
            return
            
        # Set default GPU type to T4 if not specified
        target_gpu = gpu_type.value if gpu_type else "T4"
        
        try:
            # Read the submission file
            content = await submission.read()
            source_code = content.decode('utf-8')
            
            # Create a thread for the PTX generation
            thread_name = f"PTX Generation - {submission.filename} - {target_gpu}"
            if include_sass:
                thread_name += " with SASS"
                
            thread = await interaction.channel.create_thread(
                name=thread_name,
                type=discord.ChannelType.public_thread,
            )
            
            await thread.send(f"Generating PTX code for {submission.filename} targeting {target_gpu}..." + 
                              (" (including SASS output)" if include_sass else ""))
            
            # Generate the PTX code
            success, result = await self.generate_ptx_code(source_code, target_gpu, include_sass)
            
            if success:
                if as_file:
                    # Create a temporary file containing the PTX output
                    with tempfile.NamedTemporaryFile('w', suffix='.ptx', delete=False) as temp_file:
                        temp_file.write(result)
                        temp_file_path = temp_file.name
                    
                    # Get the base filename without extension
                    base_filename = Path(submission.filename).stem
                    output_filename = f"{base_filename}_{target_gpu}.ptx"
                    
                    # Send the file
                    await thread.send(
                        f"PTX code for {submission.filename} targeting {target_gpu}:",
                        file=discord.File(temp_file_path, filename=output_filename)
                    )
                    
                    # Remove the temporary file
                    Path(temp_file_path).unlink(missing_ok=True)
                else:
                    # Split the PTX code into chunks if it's too long for Discord
                    max_msg_length = 1900  # Slightly less than 2000 to account for markdown
                    chunks = [result[i:i+max_msg_length] for i in range(0, len(result), max_msg_length)]
                    
                    for i, chunk in enumerate(chunks):
                        await thread.send(f"```{chunk}```")
                
                # Send a summary message
                await thread.send(f"✅ PTX code generation complete for {target_gpu} GPU" + 
                                 (" with SASS assembly" if include_sass else ""))
            else:
                # Send the error message
                await thread.send(f"❌ Failed to generate PTX code: {result}")
            
            # Notify user in the original channel
            await send_discord_message(interaction, f"PTX generation for {submission.filename} is complete. Check the thread for results.")
            
        except Exception as e:
            logger.error(f"Error generating PTX: {e}", exc_info=True)
            await send_discord_message(interaction, f"❌ Error generating PTX: {str(e)}")

    @app_commands.command(name="verifyruns")
    async def verify_runs(self, interaction: discord.Interaction):
        """Verify runs on Modal, GitHub Nvidia, and GitHub AMD."""

        try:
            if not interaction.response.is_done():
                await interaction.response.defer()

            modal_cog = self.bot.get_cog("ModalCog")
            github_cog = self.bot.get_cog("GitHubCog")

            if not all([modal_cog, github_cog]):
                await send_discord_message(interaction, "❌ Required cogs not found!")
                return

            nvidia = app_commands.Choice(name="NVIDIA", value="nvidia")
            amd = app_commands.Choice(name="AMD", value="amd")

            results = await asyncio.gather(
                self.verify_github_run(github_cog, nvidia, interaction, "py"),
                self.verify_github_run(github_cog, nvidia, interaction, "cu"),
                self.verify_modal_run(modal_cog, interaction, "py"),
                self.verify_github_run(github_cog, amd, interaction, "py"),
                self.verify_modal_run(modal_cog, interaction, "cu"),
            )

            if all(results):
                await send_discord_message(interaction, "✅ All runs completed successfully!")
            else:
                await send_discord_message(
                    interaction,
                    "❌ Some runs failed! Consult messages above for details.",
                )

        except Exception as e:
            logger.error(f"Error starting verification runs: {e}", exc_info=True)
            await send_discord_message(
                interaction, f"❌ Problem performing verification runs: {str(e)}"
            )
