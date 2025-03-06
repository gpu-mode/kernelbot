from enum import Enum
from typing import TYPE_CHECKING, Optional, Tuple, Type
import tempfile
import subprocess
from pathlib import Path

if TYPE_CHECKING:
    from bot import ClusterBot

import discord
from better_profanity import profanity
from consts import CUDA_FLAGS, GPU_TO_SM, SubmissionMode
from discord import app_commands
from discord.app_commands import Choice
from discord.ext import commands
from report import generate_report
from run_eval import FullResult
from task import LeaderboardTask
from utils import build_task_config, send_discord_message, setup_logging, with_error_handling

logger = setup_logging()


class ProgressReporter:
    def __init__(self, status_msg: discord.Message, header: str):
        self.header = header
        self.lines = []
        self.status = status_msg

    @staticmethod
    async def make_reporter(thread: discord.Thread, content: str):
        status_msg = await thread.send(f"**{content}**\n")
        return ProgressReporter(status_msg, content)

    async def push(self, content: str):
        self.lines.append(f"> {content}")
        await self._update_message()

    async def update(self, new_content: str):
        self.lines[-1] = f"> {new_content}"
        await self._update_message()

    async def update_header(self, new_header):
        self.header = new_header
        await self._update_message()

    async def _update_message(self):
        message = str.join("\n", [f"**{self.header}**"] + self.lines)
        await self.status.edit(content=message, suppress=True)


class SubmitCog(commands.Cog):
    """
    Base class for code submission / run schedular cogs.

    Derived classes need to implement a `_get_arch(self, gpu_type: app_commands.Choice[str])`
    method to translate the selected GPU to an architecture argument for Cuda,
    and a
    ```
    run_submission(self, config: dict, gpu_type: GPUType,
        status: ProgressReporter) -> FullResult
    ```
    coroutine, which handles the actual submission.

    This base class will register a `run` subcommand with the runner's name, which can be used
    to run a single (non-leaderboard) script.
    """

    def __init__(self, bot, name: str, gpus: Type[Enum]):
        self.bot: ClusterBot = bot
        self.name = name

        choices = [app_commands.Choice(name=c.name, value=c.value) for c in gpus]

        run_fn = self.run_script

        # note: these helpers want to set custom attributes on the function, but `method`
        # does not allow setting any attributes, so we define this wrapper
        async def run(
            interaction: discord.Interaction,
            script: discord.Attachment,
            gpu_type: app_commands.Choice[str],
        ):
            return await run_fn(interaction, script, gpu_type)

        run = app_commands.choices(gpu_type=choices)(run)
        run = app_commands.describe(
            script="The Python/CUDA script file to run",
            gpu_type=f"Choose the GPU type for {name}",
        )(run)

        # For now, direct (non-leaderboard) submissions are debug-only.
        if self.bot.debug_mode:
            self.run_script = bot.run_group.command(
                name=self.name.lower(), description=f"Run a script using {self.name}"
            )(run)

    async def submit_leaderboard(
        self,
        interaction: discord.Interaction,
        script: discord.Attachment,
        gpu_type: app_commands.Choice[str],
        task: LeaderboardTask,
        mode: SubmissionMode,
    ) -> Tuple[Optional[discord.Thread], Optional[FullResult]]:
        """
        Function invoked by `leaderboard_cog` to handle a leaderboard run.
        """
        thread, result = await self._handle_submission(
            interaction,
            gpu_type,
            script=script,
            task=task,
            mode=mode,
        )

        return thread, result

    @with_error_handling
    async def run_script(
        self,
        interaction: discord.Interaction,
        script: discord.Attachment,
        gpu_type: app_commands.Choice[str],
    ):
        """
        Function invoked by the `run` command to run a single script.
        """
        await self._handle_submission(
            interaction, gpu_type, script=script, task=None, mode=SubmissionMode.SCRIPT
        )

    async def _handle_submission(
        self,
        interaction: discord.Interaction,
        gpu_type: app_commands.Choice[str],
        script: discord.Attachment,
        task: Optional[LeaderboardTask],
        mode: SubmissionMode,
    ) -> Tuple[Optional[discord.Thread], Optional[FullResult]]:
        """
        Generic function to handle code submissions.
        Args:
            interaction: Interaction that started this command.
            gpu_type: Which GPU to run on.
            script: File that contains the submitted script.
            task: Task specification, of provided

        Returns:
            if successful, returns the created discord thread, and the result of
            the run.
        """
        thread_name = f"{self.name} - {mode.value.capitalize()} Job"

        script_content = await self._validate_input_file(interaction, script)
        if script_content is None:
            return None, None

        # TODO figure out the correct way to handle messaging here
        thread = await self.bot.create_thread(interaction, gpu_type.name, f"{thread_name}")
        await thread.send(
            f"Starting {mode.value.capitalize()} job on {self.name} for "
            f"`{script.filename}` with {gpu_type.name}..."
        )

        status = await ProgressReporter.make_reporter(thread, f"Running on {self.name}...")

        config = build_task_config(
            task=task, submission_content=script_content, arch=self._get_arch(gpu_type), mode=mode
        )

        logger.info("submitting task %s to runner %s", config, self.name)

        result = await self._run_submission(config, gpu_type, status)
        await status.update_header(f"Running on {self.name}... ✅ success")
        try:
            await generate_report(thread, result, mode=mode)
        except Exception as E:
            logger.error("Error generating report. Result: %s", result, exc_info=E)
            raise

        return thread, result

    async def _validate_input_file(
        self,
        interaction: discord.Interaction,
        script: discord.Attachment,
    ) -> Optional[str]:
        # check file extension
        if not script.filename.endswith((".py", ".cu", ".cuh", ".cpp")):
            await send_discord_message(
                interaction,
                "Please provide a Python (.py) or CUDA (.cu / .cuh / .cpp) file",
                ephemeral=True,
            )
            return None

        if profanity.contains_profanity(script.filename):
            await send_discord_message(
                interaction,
                "Please provide a non rude filename",
                ephemeral=True,
            )
            return None

        #  load and decode
        try:
            return (await script.read()).decode("utf-8")
        except UnicodeError:
            await send_discord_message(
                interaction,
                f"Could not decode your file `{script.filename}`.\nIs it UTF-8?",
                ephemeral=True,
            )
            return None

    async def _run_submission(
        self, config: dict, gpu_type: app_commands.Choice[str], status: ProgressReporter
    ) -> FullResult:
        """
        Run a submission specified by `config`.
        To be implemented in derived classes.
        Args:
            config: the config object containing all necessary runner information.
            gpu_type: Which GPU to run for.
            status: callback object that allows updating the status message in discord

        Returns:
            Result of running `config`.
        """
        raise NotImplementedError()

    def _get_arch(self, gpu_type: app_commands.Choice[str]):
        raise NotImplementedError()
        
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
