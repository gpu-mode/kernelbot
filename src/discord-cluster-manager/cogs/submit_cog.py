from enum import Enum
from typing import TYPE_CHECKING, Optional, Tuple, Type

if TYPE_CHECKING:
    from bot import ClusterBot

import discord
from better_profanity import profanity
from consts import SubmissionMode
from discord import app_commands
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
    _run_submission(self, config: dict, gpu_type: GPUType,
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

    async def submit_leaderboard_single(
        self,
        interaction: discord.Interaction,
        leaderboard_name: str,
        script: discord.Attachment,
        task: LeaderboardTask,
        gpu: app_commands.Choice[str],
        mode: SubmissionMode,
        submission_id: int,
    ):
        """
        Handle a single leaderboard submission, i.e., one code submitted with one mode
        to one runner.
        """
        script_content = await self._validate_input_file(interaction, script)
        if script_content is None:
            return None, None

        thread_name = f"{self.name} - {mode} {leaderboard_name}"
        thread = await self.bot.create_thread(interaction, gpu.name, f"{thread_name}")

        result = await self._handle_submission(
            thread,
            gpu,
            script_content=script_content,
            script_name=script.filename,
            task=task,
            mode=mode,
        )

        try:
            if result.success:
                user_id = (
                    interaction.user.global_name
                    if interaction.user.nick is None
                    else interaction.user.nick
                )

                score = None
                if "leaderboard" in result.runs and result.runs["leaderboard"].run.success:
                    score = 0.0
                    num_benchmarks = int(result.runs["leaderboard"].run.result["benchmark-count"])
                    for i in range(num_benchmarks):
                        score += (
                            float(result.runs["leaderboard"].run.result[f"benchmark.{i}.mean"])
                            / 1e9
                        )
                    score /= num_benchmarks

                with self.bot.leaderboard_db as db:
                    for key, value in result.runs.items():
                        db.create_submission_run(
                            submission_id,
                            value.start,
                            value.end,
                            mode=key,
                            runner=gpu.name,
                            score=None if key != "leaderboard" else score,
                            secret=mode == SubmissionMode.PRIVATE,
                            compilation=value.compilation,
                            result=value.run,
                            system=result.system,
                        )

                if score is not None:
                    await thread.send(
                        "## Result:\n"
                        + f"Leaderboard `{leaderboard_name}`:\n"
                        + f"> **{user_id}**'s `{script.filename}` on `{gpu.name}` ran "
                        + f"for `{score:.9f}` seconds!",
                    )
        except Exception as e:
            logger.error("Error in leaderboard submission", exc_info=e)
            await thread.send(
                "## Result:\n"
                + f"Leaderboard submission to '{leaderboard_name}' on {gpu.name} "
                + f"using {self.name} runners failed!\n",
            )

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
        script_content = await self._validate_input_file(interaction, script)
        if script_content is None:
            return

        thread_name = f"{self.name} - Script Job"
        thread = await self.bot.create_thread(interaction, gpu_type.name, f"{thread_name}")

        await self._handle_submission(
            thread,
            gpu_type,
            script_content=script_content,
            script_name=script.filename,
            task=None,
            mode=SubmissionMode.SCRIPT,
        )

    async def _handle_submission(
        self,
        thread: discord.Thread,
        gpu_type: app_commands.Choice[str],
        script_content: str,
        script_name: str,
        task: Optional[LeaderboardTask],
        mode: SubmissionMode,
    ) -> FullResult:
        """
        Generic function to handle code submissions.
        Args:
            thread: Thread in which to report results
            gpu_type: Which GPU to run on.
            script_content: Submitted source code
            script_name: (File) name of the submission
            task: Task specification, of provided

        Returns:
            the result of the run.
        """
        run_msg = f"Running {mode.value.capitalize()} job for `{script_name}` on {self.name} with {gpu_type.name}"
        status = await ProgressReporter.make_reporter(thread, f"{run_msg}...")

        config = build_task_config(
            task=task, submission_content=script_content, arch=self._get_arch(gpu_type), mode=mode
        )

        logger.info("submitting task to runner %s", self.name)

        result = await self._run_submission(config, gpu_type, status)

        if not result.success:
            await status.update_header(f"{run_msg}... ✅ failure")
            await status.push(result.error)
            return result
        else:
            await status.update_header(f"{run_msg}... ✅ success")

        try:
            await generate_report(thread, result.runs, mode=mode)
        except Exception as E:
            logger.error("Error generating report. Result: %s", result, exc_info=E)
            raise

        return result

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
