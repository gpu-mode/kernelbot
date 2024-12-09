import discord
from discord import app_commands
from discord.ext import commands
import modal
import time
from utils import setup_logging
from consts import ModalGPU
from leaderboard_eval import py_eval, cu_eval

logger = setup_logging()


class ModalCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

        self.run_modal = bot.run_group.command(
            name="modal", description="Run a script using Modal"
        )(self.run_modal)

    @app_commands.describe(
        script="The Python script file to run", gpu_type="Choose the GPU type for Modal"
    )
    @app_commands.choices(
        gpu_type=[
            app_commands.Choice(name=gpu.value, value=gpu.value) for gpu in ModalGPU
        ]
    )
    async def run_modal(
        self,
        interaction: discord.Interaction,
        script: discord.Attachment,
        gpu_type: app_commands.Choice[str],
        use_followup: bool = False,
        reference_script: discord.Attachment = None,
        reference_code: str = None,
    ) -> discord.Thread:
        if not script.filename.endswith(".py") and not script.filename.endswith(".cu"):
            await interaction.response.send_message(
                "Please provide a Python (.py) or CUDA (.cu) file"
            )
            return None

        thread = await self.bot.create_thread(interaction, gpu_type.name, "Modal Job")
        queue_start_time = time.perf_counter()
        message = f"Created thread {thread.mention} for your Modal job"

        if use_followup:
            await interaction.followup.send(message)
        else:
            await interaction.response.send_message(message)

        await thread.send(f"**Processing `{script.filename}` with {gpu_type.name}...**")

        try:
            if reference_script is not None or reference_code is not None:
                reference_content = (
                    reference_code
                    if reference_code is not None
                    else (await reference_script.read()).decode("utf-8")
                )
                eval_code = py_eval if script.filename.endswith(".py") else cu_eval

            script_content = (await script.read()).decode("utf-8")
            status_msg = await thread.send(
                "**Running on Modal...**\n> ⏳ Waiting for available GPU..."
            )

            result, execution_time_ms = await self.trigger_modal_run(
                script_content,
                script.filename,
                gpu_type,
                reference_content,
                eval_code,
            )

            # Update status message to show completion
            await status_msg.edit(
                content="**Running on Modal...**\n> ✅ Job completed!"
            )

            queue_end_time = time.perf_counter()
            queue_time_ms = (queue_end_time - queue_start_time) * 1000

            # Send metrics and results
            await thread.send(f"\n**Script size:** {len(script_content)} bytes")
            await thread.send(f"**Queue time:** {queue_time_ms:.3f} ms")
            await thread.send(f"**Execution time:** {execution_time_ms:.3f} ms\n")
            await thread.send(f"**Modal execution result:**\n```\n{result}\n```")

        except Exception as e:
            logger.error(f"Error processing request: {str(e)}", exc_info=True)
            # Update status message to show error
            await status_msg.edit(content="**Running on Modal...**\n> ❌ Job failed!")
            await thread.send(f"**Error:** {str(e)}")

        finally:
            return thread

    async def trigger_modal_run(
        self,
        script_content: str,
        filename: str,
        gpu_type: str,
        eval_content=None,
        reference_content=None,
    ) -> tuple[str, float]:
        logger.info("Attempting to trigger Modal run")

        from modal_runner import modal_app

        try:
            print(f"Running {filename} with Modal")
            with modal.enable_output():
                with modal_app.run():
                    if filename.endswith(".py"):
                        if gpu_type == "t4":
                            from modal_runner import (
                                run_pytorch_script_t4 as run_pytorch_script,
                            )
                        elif gpu_type == "l4":
                            from modal_runner import (
                                run_pytorch_script_l4 as run_pytorch_script,
                            )
                        elif gpu_type == "a100":
                            from modal_runner import (
                                run_pytorch_script_a100_40gb as run_pytorch_script,
                            )
                        elif gpu_type == "a100-80gb":
                            from modal_runner import (
                                run_pytorch_script_a100_80gb as run_pytorch_script,
                            )
                        elif gpu_type == "h100":
                            from modal_runner import (
                                run_pytorch_script_h100 as run_pytorch_script,
                            )
                        else:
                            raise Exception(
                                f"{gpu_type} is not supported or is not in Modal."
                            )

                        result, execution_time_ms = run_pytorch_script.remote(
                            script_content,
                            gpu_type,
                            eval_content,
                            reference_content,
                        )
                    elif filename.endswith(".cu"):
                        from modal_runner import run_cuda_script

                        result, execution_time_ms = run_cuda_script.remote(
                            script_content, gpu_type
                        )

            return result, execution_time_ms

        except Exception as e:
            logger.error(f"Error in trigger_modal_run: {str(e)}", exc_info=True)
            return f"Error: {str(e)}", 0

