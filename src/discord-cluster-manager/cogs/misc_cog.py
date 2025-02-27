import os
from typing import TYPE_CHECKING

import discord
import psycopg2
from discord import app_commands
from discord.ext import commands
from env import DATABASE_URL
from utils import send_discord_message, setup_logging

if TYPE_CHECKING:
    from ..bot import ClusterBot

logger = setup_logging()


class BotManagerCog(commands.Cog):
    def __init__(self, bot: "ClusterBot"):
        self.bot = bot

    @app_commands.command(name="ping")
    async def ping(self, interaction: discord.Interaction):
        """Simple ping command to check if the bot is responsive"""
        await send_discord_message(interaction, "pong")

    @app_commands.command(name="get-names")
    async def get_id_to_name_mapping(self, interaction: discord.Interaction):
        """Get a mapping of user IDs to their names"""
        await interaction.response.defer()
        try:
            with self.bot.leaderboard_db as db:
                db.cursor.execute("""
                    SELECT DISTINCT user_id
                    FROM leaderboard.submission
                """)
                user_ids = [row[0] for row in db.cursor.fetchall()]

            user_mapping = {}
            for user_id in user_ids:
                try:
                    discord_id = int(user_id)
                    user = await self.bot.fetch_user(discord_id)
                    user_mapping[user_id] = user.global_name or user.name
                except (ValueError, discord.NotFound, discord.HTTPException) as e:
                    logger.error(f"Error fetching user {user_id}: {str(e)}")
                    user_mapping[user_id] = "Unknown User"
            import json
            import tempfile

            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w")
            json.dump(user_mapping, temp_file, indent=2)
            temp_file.close()

            await interaction.followup.send(
                content="Here's the mapping of user IDs to names:",
                file=discord.File(temp_file.name, filename="user_mapping.json"),
            )

            import os

            os.unlink(temp_file.name)

        except Exception as e:
            error_message = f"Error generating user mapping: {str(e)}"
            logger.error(error_message, exc_info=True)
            await send_discord_message(interaction, error_message)

    @app_commands.command(name="update-db-with-names")
    @app_commands.describe(attachment="The JSON file containing user ID to name mapping")
    async def update_db_with_names(
        self, interaction: discord.Interaction, attachment: discord.Attachment
    ):
        """Update the database with user names from a JSON file"""
        await interaction.response.defer()

        try:
            if not attachment.filename.endswith(".json"):
                await send_discord_message(
                    interaction, "Please attach a JSON file with .json extension."
                )
                return

            json_content = await attachment.read()
            import json

            user_mapping = json.loads(json_content)

            updated_count = 0
            with self.bot.leaderboard_db as db:
                for user_id, user_name in user_mapping.items():
                    try:
                        db.cursor.execute(
                            """
                            UPDATE leaderboard.submission
                            SET user_name = %s
                            WHERE user_id = %s
                        """,
                            (user_name, user_id),
                        )
                        updated_count += db.cursor.rowcount
                    except Exception as e:
                        logger.error(f"Error updating user {user_id}: {str(e)}")

                db.connection.commit()

            await send_discord_message(
                interaction,
                f"Successfully updated {updated_count} submission records with user names.",
            )

        except json.JSONDecodeError:
            await send_discord_message(interaction, "Invalid JSON format in the attached file.")
        except Exception as e:
            error_message = f"Error updating database with user names: {str(e)}"
            logger.error(error_message, exc_info=True)
            await send_discord_message(interaction, error_message)

    @app_commands.command(name="verifydb")
    async def verify_db(self, interaction: discord.Interaction):
        """Command to verify database connectivity"""
        if not DATABASE_URL:
            message = "DATABASE_URL not set."
            logger.error(message)
            await send_discord_message(interaction, message)
            return

        try:
            with psycopg2.connect(DATABASE_URL, sslmode="require") as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT RANDOM()")
                    result = cursor.fetchone()
                    if result:
                        random_value = result[0]
                        await send_discord_message(
                            interaction, f"Your lucky number is {random_value}."
                        )
                    else:
                        await send_discord_message(interaction, "No result returned.")
        except Exception as e:
            message = "Error interacting with the database"
            logger.error(f"{message}: {str(e)}", exc_info=True)
            await send_discord_message(interaction, f"{message}.")

    @app_commands.command(name="get-api-url")
    async def get_api_url(self, interaction: discord.Interaction):
        if not self.bot.debug_mode:
            await send_discord_message(
                interaction, "Submission through the API are coming soon! Stay tuned... 👀"
            )
            return

        if not os.environ.get("HEROKU_APP_DEFAULT_DOMAIN_NAME"):
            await send_discord_message(
                interaction,
                "No `HEROKU_APP_DEFAULT_DOMAIN_NAME` present,"
                " are you sure you aren't running locally?",
            )
        else:
            await send_discord_message(
                interaction,
                f"API URL: https://{os.environ['HEROKU_APP_DEFAULT_DOMAIN_NAME']}",
            )
