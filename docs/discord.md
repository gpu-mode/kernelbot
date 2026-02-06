# Discord Bot Setup

To run and develop the bot locally, you need to create a Discord bot application and add it to your own "staging" server.

## Create a Bot Application

Follow the Discord.js guides:
1. [Creating your bot](https://discordjs.guide/preparations/setting-up-a-bot-application.html#creating-your-bot)
2. [Adding your bot to servers](https://discordjs.guide/preparations/adding-your-bot-to-servers.html#bot-invite-links)

## Required Permissions

The bot needs the `Message Content Intent` and `Server Members Intent` permissions turned on.

<details>
<summary>Click for visual</summary>
<img width="1440" alt="DCS_bot_perms" src="https://github.com/user-attachments/assets/31ee441a-f8a9-4a2f-89d1-fda171947bfd" />
</details>

## Required Scopes

The bot needs `applications.commands` and `bot` scopes.

<details>
<summary>Click for visual</summary>
<img width="1440" alt="Screenshot 2024-11-24 at 12 34 09 PM" src="https://github.com/user-attachments/assets/31302214-1d5a-416a-b7b4-93a44442be51">
</details>

## Generate Invite Link

Generate an invite link for the bot and enter it into any browser.

<details>
<summary>Click for visual</summary>
<img width="1440" alt="Screenshot 2024-11-24 at 12 44 08 PM" src="https://github.com/user-attachments/assets/54c34b6b-c944-4ce7-96dd-e40cfe79ffb3">
</details>

> [!NOTE]
> Bot permissions involving threads/mentions/messages should suffice, but you can give it `Administrator` since it's just a test bot in your own testing Discord server.

## Environment Variables

Add these to your `.env` file:

```bash
DISCORD_TOKEN=                      # Bot token (production)
DISCORD_DEBUG_TOKEN=                # Bot token (local development)
DISCORD_CLUSTER_STAGING_ID=         # Server ID (production)
DISCORD_DEBUG_CLUSTER_STAGING_ID=   # Server ID (local development)
```

> [!NOTE]
> For local development, you can set the DEBUG variants to the same values as the production ones.

### Finding Your Bot Token

Found in your bot's page within the [Discord Developer Portal](https://discord.com/developers/applications):

<details>
<summary>Click for visual</summary>
<img width="1440" alt="Screenshot 2024-11-24 at 11 01 19 AM" src="https://github.com/user-attachments/assets/b98bb4e0-8489-4441-83fb-256053aac34d">
</details>

### Finding Your Server ID

Right-click your staging Discord server and select `Copy Server ID`:

<details>
<summary>Click for visual</summary>
<img width="1440" alt="Screenshot 2024-11-24 at 10 58 27 AM" src="https://github.com/user-attachments/assets/0754438c-59ef-4db2-bcaa-c96106c16756">
</details>
