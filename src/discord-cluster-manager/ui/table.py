from typing import Any, Dict, List

import discord


class TableView(discord.ui.View):
    def __init__(self, data: List[Dict[str, Any]], items_per_page: int = 10):
        super().__init__()
        self.data = data
        self.current_page = 0
        self.items_per_page = items_per_page
        self.total_pages = max(1, (len(data) + items_per_page - 1) // items_per_page)
        self.update_buttons()

    def update_buttons(self):
        self.previous_page.disabled = self.current_page == 0
        self.next_page.disabled = self.current_page >= self.total_pages - 1
        self.page_counter.label = f"Page {self.current_page + 1}/{self.total_pages}"

    @discord.ui.button(label="◀", style=discord.ButtonStyle.primary)
    async def previous_page(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.current_page = max(0, self.current_page - 1)
        self.update_buttons()
        await interaction.response.edit_message(
            embed=create_table_page(self.data, self.current_page, self.items_per_page), view=self
        )

    @discord.ui.button(label="Page 1/1", style=discord.ButtonStyle.secondary, disabled=True)
    async def page_counter(self, interaction: discord.Interaction, button: discord.ui.Button):
        pass

    @discord.ui.button(label="▶", style=discord.ButtonStyle.primary)
    async def next_page(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.current_page = min(self.total_pages - 1, self.current_page + 1)
        self.update_buttons()
        await interaction.response.edit_message(
            embed=create_table_page(self.data, self.current_page, self.items_per_page), view=self
        )


def create_table_page(data: List[Dict[str, Any]], page: int, items_per_page: int) -> discord.Embed:
    if not data:
        return discord.Embed(description="No data to display")

    headers = list(data[0].keys())

    col_widths = {header: len(header) for header in headers}
    for item in data:
        for header in headers:
            col_widths[header] = max(col_widths[header], len(str(item[header])))

    start_idx = page * items_per_page
    end_idx = min(start_idx + items_per_page, len(data))
    page_data = data[start_idx:end_idx]

    table_rows = []

    header_row = "  ".join(header.ljust(col_widths[header]) for header in headers)
    separator = "  ".join("─" * col_widths[header] for header in headers)

    table_rows.append(header_row)
    table_rows.append(separator)

    for item in page_data:
        row = "  ".join(str(item[header]).ljust(col_widths[header]) for header in headers)
        table_rows.append(row)

    return discord.Embed(description=f"```\n{'\n'.join(table_rows)}\n```")


def create_table(
    title: str, data: List[Dict[str, Any]], items_per_page: int = 10
) -> tuple[discord.Embed, TableView]:
    """
    Create a paginated table for Discord with navigation buttons.

    Args:
        title (str): The title of the table
        data (List[Dict[str, Any]]): List of dictionaries where each dictionary represents a row
        items_per_page (int, optional): Number of items to display per page. Defaults to 10.

    Returns:
        tuple[discord.Embed, TableView]: The embed containing the table and the view with navigation
        buttons
    """
    if not data:
        return discord.Embed(title=title, description="No data to display"), None

    view = TableView(data, items_per_page)
    embed = create_table_page(data, 0, items_per_page)
    embed.title = title

    return embed, view
