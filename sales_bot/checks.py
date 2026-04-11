from __future__ import annotations

from typing import TYPE_CHECKING

from discord import app_commands

from sales_bot.exceptions import PermissionDeniedError

if TYPE_CHECKING:
    import discord

    from sales_bot.bot import SalesBot


def admin_only() -> app_commands.Check:
    async def predicate(interaction: discord.Interaction) -> bool:
        bot = interaction.client
        if not isinstance(bot, SalesBot):
            return False

        if await bot.services.admins.is_admin(interaction.user.id):
            return True

        raise PermissionDeniedError("Only bot admins can use this command.")

    return app_commands.check(predicate)
