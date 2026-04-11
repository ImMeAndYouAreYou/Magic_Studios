from __future__ import annotations

import logging
from pathlib import Path

import aiohttp
import discord
from aiohttp import web
from discord import app_commands
from discord.ext import commands

from sales_bot.config import Settings
from sales_bot.db import Database
from sales_bot.exceptions import PermissionDeniedError, SalesBotError
from sales_bot.services import ServiceContainer
from sales_bot.services.admins import AdminService
from sales_bot.services.blacklist import BlacklistService
from sales_bot.services.delivery import DeliveryService
from sales_bot.services.oauth import RobloxOAuthService
from sales_bot.services.ownership import OwnershipService
from sales_bot.services.payments import PaymentService
from sales_bot.services.systems import SystemService
from sales_bot.services.vouches import VouchService
from sales_bot.ui.appeals import AppealDecisionView
from sales_bot.web import create_web_app


LOGGER = logging.getLogger(__name__)


class SalesBot(commands.Bot):
    EXTENSIONS = (
        "sales_bot.cogs.admin",
        "sales_bot.cogs.systems",
        "sales_bot.cogs.blacklist",
        "sales_bot.cogs.payments",
        "sales_bot.cogs.ownership",
        "sales_bot.cogs.vouches",
        "sales_bot.cogs.oauth",
        "sales_bot.cogs.support",
    )

    def __init__(self, settings: Settings) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix=commands.when_mentioned, intents=intents)
        self.settings = settings
        self.database = Database(
            path=settings.sqlite_path,
            schema_path=Path(__file__).with_name("sql") / "schema.sql",
        )
        self.http_session: aiohttp.ClientSession | None = None
        self.web_runner: web.AppRunner | None = None
        self.services: ServiceContainer
        self.tree.on_error = self.on_app_command_error

    async def setup_hook(self) -> None:
        await self.database.connect()
        self.http_session = aiohttp.ClientSession()
        self.services = ServiceContainer(
            admins=AdminService(self.database, self.settings.owner_user_id),
            blacklist=BlacklistService(self.database),
            systems=SystemService(self.database, self.settings.data_dir / "systems"),
            ownership=OwnershipService(self.database),
            delivery=DeliveryService(),
            payments=PaymentService(self.database),
            vouches=VouchService(self.database),
            oauth=RobloxOAuthService(self.database, self.settings),
        )


        for extension in self.EXTENSIONS:
            await self.load_extension(extension)

        await self._restore_persistent_views()
        await self._start_web_server()

        if self.settings.sync_commands_on_startup:
            await self._sync_commands()

    async def _sync_commands(self) -> None:
        if self.settings.dev_guild_id:
            guild = discord.Object(id=self.settings.dev_guild_id)
            self.tree.copy_global_to(guild=guild)
            synced = await self.tree.sync(guild=guild)
            LOGGER.info("Synced %s commands to dev guild %s", len(synced), self.settings.dev_guild_id)
            return

        synced = await self.tree.sync()
        LOGGER.info("Synced %s global commands", len(synced))

    async def _restore_persistent_views(self) -> None:
        pending_appeals = await self.services.blacklist.list_pending_appeals()
        for appeal in pending_appeals:
            if appeal.owner_message_id:
                self.add_view(
                    AppealDecisionView(self, appeal.id, appeal.user_id),
                    message_id=appeal.owner_message_id,
                )

    async def _start_web_server(self) -> None:
        app = create_web_app(self)
        self.web_runner = web.AppRunner(app)
        await self.web_runner.setup()
        site = web.TCPSite(self.web_runner, self.settings.web_host, self.settings.web_port)
        await site.start()
        LOGGER.info(
            "HTTP server listening on %s:%s",
            self.settings.web_host,
            self.settings.web_port,
        )

    async def close(self) -> None:
        if self.web_runner is not None:
            await self.web_runner.cleanup()
            self.web_runner = None

        if self.http_session is not None:
            await self.http_session.close()
            self.http_session = None

        await self.database.close()
        await super().close()

    async def on_ready(self) -> None:
        LOGGER.info("Logged in as %s (%s)", self.user, self.user.id if self.user else "unknown")

    async def on_app_command_error(
        self,
        interaction: discord.Interaction,
        error: app_commands.AppCommandError,
    ) -> None:
        original_error = getattr(error, "original", error)
        LOGGER.exception("Application command error", exc_info=original_error)

        if isinstance(original_error, PermissionDeniedError):
            message = str(original_error)
        elif isinstance(original_error, SalesBotError):
            message = str(original_error)
        elif isinstance(error, app_commands.CommandOnCooldown):
            message = f"Try again in {error.retry_after:.1f} seconds."
        else:
            message = "An unexpected error occurred while processing that command."

        responder = interaction.followup.send if interaction.response.is_done() else interaction.response.send_message
        await responder(message, ephemeral=True)
