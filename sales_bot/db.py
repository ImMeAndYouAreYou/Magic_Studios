from __future__ import annotations

import asyncio
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
from sales_bot.services.ai_assistant import AIAssistantService
from sales_bot.services.admins import AdminService
from sales_bot.services.blacklist import BlacklistService
from sales_bot.services.cart import CartService
from sales_bot.services.delivery import DeliveryService
from sales_bot.services.discount_codes import DiscountCodeService
from sales_bot.services.discounts import DiscountService
from sales_bot.services.engagement import EventService, GiveawayService, PollService
from sales_bot.services.notifications import NotificationService
from sales_bot.services.oauth import RobloxOAuthService
from sales_bot.services.orders import OrderService
from sales_bot.services.ownership import OwnershipService
from sales_bot.services.panels import AdminPanelService
from sales_bot.services.payments import PaymentService
from sales_bot.services.redeem_codes import RedeemCodeService
from sales_bot.services.roblox_creator import RobloxCreatorService
from sales_bot.services.special_systems import SpecialSystemService
from sales_bot.services.systems import SystemService
from sales_bot.services.vouches import VouchService
from sales_bot.services.web_auth import WebAuthService
from sales_bot.ui.ownership import ClaimRolePanelView
from sales_bot.ui.orders import OrderDecisionView, OrderPanelView
from sales_bot.web import create_web_app


LOGGER = logging.getLogger(__name__)


class SalesBot(commands.Bot):
    EXTENSIONS = (
        "sales_bot.cogs.admin",
        "sales_bot.cogs.systems",
        "sales_bot.cogs.blacklist",
        "sales_bot.cogs.payments",
        "sales_bot.cogs.ownership",
        "sales_bot.cogs.orders",
        "sales_bot.cogs.vouches",
        "sales_bot.cogs.oauth",
        "sales_bot.cogs.roblox_owner",
        "sales_bot.cogs.support",
        "sales_bot.cogs.engagement",
        "sales_bot.cogs.ai_support",
    )

    def __init__(self, settings: Settings) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix=commands.when_mentioned, intents=intents)
        self.settings = settings
        self.database = Database(
            path=settings.sqlite_path,
            schema_path=Path(__file__).with_name("sql") / "schema.sql",
            database_url=settings.database_url,
        )
        self.http_session: aiohttp.ClientSession | None = None
        self.web_runner: web.AppRunner | None = None
        self.services: ServiceContainer
        self._command_sync_lock = asyncio.Lock()
        self._maintenance_task: asyncio.Task[None] | None = None
        self._roblox_gamepass_cache_warmup_task: asyncio.Task[None] | None = None
        self.tree.on_error = self.on_app_command_error

    async def setup_hook(self) -> None:
        await self.database.connect()
        self.http_session = aiohttp.ClientSession()
        self.services = ServiceContainer(
            admins=AdminService(self.database, self.settings.owner_user_id),
            blacklist=BlacklistService(self.database),
            cart=CartService(self.database),
            discount_codes=DiscountCodeService(self.database),
            discounts=DiscountService(self.database),
            systems=SystemService(self.database, self.settings.data_dir / "systems"),
            ownership=OwnershipService(self.database),
            orders=OrderService(self.database),
            delivery=DeliveryService(),
            notifications=NotificationService(self.database),
            payments=PaymentService(self.database),
            redeem_codes=RedeemCodeService(self.database),
            vouches=VouchService(self.database),
            oauth=RobloxOAuthService(self.database, self.settings),
            roblox_creator=RobloxCreatorService(self.database, self.settings),
            panels=AdminPanelService(self.database, self.settings.admin_panel_session_minutes),
            polls=PollService(self.database),
            giveaways=GiveawayService(self.database),
            events=EventService(self.database),
            ai_assistant=AIAssistantService(self.database, self.settings),
            web_auth=WebAuthService(self.database, self.settings),
            special_systems=SpecialSystemService(self.database),
        )

        for extension in self.EXTENSIONS:
            await self.load_extension(extension)

        await self._restore_persistent_views()
        await self._start_web_server()
        self._maintenance_task = asyncio.create_task(self._maintenance_loop(), name="engagement-maintenance")
        self._roblox_gamepass_cache_warmup_task = asyncio.create_task(
            self.services.roblox_creator.warm_gamepass_cache(self),
            name="roblox-gamepass-cache-warmup",
        )

        if self.settings.sync_commands_on_startup:
            await self._sync_commands()

    async def _maintenance_loop(self) -> None:
        await self.wait_until_ready()
        while not self.is_closed():
            try:
                finalized_polls = await self.services.polls.close_due_polls(self)
                finalized_giveaways = await self.services.giveaways.close_due_giveaways(self)
                finalized_events = await self.services.events.close_due_events(self)
                if finalized_polls or finalized_giveaways or finalized_events:
                    LOGGER.info(
                        "Maintenance finalized %s polls, %s giveaways, and %s events",
                        finalized_polls,
                        finalized_giveaways,
                        finalized_events,
                    )
            except asyncio.CancelledError:
                raise
            except Exception:
                LOGGER.exception("Background maintenance loop failed")

            await asyncio.sleep(30)

    async def _sync_commands(self) -> None:
        async with self._command_sync_lock:
            if self.settings.dev_guild_id:
                guild = discord.Object(id=self.settings.dev_guild_id)
                self.tree.copy_global_to(guild=guild)
                synced = await self.tree.sync(guild=guild)
                LOGGER.info("Synced %s commands to dev guild %s", len(synced), self.settings.dev_guild_id)
                return

            if self.settings.primary_guild_id:
                guild = discord.Object(id=self.settings.primary_guild_id)
                self.tree.clear_commands(guild=guild)
                await self.tree.sync(guild=guild)
                LOGGER.info("Cleared guild-only command overrides for primary guild %s", self.settings.primary_guild_id)

            synced = await self.tree.sync()
            LOGGER.info("Synced %s global commands", len(synced))

    def _schedule_command_resync(self) -> None:
        if self._command_sync_lock.locked():
            return
        asyncio.create_task(self._sync_commands())

    async def _restore_persistent_views(self) -> None:
        self.add_view(OrderPanelView(self))
        self.add_view(ClaimRolePanelView(self))

        pending_orders = await self.services.orders.list_pending_requests()
        for order in pending_orders:
            if order.owner_message_id:
                self.add_view(
                    OrderDecisionView(self, order.id, order.user_id),
                    message_id=order.owner_message_id,
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
        if self._roblox_gamepass_cache_warmup_task is not None:
            self._roblox_gamepass_cache_warmup_task.cancel()
            await asyncio.gather(self._roblox_gamepass_cache_warmup_task, return_exceptions=True)
            self._roblox_gamepass_cache_warmup_task = None

        if self._maintenance_task is not None:
            self._maintenance_task.cancel()
            await asyncio.gather(self._maintenance_task, return_exceptions=True)
            self._maintenance_task = None

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

        if isinstance(original_error, discord.NotFound) and original_error.code == 10062:
            LOGGER.debug("Ignoring expired interaction error handling after command response race")
            return

        if isinstance(original_error, discord.HTTPException) and original_error.code == 40060:
            LOGGER.warning("Skipping duplicate interaction acknowledgement error handling")
            return

        if isinstance(error, app_commands.CommandSignatureMismatch):
            LOGGER.warning("Command signature mismatch detected for %s. Scheduling command resync.", error.command.name)
            self._schedule_command_resync()
            message = (
                "הפקודה עודכנה אבל דיסקורד עדיין מחזיק גרסה ישנה שלה. "
                "ביצעתי סנכרון מחדש; סגור ופתח שוב את תפריט הפקודות ונסה שוב בעוד כמה שניות."
            )
            responder = interaction.followup.send if interaction.response.is_done() else interaction.response.send_message
            try:
                await responder(message, ephemeral=True)
            except discord.HTTPException:
                pass
            return

        command_name = interaction.command.qualified_name if interaction.command else "unknown"
        user_id = interaction.user.id if interaction.user else "unknown"

        if isinstance(error, app_commands.CheckFailure):
            LOGGER.info(
                "Application command blocked for %s by user %s: %s",
                command_name,
                user_id,
                error,
            )
        elif isinstance(error, app_commands.CommandOnCooldown):
            LOGGER.info(
                "Application command on cooldown for %s by user %s: retry after %.1fs",
                command_name,
                user_id,
                error.retry_after,
            )
        elif isinstance(original_error, SalesBotError):
            LOGGER.warning(
                "Application command failed for %s by user %s: %s",
                command_name,
                user_id,
                original_error,
            )
        else:
            LOGGER.exception(
                "Application command error",
                exc_info=(type(original_error), original_error, original_error.__traceback__),
            )

        if isinstance(original_error, PermissionDeniedError):
            message = str(original_error)
        elif isinstance(original_error, SalesBotError):
            message = str(original_error)
        elif isinstance(error, app_commands.CheckFailure):
            message = str(error) or "You cannot use that command."
        elif isinstance(error, app_commands.CommandOnCooldown):
            message = f"Try again in {error.retry_after:.1f} seconds."
        else:
            message = "An unexpected error occurred while processing that command."

        responder = interaction.followup.send if interaction.response.is_done() else interaction.response.send_message
        try:
            await responder(message, ephemeral=True)
        except discord.HTTPException as exc:
            if exc.code in {40060, 10062}:
                try:
                    await interaction.followup.send(message, ephemeral=True)
                    return
                except discord.HTTPException:
                    return
            LOGGER.exception("Failed to send interaction error response")
