from __future__ import annotations

import asyncio

from dotenv import load_dotenv

from sales_bot.bot import SalesBot
from sales_bot.config import Settings
from sales_bot.logging_config import configure_logging


async def main() -> None:
    load_dotenv()
    settings = Settings.from_env()
    configure_logging(settings.log_level)

    async with SalesBot(settings) as bot:
        await bot.start(settings.discord_token)


if __name__ == "__main__":
    asyncio.run(main())
