from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Iterable, Sequence

import aiosqlite
import asyncpg

type PgOperationResult = Any


class Database:
    def __init__(self, path: Path, schema_path: Path, database_url: str | None = None) -> None:
        self.path = path
        self.schema_path = schema_path
        self.database_url = database_url
        self._connection: aiosqlite.Connection | None = None
        # Changed: now a pool instead of a single connection
        self._pg_pool: asyncpg.Pool | None = None
        self._pg_lock = asyncio.Lock()

    @property
    def connection(self) -> aiosqlite.Connection | asyncpg.Connection:
        # This property is only used for SQLite – PostgreSQL now uses the pool.
        # Keeping it for backward compatibility with the SQLite path.
        if self.database_url:
            if self._pg_pool is None:
                raise RuntimeError("Database connection has not been initialized")
            # Returning None would be misleading; better to raise a clear error
            raise RuntimeError("Use pool.acquire() for PostgreSQL instead of raw connection")
        if self._connection is None:
            raise RuntimeError("Database connection has not been initialized")
        return self._connection

    async def connect(self) -> None:
        if self.database_url:
            # Create a pool with a single connection.
            # Pool will automatically reconnect if the connection drops.
            self._pg_pool = await asyncpg.create_pool(
                self.database_url,
                min_size=1,
                max_size=1,
                statement_cache_size=0,
                # Close connections that have been idle for 5 minutes
                # to prevent Render's idle timeout (typically ~10 min).
                max_inactive_connection_lifetime=300,
            )
            schema_path = self.schema_path.with_name("schema_postgres.sql")
            schema = schema_path.read_text(encoding="utf-8")
            # Acquire a connection just for the initial schema run
            async with self._pg_pool.acquire() as conn:
                await conn.execute(schema)
            await self._run_migrations()
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = await aiosqlite.connect(self.path)
        self._connection.row_factory = aiosqlite.Row
        schema = self.schema_path.read_text(encoding="utf-8")
        await self._connection.executescript(schema)
        await self._run_migrations()
        await self._connection.commit()

    async def _run_migrations(self) -> None:
        await self._ensure_column("systems", "roblox_gamepass_id", "TEXT")
        await self._ensure_column("systems", "is_visible_on_website", "BOOLEAN NOT NULL DEFAULT TRUE")
        await self._ensure_column("systems", "is_for_sale", "BOOLEAN NOT NULL DEFAULT TRUE")
        await self._ensure_column("systems", "is_in_stock", "BOOLEAN NOT NULL DEFAULT TRUE")
        await self._ensure_column("systems", "website_price", "TEXT")
        await self._ensure_column("systems", "website_currency", "TEXT NOT NULL DEFAULT 'ILS'")
        await self._ensure_column("systems", "is_special_system", "BOOLEAN NOT NULL DEFAULT FALSE")
        await self._ensure_column("blacklist_entries", "reason", "TEXT NOT NULL DEFAULT ''")
        await self._ensure_column("order_requests", "roblox_username", "TEXT")
        await self._ensure_column("order_requests", "admin_reply", "TEXT")
        await self._ensure_column("website_checkout_orders", "paypal_status", "TEXT NOT NULL DEFAULT 'not-started'")
        await self._ensure_column("website_checkout_orders", "paypal_order_id", "TEXT")
        await self._ensure_column("website_checkout_orders", "paypal_capture_id", "TEXT")
        await self._ensure_column("website_checkout_orders", "paypal_approval_url", "TEXT")
        await self._ensure_column("website_checkout_orders", "paypal_payload_json", "TEXT")
        await self._ensure_column("website_checkout_orders", "fulfillment_mode", "TEXT NOT NULL DEFAULT 'self'")
        await self.execute(
            "CREATE INDEX IF NOT EXISTS idx_website_checkout_orders_paypal_order_id ON website_checkout_orders(paypal_order_id)"
        )

    async def _ensure_column(self, table_name: str, column_name: str, column_sql: str) -> None:
        if self.database_url:
            row = await self.fetchone(
                """
                SELECT 1
                FROM information_schema.columns
                WHERE table_schema = current_schema()
                  AND table_name = ?
                  AND column_name = ?
                """,
                (table_name, column_name),
            )
            if row is not None:
                return
            await self.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_sql}")
            return

        rows = await self.fetchall(f"PRAGMA table_info({table_name})")
        if any(str(row["name"]) == column_name for row in rows):
            return
        sqlite_connection = self.connection
        if not isinstance(sqlite_connection, aiosqlite.Connection):
            return
        await sqlite_connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_sql}")

    async def close(self) -> None:
        if self._pg_pool is not None:
            await self._pg_pool.close()
            self._pg_pool = None
        if self._connection is not None:
            await self._connection.close()
            self._connection = None

    async def execute(self, query: str, parameters: Sequence[Any] = ()) -> None:
        if self.database_url:
            await self._run_pg(lambda pg_connection: pg_connection.execute(self._translate_query(query), *parameters))
            return

        sqlite_connection = self.connection
        assert isinstance(sqlite_connection, aiosqlite.Connection)
        await sqlite_connection.execute(query, parameters)
        await sqlite_connection.commit()

    async def executemany(self, query: str, parameters: Iterable[Sequence[Any]]) -> None:
        if self.database_url:
            parameter_list = list(parameters)
            await self._run_pg(lambda pg_connection: pg_connection.executemany(self._translate_query(query), parameter_list))
            return

        sqlite_connection = self.connection
        assert isinstance(sqlite_connection, aiosqlite.Connection)
        await sqlite_connection.executemany(query, parameters)
        await sqlite_connection.commit()

    async def fetchone(self, query: str, parameters: Sequence[Any] = ()) -> aiosqlite.Row | None:
        if self.database_url:
            return await self._run_pg(lambda pg_connection: pg_connection.fetchrow(self._translate_query(query), *parameters))

        sqlite_connection = self.connection
        assert isinstance(sqlite_connection, aiosqlite.Connection)
        async with sqlite_connection.execute(query, parameters) as cursor:
            return await cursor.fetchone()

    async def fetchall(self, query: str, parameters: Sequence[Any] = ()) -> list[aiosqlite.Row]:
        if self.database_url:
            rows = await self._run_pg(lambda pg_connection: pg_connection.fetch(self._translate_query(query), *parameters))
            return list(rows)

        sqlite_connection = self.connection
        assert isinstance(sqlite_connection, aiosqlite.Connection)
        async with sqlite_connection.execute(query, parameters) as cursor:
            return await cursor.fetchall()

    async def insert(self, query: str, parameters: Sequence[Any] = ()) -> int:
        if self.database_url:
            translated = self._translate_query(query)
            if "RETURNING" not in translated.upper():
                translated = translated.rstrip().rstrip(";") + " RETURNING id"
            value = await self._run_pg(lambda pg_connection: pg_connection.fetchval(translated, *parameters))
            return int(value)

        sqlite_connection = self.connection
        assert isinstance(sqlite_connection, aiosqlite.Connection)
        cursor = await sqlite_connection.execute(query, parameters)
        await sqlite_connection.commit()
        return int(cursor.lastrowid)

    @staticmethod
    def _translate_query(query: str) -> str:
        translated = query.replace(" COLLATE NOCASE", "")
        parts = translated.split("?")
        if len(parts) == 1:
            return translated

        rebuilt: list[str] = [parts[0]]
        for index, part in enumerate(parts[1:], start=1):
            rebuilt.append(f"${index}")
            rebuilt.append(part)
        return "".join(rebuilt)

    async def _run_pg(self, operation: Callable[[asyncpg.Connection], Awaitable[PgOperationResult]]) -> PgOperationResult:
        if self._pg_pool is None:
            raise RuntimeError("PostgreSQL pool has not been initialized")
        # Acquire a fresh connection from the pool. The pool ensures it's alive.
        async with self._pg_pool.acquire() as pg_connection:
            async with self._pg_lock:   # keep serialization for safety
                return await operation(pg_connection)
