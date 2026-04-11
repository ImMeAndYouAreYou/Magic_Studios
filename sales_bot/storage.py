from __future__ import annotations

import re
from pathlib import Path
from uuid import uuid4

import discord


def slugify(value: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-")
    return sanitized[:48] or "item"


async def save_attachment(attachment: discord.Attachment, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    filename = f"{uuid4().hex}_{attachment.filename}"
    target = directory / filename
    await attachment.save(target)
    return target


def remove_path(path: str | Path | None) -> None:
    if not path:
        return
    target = Path(path)
    if target.is_file():
        target.unlink(missing_ok=True)
        parent = target.parent
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()


def system_message_files(file_path: str, image_path: str | None) -> tuple[list[discord.File], str | None]:
    attachments: list[discord.File] = []
    image_name: str | None = None

    if image_path:
        image_source = Path(image_path)
        image_name = image_source.name
        attachments.append(discord.File(image_source, filename=image_name))

    file_source = Path(file_path)
    attachments.append(discord.File(file_source, filename=file_source.name))
    return attachments, image_name
