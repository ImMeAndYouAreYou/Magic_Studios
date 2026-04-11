from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class SystemRecord:
    id: int
    name: str
    description: str
    file_path: str
    image_path: str | None
    paypal_link: str | None
    created_by: int | None
    created_at: str


@dataclass(slots=True, frozen=True)
class BlacklistEntry:
    user_id: int
    display_label: str
    blacklisted_by: int | None
    blacklisted_at: str


@dataclass(slots=True, frozen=True)
class AppealRecord:
    id: int
    user_id: int
    answer_one: str
    answer_two: str
    owner_message_id: int | None
    status: str
    submitted_at: str
    reviewed_at: str | None
    reviewed_by: int | None


@dataclass(slots=True, frozen=True)
class PurchaseRecord:
    id: int
    user_id: int
    system_id: int
    status: str
    paypal_link: str
    created_at: str
    completed_at: str | None


@dataclass(slots=True, frozen=True)
class DeliveryRecord:
    id: int
    user_id: int
    system_id: int
    channel_id: int
    message_id: int
    source: str
    sent_at: str


@dataclass(slots=True, frozen=True)
class VouchStats:
    total: int
    average_rating: float


@dataclass(slots=True, frozen=True)
class RobloxLinkRecord:
    user_id: int
    roblox_sub: str
    roblox_username: str | None
    roblox_display_name: str | None
    profile_url: str | None
    linked_at: str
