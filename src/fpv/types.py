"""Core identity and status types shared by the CLI, catalog and pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

USER_ID_MIN = 1
USER_ID_MAX = 600


class Hand(StrEnum):
    LEFT = "Left"
    RIGHT = "Right"


class Finger(StrEnum):
    THUMB = "thumb"
    INDEX = "index"
    MIDDLE = "middle"
    RING = "ring"
    LITTLE = "little"


class Status(StrEnum):
    ENROLLED = "ENROLLED"
    REJECTED_QUALITY = "REJECTED_QUALITY"
    REJECTED_MINUTIAE = "REJECTED_MINUTIAE"
    MATCH = "MATCH"
    NO_MATCH = "NO_MATCH"
    NOT_ENROLLED = "NOT_ENROLLED"
    ERROR = "ERROR"


def _allowed(enum_cls: type[StrEnum]) -> str:
    return ", ".join(m.value for m in enum_cls)


def parse_hand(value: str | Hand) -> Hand:
    if isinstance(value, Hand):
        return value
    text = str(value).strip().lower()
    for member in Hand:
        if member.value.lower() == text:
            return member
    raise ValueError(f"invalid hand {value!r}; allowed values: {_allowed(Hand)}")


def parse_finger(value: str | Finger) -> Finger:
    if isinstance(value, Finger):
        return value
    text = str(value).strip().lower()
    for member in Finger:
        if member.value == text:
            return member
    raise ValueError(f"invalid finger {value!r}; allowed values: {_allowed(Finger)}")


def parse_user_id(value: int | str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"invalid user ID {value!r}; allowed values: integers {USER_ID_MIN}-{USER_ID_MAX}")
    if isinstance(value, int):
        uid = value
    else:
        text = str(value).strip()
        if not text.isdecimal() or not text.isascii():
            raise ValueError(
                f"invalid user ID {value!r}; allowed values: integers {USER_ID_MIN}-{USER_ID_MAX}"
            )
        uid = int(text)
    if not USER_ID_MIN <= uid <= USER_ID_MAX:
        raise ValueError(f"invalid user ID {value!r}; allowed values: integers {USER_ID_MIN}-{USER_ID_MAX}")
    return uid


@dataclass(frozen=True, slots=True)
class Identity:
    """A claimed or labeled identity: (user_id, hand, finger)."""

    user_id: int
    hand: Hand
    finger: Finger

    def __post_init__(self) -> None:
        if isinstance(self.user_id, bool) or not isinstance(self.user_id, int):
            raise ValueError(f"user_id must be an int, got {self.user_id!r}")
        if not USER_ID_MIN <= self.user_id <= USER_ID_MAX:
            raise ValueError(
                f"invalid user ID {self.user_id!r}; allowed values: integers {USER_ID_MIN}-{USER_ID_MAX}"
            )
        if not isinstance(self.hand, Hand):
            raise ValueError(f"hand must be a Hand, got {self.hand!r}")
        if not isinstance(self.finger, Finger):
            raise ValueError(f"finger must be a Finger, got {self.finger!r}")

    @classmethod
    def parse(cls, user: int | str, hand: str | Hand, finger: str | Finger) -> "Identity":
        """Parse user input case-insensitively; raises ValueError listing the allowed values."""
        return cls(parse_user_id(user), parse_hand(hand), parse_finger(finger))

    def __str__(self) -> str:
        return f"user {self.user_id}, {self.hand.value} {self.finger.value}"
