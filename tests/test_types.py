import pytest

from fpv.types import Finger, Hand, Identity, Status


@pytest.mark.parametrize(
    ("user", "hand", "finger", "expected"),
    [
        ("1", "Left", "thumb", Identity(1, Hand.LEFT, Finger.THUMB)),
        (600, "Right", "little", Identity(600, Hand.RIGHT, Finger.LITTLE)),
        (" 42 ", "left", "INDEX", Identity(42, Hand.LEFT, Finger.INDEX)),
        ("100", "RIGHT", "Middle", Identity(100, Hand.RIGHT, Finger.MIDDLE)),
        ("7", "rIgHt", "ring", Identity(7, Hand.RIGHT, Finger.RING)),
    ],
)
def test_parse_accepts_and_normalizes(user, hand, finger, expected):
    ident = Identity.parse(user, hand, finger)
    assert ident == expected
    assert ident.hand.value in ("Left", "Right")
    assert ident.finger.value == ident.finger.value.lower()


@pytest.mark.parametrize("user", [0, "0", 601, "601", -1, "abc", "", "1.5", "１２", True])
def test_parse_rejects_bad_user(user):
    with pytest.raises(ValueError, match="1-600"):
        Identity.parse(user, "Left", "thumb")


@pytest.mark.parametrize("hand", ["", "L", "Lefty", "both", "middle"])
def test_parse_rejects_bad_hand(hand):
    with pytest.raises(ValueError, match="Left, Right"):
        Identity.parse(1, hand, "thumb")


@pytest.mark.parametrize("finger", ["", "pinky", "thumbs", "index_finger", "Left"])
def test_parse_rejects_bad_finger(finger):
    with pytest.raises(ValueError, match="thumb, index, middle, ring, little"):
        Identity.parse(1, "Left", finger)


def test_constructor_validates_range_and_types():
    with pytest.raises(ValueError):
        Identity(0, Hand.LEFT, Finger.THUMB)
    with pytest.raises(ValueError):
        Identity(1, "Left", Finger.THUMB)  # type: ignore[arg-type]


def test_identity_is_frozen_and_hashable():
    a = Identity.parse(5, "left", "ring")
    assert a == Identity.parse("5", "LEFT", "RING")
    assert len({a, Identity.parse(5, "Left", "ring")}) == 1
    with pytest.raises(AttributeError):
        a.user_id = 6  # type: ignore[misc]


def test_status_values():
    assert {s.value for s in Status} == {
        "ENROLLED", "REJECTED_QUALITY", "REJECTED_MINUTIAE", "MATCH", "NO_MATCH", "NOT_ENROLLED", "ERROR",
    }
