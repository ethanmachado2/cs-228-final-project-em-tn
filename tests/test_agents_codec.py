"""Contract <-> A2A part round trips and the error-code mapping (Task 1, Req 1.1-1.3)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("a2a")

from fpv import config as C  # noqa: E402
from fpv.adapter import XytMinutia, XytTemplate  # noqa: E402
from fpv.agents import codec  # noqa: E402
from fpv.extract import ExtractionResult  # noqa: E402
from fpv.matcher import InsufficientMinutiae, MatcherError, MatchResult  # noqa: E402
from fpv.preprocess import PreprocessResult  # noqa: E402
from fpv.quality import QualityResult, QualityToolError  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def conf() -> C.PipelineConfig:
    return C.load(ROOT / "configs" / "pipeline.v1.yaml", root=ROOT)


def wire(p: codec.Payload) -> codec.Payload:
    """Through protobuf parts and the JSON wire format, as the SDK sends them."""
    from a2a.types import Message
    from google.protobuf.json_format import MessageToJson, Parse

    msg = Message(message_id="m", parts=codec.to_parts(p))
    return codec.from_parts(Parse(MessageToJson(msg), Message()).parts)


def arr(shape=(309, 288), seed=0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=shape, dtype=np.uint8)


def test_image_round_trip_is_byte_identical():
    img = arr()
    out = codec.decode_image(wire(codec.encode_image(img)))
    assert out.dtype == np.uint8 and np.array_equal(out, img)


def test_quality_round_trip():
    q = QualityResult(score=47, actionable_feedback={"UniformImage": 0.1 + 0.2, "FingerJetFX_MinutiaeCount": 31.0},
                      raw_output='"Filename",...\n', tool_version="2.3.0", duration_ms=123.456789012345,
                      upscale=3.0, error=None, scored_shape=(309, 288))
    assert codec.decode_quality(wire(codec.encode_quality(q))) == q
    none = QualityResult(score=None, actionable_feedback={}, raw_output="", tool_version="2.3.0",
                         duration_ms=1.0, error="QualityScore=NA")
    assert codec.decode_quality(wire(codec.encode_quality(none))) == none


def test_preprocess_round_trip(conf):
    att = conf.attempt(2)
    cid = conf.configuration_id(att)
    pre = PreprocessResult(attempt=att, enhanced=arr(), mask=(arr(seed=1) > 127).astype(np.uint8) * 255,
                           scale_factor=att.scale_factor, original_shape=(103, 96), configuration_id=cid,
                           timings_ms={"segmentation": 10.58123456789, "enhancement": 148.29})
    out = codec.decode_preprocess(wire(codec.encode_preprocess(pre)), conf)
    assert out.attempt == att and out.configuration_id == cid and out.original_shape == (103, 96)
    assert out.scale_factor == pre.scale_factor and dict(out.timings_ms) == dict(pre.timings_ms)
    assert np.array_equal(out.enhanced, pre.enhanced) and np.array_equal(out.mask, pre.mask)


def test_attempt_with_wrong_configuration_id_is_invalid(conf):
    with pytest.raises(codec.InvalidInput):
        codec.decode_attempt({"attempt_no": 1, "configuration_id": "cfg_nope"}, conf)
    with pytest.raises(codec.InvalidInput):
        codec.decode_attempt({"attempt_no": 9, "configuration_id": "x"}, conf)


def test_extraction_round_trip_equals_pyfing_namedtuples():
    ms = tuple(codec.Minutia(x, 2 * x + 1, -3.141592653589793 + x / 7, "ending" if x % 2 else "bifurcation",
                             0.123456789 * x) for x in range(25))
    ex = ExtractionResult(minutiae=ms, count=25, configuration_id="cfg_x", duration_ms=17.9)
    out = codec.decode_extraction(wire(codec.encode_extraction(ex)))
    assert out == ex
    assert all(type(m.x) is int and type(m.direction) is float for m in out.minutiae)
    assert out.minutiae[3] == (3, 7, -3.141592653589793 + 3 / 7, "ending", 0.123456789 * 3)


def test_xyt_and_match_round_trip():
    t = XytTemplate(tuple(XytMinutia(i, 100 - i, (37 * i) % 360, 50 + i) for i in range(30)))
    assert codec.decode_xyt(wire(codec.encode_xyt(t))) == t
    m = MatchResult(score=296, probe_count=35, gallery_count=41, tool_version="5.0.0", duration_ms=268.9)
    assert codec.decode_match(wire(codec.encode_match(m))) == m


def test_missing_and_mistyped_fields_are_invalid_input():
    with pytest.raises(codec.InvalidInput):
        codec.decode_image(codec.Payload())
    with pytest.raises(codec.InvalidInput):
        codec.decode_image(codec.Payload(files={"image": (b"BM...", codec.PNG)}))
    with pytest.raises(codec.InvalidInput):
        codec.decode_match(codec.Payload({"score": 1.5, "probe_count": 1, "gallery_count": 1,
                                          "tool_version": "x", "duration_ms": 0}))


@pytest.mark.parametrize("exc, code", [
    (codec.InvalidInput("bad"), "invalid_input"),
    (QualityToolError("nfiq2 crashed"), "nfiq2_tool_error"),
    (InsufficientMinutiae("gallery", 7), "insufficient_minutiae"),
    (MatcherError("timeout"), "bozorth3_tool_error"),
    (RuntimeError("boom"), "stage_error"),
])
def test_error_mapping_round_trip(exc, code):
    d = wire(codec.Payload(codec.encode_error(exc))).data
    assert d["error_code"] == code and d["error_type"] == type(exc).__name__
    back = codec.decode_error(d)
    assert str(back) == str(exc)
    if isinstance(exc, (QualityToolError, InsufficientMinutiae, MatcherError)):
        assert type(back) is type(exc)
    else:
        assert isinstance(back, codec.RemoteStageError) and back.code == code
    if isinstance(exc, InsufficientMinutiae):
        assert (back.which, back.count) == ("gallery", 7)
