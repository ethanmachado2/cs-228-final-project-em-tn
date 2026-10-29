import copy
import re

import pytest
import yaml

from fpv import __version__
from fpv import config as C
from fpv.cli import main


@pytest.fixture
def raw(default_config_path):
    return yaml.safe_load(default_config_path.read_text())


# ------------------------------------------------------------------ load / validate


def test_default_config_loads(default_config_path, project_root):
    conf = C.load(default_config_path, root=project_root)
    assert conf.thresholds == C.Thresholds(nfiq2_min=5, minutiae_min=12, bozorth3_match=40)
    assert conf.max_presentations == 3
    assert conf.max_attempts == 3
    assert conf.adapter_version == "xyt-adapter/1"
    assert conf.tools["nfiq2"] == "2.3.0" and conf.tools["nbis"] == "5.0.0" and conf.tools["pyfing"] == "0.7.3"
    assert conf.path("db") == project_root / "data/db/fpv.sqlite"
    assert re.fullmatch(r"cs_[0-9a-f]{12}", conf.config_set_id)


@pytest.mark.parametrize(
    ("section", "key", "bad"),
    [
        ("thresholds", "nfiq2_min", -1),
        ("thresholds", "nfiq2_min", 101),
        ("thresholds", "minutiae_min", 9),
        ("thresholds", "minutiae_min", 0),
        ("thresholds", "bozorth3_match", -1),
        ("preprocess", "max_attempts", 0),
        ("preprocess", "max_attempts", 4),
        ("session", "max_presentations", 0),
        ("thresholds", "nfiq2_min", "5"),
        ("thresholds", "bozorth3_match", 40.5),
        ("thresholds", "minutiae_min", True),
    ],
)
def test_validation_rejects_out_of_range(raw, section, key, bad):
    raw[section][key] = bad
    with pytest.raises(C.ConfigError, match=rf"{section}\.{key}"):
        C.validate(raw)


@pytest.mark.parametrize(
    ("section", "key", "edge"),
    [
        ("thresholds", "nfiq2_min", 0),
        ("thresholds", "nfiq2_min", 100),
        ("thresholds", "minutiae_min", 10),
        ("thresholds", "bozorth3_match", 0),
        ("preprocess", "max_attempts", 1),
        ("session", "max_presentations", 1),
    ],
)
def test_validation_accepts_boundaries(raw, section, key, edge):
    raw[section][key] = edge
    C.validate(raw)


@pytest.mark.parametrize("bad", [0.5, 0.0, -1.0, 4.5, "3", True, None])
def test_validation_rejects_bad_nfiq2_upscale(raw, bad):
    raw["quality"]["nfiq2_upscale"] = bad
    with pytest.raises(C.ConfigError, match=r"quality\.nfiq2_upscale"):
        C.validate(raw)


@pytest.mark.parametrize("edge", [1.0, 2, 4.0])
def test_validation_accepts_nfiq2_upscale_range(raw, edge):
    raw["quality"]["nfiq2_upscale"] = edge
    assert C.validate(raw).nfiq2_upscale == float(edge)


def test_default_nfiq2_upscale_is_3x(default_config_path, project_root):
    assert C.load(default_config_path, root=project_root).nfiq2_upscale == 3.0


def test_nfiq2_upscale_changes_config_set_id(raw):
    base = C.validate(raw).config_set_id
    raw["quality"]["nfiq2_upscale"] = 2.0
    assert C.validate(raw).config_set_id != base


@pytest.mark.parametrize("drop", ["thresholds", "session", "quality", "preprocess", "adapter", "tools", "paths"])
def test_validation_requires_sections(raw, drop):
    del raw[drop]
    with pytest.raises(C.ConfigError, match=drop):
        C.validate(raw)


def test_validation_rejects_missing_threshold_and_unpinned_tool(raw):
    bad = copy.deepcopy(raw)
    del bad["thresholds"]["bozorth3_match"]
    with pytest.raises(C.ConfigError, match="bozorth3_match is required"):
        C.validate(bad)
    bad = copy.deepcopy(raw)
    bad["tools"]["nfiq2"] = 2.3
    with pytest.raises(C.ConfigError, match="tools.nfiq2"):
        C.validate(bad)


def test_load_reports_file_errors(tmp_path):
    with pytest.raises(C.ConfigError, match="cannot read"):
        C.load(tmp_path / "missing.yaml")
    p = tmp_path / "bad.yaml"
    p.write_text("thresholds: [unclosed")
    with pytest.raises(C.ConfigError, match="invalid YAML"):
        C.load(p)


def test_cli_refuses_to_start_on_invalid_config(tmp_path, raw):
    raw["thresholds"]["nfiq2_min"] = 500
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(raw))
    errors = []
    assert main(["--config", str(p), "doctor"], out=lambda s: None, err=errors.append) == 2
    assert "nfiq2_min=500" in errors[0]


def test_resolve_config_path_prefers_explicit_then_frozen(tmp_path):
    (tmp_path / "configs").mkdir()
    assert C.resolve_config_path(None, tmp_path) == tmp_path / C.DEFAULT_CONFIG
    (tmp_path / C.FROZEN_CONFIG).write_text("x")
    assert C.resolve_config_path(None, tmp_path) == tmp_path / C.FROZEN_CONFIG
    assert C.resolve_config_path("other.yaml", tmp_path) == C.Path("other.yaml")


# ------------------------------------------------------------------ canonical JSON / ids


def test_canonical_json_is_sorted_and_compact():
    assert C.canonical_json({"b": 1, "a": [1.5, None, True], "c": {"z": "é", "y": 2}}) == (
        '{"a":[1.5,null,true],"b":1,"c":{"y":2,"z":"\\u00e9"}}'
    )
    assert C.canonical_json((1, 2)) == C.canonical_json([1, 2])
    with pytest.raises(C.ConfigError):
        C.canonical_json({"x": float("nan")})


def test_config_set_id_stable_for_same_config(raw):
    a = C.validate(raw).config_set_id
    b = C.validate(copy.deepcopy(raw)).config_set_id
    reordered = dict(reversed(list(raw.items())))
    assert a == b == C.validate(reordered).config_set_id
    assert a == C.config_set_id(raw)


def _leaves(obj, path=()):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _leaves(v, (*path, k))
    elif isinstance(obj, list) and obj:
        for i, v in enumerate(obj):
            yield from _leaves(v, (*path, i))
    else:
        yield path, obj


def _set(obj, path, value):
    for p in path[:-1]:
        obj = obj[p]
    obj[path[-1]] = value


def _mutate(value):
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value * 2 + 0.001
    if isinstance(value, str):
        return value + "x"
    if isinstance(value, list):
        return [*value, {"attempt_no": 1}]
    raise AssertionError(value)


def test_config_set_id_changes_on_any_edit(raw):
    base = C.config_set_id(raw)
    seen = {base}
    leaves = list(_leaves(raw))
    assert len(leaves) > 20
    for path, value in leaves:
        edited = copy.deepcopy(raw)
        _set(edited, path, _mutate(value))
        cid = C.config_set_id(edited)
        assert cid != base, f"editing {path} did not change config_set_id"
        seen.add(cid)
    assert len(seen) == len(leaves) + 1  # every edit gives a distinct id
    # adding a new key changes it too
    edited = copy.deepcopy(raw)
    edited["thresholds"]["extra"] = 1
    assert C.config_set_id(edited) != base


def test_config_set_id_ignores_provenance(raw):
    with_prov = {**raw, "provenance": {"date": "2026-01-01", "runs": ["r1"]}}
    assert C.config_set_id(with_prov) == C.config_set_id(raw)


def test_configuration_id_format_and_sensitivity():
    attempt = {"attempt_no": 1, "segmentation": "GMFS"}
    tools = {"nfiq2": "2.3.0"}
    cid = C.configuration_id(attempt, "xyt-adapter/1", tools)
    assert re.fullmatch(r"cfg_[0-9a-f]{12}", cid)
    assert cid == C.configuration_id(dict(attempt), "xyt-adapter/1", dict(tools))
    assert cid != C.configuration_id({**attempt, "segmentation": "SUFS"}, "xyt-adapter/1", tools)
    assert cid != C.configuration_id(attempt, "xyt-adapter/2", tools)
    assert cid != C.configuration_id(attempt, "xyt-adapter/1", {"nfiq2": "2.3.1"})


# ------------------------------------------------------------------ pipeline_version


def test_pipeline_version_from_git_clean():
    v = C.pipeline_version(env={}, git=lambda: ("abc1234", False))
    assert v == f"{__version__}+gabc1234"


def test_pipeline_version_from_git_dirty():
    v = C.pipeline_version(env={}, git=lambda: ("abc1234", True))
    assert v == f"{__version__}+gabc1234.dirty"


def test_pipeline_version_unknown_without_commit():
    assert C.pipeline_version(env={}, git=lambda: None) == f"{__version__}+unknown"
    assert C.pipeline_version(env={"GIT_COMMIT": "unknown"}, git=lambda: None) == f"{__version__}+unknown"
    assert C.pipeline_version(env={"GIT_COMMIT": "not-a-sha"}, git=lambda: None) == f"{__version__}+unknown"


def test_pipeline_version_prefers_git_commit_env():
    def boom():
        raise AssertionError("git must not be consulted when GIT_COMMIT is set")

    assert C.pipeline_version(env={"GIT_COMMIT": "DEADBEEF"}, git=boom) == f"{__version__}+gdeadbeef"
    assert C.pipeline_version(env={"GIT_COMMIT": "61e588a.dirty"}, git=boom) == f"{__version__}+g61e588a.dirty"


def test_pipeline_version_format_real_environment():
    v = C.pipeline_version()
    assert re.fullmatch(rf"{re.escape(__version__)}\+(g[0-9a-f]{{7,40}}(\.dirty)?|unknown)", v)


# ------------------------------------------------------------------ attempt configs (Task 4)


def test_default_attempts_parse(default_config_path, project_root):
    conf = C.load(default_config_path, root=project_root)
    a1, a2, a3 = conf.attempts
    assert (a1.name, a1.segmentation, a1.orientation, a1.frequency, a1.enhancement) == (
        "traditional", "GMFS", "GBFOE", "XSFFE", "GBFEN")
    assert (a2.name, a2.segmentation, a2.orientation, a2.frequency, a2.enhancement) == (
        "deep", "SUFS", "SNFOE", "SNFFE", "SNFEN")
    assert a1.clahe is None and a1.upscale == 1.0 and dict(a1.sbmex_overrides) == {}
    # Task 5: every attempt resamples 3x and runs pyfing at 500 dpi.
    assert all(a.resample == 3.0 and a.dpi == 500 and a.scale_factor == 3.0 for a in (a1, a2, a3))
    assert a3.clahe == C.ClaheConfig(2.0, 8) and a3.upscale == 1.0
    assert dict(a3.sbmex_overrides) == {"minimum_background_distance": 4}
    assert conf.attempt(2) is a2
    ids = {conf.configuration_id(a) for a in conf.attempts}
    assert len(ids) == 3 and all(re.fullmatch(r"cfg_[0-9a-f]{12}", i) for i in ids)
    assert conf.configuration_id(3) == C.configuration_id(a3, conf.adapter_version, conf.tools)


def test_attempt_config_is_frozen(raw):
    a = C.validate(raw).attempts[0]
    with pytest.raises(AttributeError):
        a.upscale = 2.0  # type: ignore[misc]
    with pytest.raises(TypeError):
        C.validate(raw).attempts[2].sbmex_overrides["x"] = 1  # type: ignore[index]


def test_attempt_lookup_rejects_undefined(raw):
    conf = C.validate(raw)
    for bad in (0, 4, -1):
        with pytest.raises(KeyError):
            conf.attempt(bad)


@pytest.mark.parametrize("field", ["segmentation", "orientation", "frequency", "enhancement"])
def test_parsing_rejects_unknown_method(raw, field):
    raw["preprocess"]["attempts"][1][field] = "LEADER"
    with pytest.raises(C.ConfigError, match=rf"unknown {field} method 'LEADER'"):
        C.validate(raw)


@pytest.mark.parametrize("numbers", [[1, 3, 2], [0, 1, 2], [2, 3, 4], [1, 1, 2], [1, 2, 4]])
def test_parsing_rejects_bad_attempt_no_sequence(raw, numbers):
    for a, n in zip(raw["preprocess"]["attempts"], numbers):
        a["attempt_no"] = n
    with pytest.raises(C.ConfigError, match="attempt_no"):
        C.validate(raw)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda a: a.update(upscale=1.5), "only support 500 dpi"),       # attempt 1 is XSFFE/GBFEN
        (lambda a: a.update(upscale=0.5), "upscale"),
        (lambda a: a.update(resample=0.5), "resample"),
        (lambda a: a.update(resample=4.5), "resample"),
        (lambda a: a.update(resample=True), "resample"),
        (lambda a: a.update(colour="red"), "unknown keys"),
        (lambda a: a.pop("enhancement"), "missing keys"),
        (lambda a: a.update(clahe={"clip_limit": 0, "tile": 8}), "clip_limit"),
        (lambda a: a.update(clahe={"clip_limit": 2.0, "tile": 0}), "tile"),
        (lambda a: a.update(sbmex_overrides={"x": [1]}), "sbmex_overrides"),
    ],
)
def test_parsing_rejects_invalid_attempt_fields(raw, mutate, match):
    mutate(raw["preprocess"]["attempts"][0])
    with pytest.raises(C.ConfigError, match=match):
        C.validate(raw)


def test_parsing_rejects_empty_or_too_few_attempts(raw):
    bad = copy.deepcopy(raw)
    bad["preprocess"]["attempts"] = []
    with pytest.raises(C.ConfigError, match="non-empty"):
        C.validate(bad)
    bad = copy.deepcopy(raw)
    bad["preprocess"]["attempts"] = bad["preprocess"]["attempts"][:2]
    with pytest.raises(C.ConfigError, match="max_attempts=3 exceeds"):
        C.validate(bad)
    bad["preprocess"]["max_attempts"] = 2
    assert len(C.validate(bad).attempts) == 2


def test_configuration_id_ignores_equivalent_spellings(raw):
    base = C.validate(raw)
    a = raw["preprocess"]["attempts"]
    a[0]["upscale"] = 1          # explicit default, int spelling
    a[0]["clahe"] = None
    a[2]["resample"] = 3         # int instead of float
    a[2] = dict(reversed(list(a[2].items())))  # key order
    other = C.validate(raw)
    assert [base.configuration_id(x) for x in base.attempts] == [other.configuration_id(x) for x in other.attempts]


# One alternative value per AttemptConfig field (each still valid for attempt 3).
_ATTEMPT_EDITS = {
    "attempt_no": 4,
    "name": "deep_boost2",
    "segmentation": "GMFS",
    "orientation": "GBFOE",
    "frequency": "XSFFE",
    "enhancement": "GBFEN",
    "clahe": None,
    "upscale": 2.0,
    "resample": 2.0,
    "sbmex_overrides": {"minimum_background_distance": 5},
}


def test_configuration_id_sensitivity_per_field(raw):
    conf = C.validate(raw)
    base_attempt = conf.attempts[2]
    base = conf.configuration_id(base_attempt)
    assert base == C.configuration_id(C.AttemptConfig.from_dict(base_attempt.to_dict()),
                                      conf.adapter_version, dict(conf.tools))
    seen = {base}
    for name, value in _ATTEMPT_EDITS.items():
        d = base_attempt.to_dict()
        d[name] = value
        if name in ("frequency", "enhancement"):
            d["upscale"] = 1.0  # XSFFE/GBFEN are 500 dpi only
        cid = C.configuration_id(C.AttemptConfig.from_dict(d), conf.adapter_version, conf.tools)
        assert cid != base, f"changing {name} did not change configuration_id"
        seen.add(cid)
    for key in ("clip_limit", "tile"):
        d = base_attempt.to_dict()
        d["clahe"] = {**d["clahe"], key: d["clahe"][key] + 1}
        seen.add(C.configuration_id(C.AttemptConfig.from_dict(d), conf.adapter_version, conf.tools))
    # adapter version and each pinned tool version
    seen.add(C.configuration_id(base_attempt, "xyt-adapter/2", conf.tools))
    for tool in conf.tools:
        seen.add(C.configuration_id(base_attempt, conf.adapter_version, {**conf.tools, tool: conf.tools[tool] + ".1"}))
    assert len(seen) == 1 + len(_ATTEMPT_EDITS) + 2 + 1 + len(conf.tools)


def test_attempt_to_dict_round_trips(raw):
    for a in C.validate(raw).attempts:
        assert C.AttemptConfig.from_dict(a.to_dict()) == a
        C.canonical_json(a.to_dict())


def test_resample_and_upscale_combine_into_scale_factor(raw):
    d = C.validate(raw).attempts[1].to_dict()
    d.update(resample=2.0, upscale=1.5)
    a = C.AttemptConfig.from_dict(d)
    assert a.scale_factor == 3.0 and a.dpi == 750       # pyfing only sees the upscale
    traditional = C.validate(raw).attempts[0]
    assert traditional.resample == 3.0 and traditional.dpi == 500   # resample is fine for XSFFE/GBFEN
