"""`fpv` command-line interface.

Task 1 provides the global `--config` flag and `fpv doctor`. Later tasks register their
diagnostic subcommands (catalog, quality, preprocess, extract, match, ...). `enroll` / `verify`
(Task 10) own the interactive session loop; the controller itself never prompts or prints.

Exit codes: 0 ENROLLED/MATCH, 10 NO_MATCH, 11 NOT_ENROLLED, 20 REJECTED_QUALITY,
21 REJECTED_MINUTIAE, 22 presentation limit reached, 2 invalid input / cancelled, 1 ERROR.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.metadata as md
import os
import re
import shutil
import subprocess
import sys
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from fpv import __version__
from fpv import config as cfgmod

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2

NBIS_VERSION_FILE = Path(os.environ.get("FPV_NBIS_VERSION_FILE", "/usr/local/share/nbis/VERSION"))
_NFIQ2_VERSION_RE = re.compile(r"NFIQ 2:\s*([0-9][0-9.]*)")
CONFLICTING_OPENCV = ("opencv-python-headless", "opencv-python", "opencv-contrib-python")

Output = Callable[[str], None]


# --------------------------------------------------------------------------- doctor


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    ok: bool
    detail: str


def _pin_check(name: str, found: str | None, pinned: str | None, extra: str = "") -> Check:
    if found is None:
        return Check(name, False, f"not available{extra}")
    if pinned is not None and found != pinned:
        return Check(name, False, f"{found} (pinned {pinned}){extra}")
    return Check(name, True, f"{found}{extra}")


def nfiq2_version(nfiq2_bin: str = "nfiq2", timeout_s: float = 30) -> str | None:
    """Parse the NFIQ 2 version from the CLI banner (`nfiq2 -h` prints version info)."""
    if shutil.which(nfiq2_bin) is None:
        return None
    try:
        proc = subprocess.run([nfiq2_bin, "-h"], capture_output=True, text=True,
                              stdin=subprocess.DEVNULL, timeout=timeout_s)
    except (OSError, subprocess.SubprocessError):
        return None
    m = _NFIQ2_VERSION_RE.search(proc.stdout + proc.stderr)
    return m.group(1) if m else None


def nbis_version(version_file: Path = NBIS_VERSION_FILE) -> str | None:
    """NBIS binaries have no --version flag; the image records the built release in a file."""
    try:
        first = version_file.read_text(encoding="utf-8").split()
    except OSError:
        return None
    return first[0] if first else None


def bozorth3_usable(bozorth3_bin: str = "bozorth3", timeout_s: float = 30) -> tuple[bool, str]:
    path = shutil.which(bozorth3_bin)
    if path is None:
        return False, "binary not found on PATH"
    try:
        proc = subprocess.run([bozorth3_bin, "-h"], capture_output=True, text=True,
                              stdin=subprocess.DEVNULL, timeout=timeout_s)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"failed to run: {type(exc).__name__}"
    if "Usage" not in proc.stdout + proc.stderr:
        return False, "usage text not printed"
    return True, path


def _dist_version(dist: str) -> str | None:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


def doctor_checks(conf: cfgmod.PipelineConfig) -> list[Check]:
    pins = conf.tools
    checks: list[Check] = []

    checks.append(Check("config", True, f"{conf.source_path} ({conf.config_set_id})"))
    checks.append(Check("pipeline_version", True, cfgmod.pipeline_version()))

    nfiq2_path = shutil.which("nfiq2")
    checks.append(_pin_check("nfiq2", nfiq2_version(), pins.get("nfiq2"),
                             f" [{nfiq2_path}]" if nfiq2_path else ""))

    ok, detail = bozorth3_usable()
    nbis = nbis_version()
    if not ok:
        checks.append(Check("nbis/bozorth3", False, detail))
    else:
        checks.append(_pin_check("nbis/bozorth3", nbis, pins.get("nbis"), f" [{detail}]"))

    checks.append(_pin_check("pyfing", _dist_version("pyfing"), pins.get("pyfing")))
    checks.append(_pin_check("tensorflow", _dist_version("tensorflow"), pins.get("tensorflow")))
    checks.append(_pin_check("keras", _dist_version("keras"), pins.get("keras")))

    try:
        cv = importlib.import_module("cv2")
        cv_version: str | None = getattr(cv, "__version__", None)
        has_ximgproc = hasattr(cv, "ximgproc") and hasattr(cv.ximgproc, "thinning")
    except Exception:  # noqa: BLE001 - any import failure is a failed check
        cv_version, has_ximgproc = None, False
    checks.append(_pin_check("opencv", cv_version, pins.get("opencv"),
                             f" (opencv-contrib-python-headless {_dist_version('opencv-contrib-python-headless')})"))
    checks.append(Check("cv.ximgproc", has_ximgproc, "available" if has_ximgproc else "missing"))

    conflicts = [d for d in CONFLICTING_OPENCV if _dist_version(d) is not None]
    checks.append(Check("opencv conflicts", not conflicts,
                        "none" if not conflicts else f"also installed: {', '.join(conflicts)}"))

    try:
        importlib.import_module("pyfing")
        checks.append(Check("import pyfing", True, "ok"))
    except Exception as exc:  # noqa: BLE001
        checks.append(Check("import pyfing", False, f"{type(exc).__name__}: {exc}"))

    return checks


def cmd_doctor(args: argparse.Namespace, conf: cfgmod.PipelineConfig, out: Output) -> int:
    checks = doctor_checks(conf)
    width = max(len(c.name) for c in checks)
    for c in checks:
        out(f"{c.name:<{width}}  {'OK  ' if c.ok else 'FAIL'}  {c.detail}")
    failed = [c.name for c in checks if not c.ok]
    out(f"doctor: {'all checks passed' if not failed else 'FAILED: ' + ', '.join(failed)}")
    return EXIT_OK if not failed else EXIT_ERROR


# --------------------------------------------------------------------------- catalog


class CliInputError(Exception):
    """Invalid command-line input; reported on stderr with exit code EXIT_USAGE."""


def cmd_catalog(args: argparse.Namespace, conf: cfgmod.PipelineConfig, out: Output) -> int:
    from fpv.catalog import Catalog  # imported lazily: needs OpenCV, `doctor` must not
    from fpv.imaging import InputImageError

    path = Path(args.path)
    if not path.is_file():
        raise CliInputError(f"no such file: {path}")
    try:
        catalog = Catalog.load(conf.path("splits"), conf.path("exclusions"), conf.path("raw"))
    except (OSError, ValueError) as exc:
        out(f"catalog: cannot load split/exclusion CSVs: {exc}")
        return EXIT_ERROR
    try:
        rec = catalog.record(path)
    except InputImageError as exc:
        raise CliInputError(str(exc)) from exc

    rows: list[tuple[str, str]] = [
        ("path", rec.source_path),
        ("sample_id", rec.sample_id),
        ("sha256", rec.sha256),
        ("size", f"{rec.width}x{rec.height}"),
    ]
    lab = rec.label
    if lab is None:
        rows.append(("labeled", "no (filename/directory does not match SOCOFing; no ground truth)"))
    else:
        rows += [
            ("labeled", "yes"),
            ("subject", str(lab.identity.user_id)),
            ("gender", lab.gender),
            ("hand", lab.identity.hand.value),
            ("finger", lab.identity.finger.value),
            ("alteration", lab.alteration),
            ("difficulty", lab.difficulty),
        ]
    rows += [("split", rec.split or "none"), ("excluded", "yes" if rec.excluded else "no")]
    width = max(len(k) for k, _ in rows)
    for key, value in rows:
        out(f"{key:<{width}}  {value}")
    return EXIT_OK


# --------------------------------------------------------------------------- quality

EXIT_REJECTED_QUALITY = 20


def cmd_quality(args: argparse.Namespace, conf: cfgmod.PipelineConfig, out: Output) -> int:
    from fpv.imaging import InputImageError, load_gray8
    from fpv.quality import QualitySettings, QualityToolError, assess_quality, quality_gate

    path = Path(args.image)
    try:
        img = load_gray8(path)
    except InputImageError as exc:
        raise CliInputError(str(exc)) from exc
    settings = QualitySettings.from_config(conf)
    try:
        result = assess_quality(img, settings)
    except QualityToolError as exc:
        out(f"quality: ERROR (nfiq2_tool_error): {exc}")
        return EXIT_ERROR
    decision = quality_gate(result, conf.thresholds.nfiq2_min)

    h, w = img.shape
    sh, sw = result.scored_shape
    out(f"image     {path} ({w}x{h}, scored at {sw}x{sh}, upscale {result.upscale:g}x)")
    out(f"score     {result.score if result.score is not None else 'NA'}")
    if result.error:
        out(f"error     {result.error}")
    if result.actionable_feedback:
        out("feedback")
        width = max(len(k) for k in result.actionable_feedback)
        for name, value in result.actionable_feedback.items():
            out(f"  {name:<{width}}  {value:g}")
    else:
        out("feedback  none")
    verdict = "PASS" if decision.passed else f"FAIL ({decision.reason})"
    out(f"gate      {verdict} (nfiq2_min={decision.threshold})")
    return EXIT_OK if decision.passed else EXIT_REJECTED_QUALITY


# --------------------------------------------------------------------------- preprocess


def cmd_preprocess(args: argparse.Namespace, conf: cfgmod.PipelineConfig, out: Output) -> int:
    from fpv.imaging import InputImageError, load_gray8, png_bytes
    from fpv.preprocess import preprocess, quality_after
    from fpv.quality import QualitySettings, QualityToolError

    path = Path(args.image)
    try:
        img = load_gray8(path)
    except InputImageError as exc:
        raise CliInputError(str(exc)) from exc
    try:
        attempt = conf.attempt(args.attempt)
    except KeyError as exc:
        raise CliInputError(exc.args[0]) from exc
    out_path = Path(args.out)
    if not out_path.parent.is_dir():
        raise CliInputError(f"output directory does not exist: {out_path.parent}")

    cid = conf.configuration_id(attempt)
    pre = preprocess(img, attempt, configuration_id=cid)
    out_path.write_bytes(png_bytes(pre.enhanced, ppi=int(round(attempt.dpi))))

    h, w = img.shape
    eh, ew = pre.enhanced.shape
    timings = ", ".join(f"{k} {v:.1f}" for k, v in pre.timings_ms.items())
    out(f"image             {path} ({w}x{h})")
    out(f"attempt           {attempt.attempt_no} ({attempt.name}: {attempt.segmentation}, "
        f"{attempt.orientation}, {attempt.frequency}, {attempt.enhancement}"
        f"{', CLAHE' if attempt.clahe else ''}, resample {attempt.resample:g}x, "
        f"upscale {attempt.upscale:g}x, dpi {attempt.dpi})")
    out(f"configuration_id  {cid}")
    out(f"enhanced          {out_path} ({ew}x{eh}, foreground {(pre.mask > 0).mean():.0%})")
    out(f"timings_ms        {timings}")
    try:
        q = quality_after(pre, QualitySettings.from_config(conf))
    except QualityToolError as exc:
        out(f"quality_after     ERROR (nfiq2_tool_error): {exc}")
        return EXIT_ERROR
    score = q.score if q.score is not None else "NA (null)"
    sh, sw = q.scored_shape
    out(f"quality_after     {score} (scored at {sw}x{sh}, nfiq2 upscale {q.upscale:g}x)")
    if q.error:
        out(f"quality_error     {q.error}")
    return EXIT_OK


# --------------------------------------------------------------------------- extract

EXIT_REJECTED_MINUTIAE = 21


def cmd_extract(args: argparse.Namespace, conf: cfgmod.PipelineConfig, out: Output) -> int:
    """Full attempt policy (no raw NFIQ 2 gate) and the adapted `.xyt` template (Req 6.6)."""
    from fpv.acquire import Acquired, acquire_template
    from fpv.adapter import write_xyt
    from fpv.imaging import InputImageError, load_gray8
    from fpv.quality import QualityToolError

    path = Path(args.image)
    try:
        img = load_gray8(path)
    except InputImageError as exc:
        raise CliInputError(str(exc)) from exc
    out_path = Path(args.out)
    if not out_path.parent.is_dir():
        raise CliInputError(f"output directory does not exist: {out_path.parent}")

    try:
        result = acquire_template(img, conf)
    except QualityToolError as exc:
        out(f"extract: ERROR (nfiq2_tool_error): {exc}")
        return EXIT_ERROR

    h, w = img.shape
    out(f"image             {path} ({w}x{h})")
    for r in result.attempts:
        qa = r.quality_after if r.quality_after is not None else "NA"
        if r.error is not None:
            out(f"attempt {r.attempt_no}         {r.configuration_id}  error: {r.error}")
        else:
            out(f"attempt {r.attempt_no}         {r.configuration_id}  quality_after {qa}, "
                f"minutiae {r.minutiae_count} (raw {r.raw_count})")
    minimum = conf.thresholds.minutiae_min
    if not isinstance(result, Acquired):
        out(f"result            {result.status.value} ({result.reason}; minutiae_min={minimum}); "
            f"no template written")
        return EXIT_REJECTED_MINUTIAE if result.status.value == "REJECTED_MINUTIAE" else EXIT_ERROR

    write_xyt(result.template, out_path)
    out(f"configuration_id  {result.configuration_id} (attempt {result.attempt_no}, "
        f"scale_factor {result.scale_factor:g})")
    out(f"minutiae          {result.template.count} (minutiae_min={minimum})")
    out(f"template          {out_path} ({result.template.adapter_version})")
    return EXIT_OK


# --------------------------------------------------------------------------- match

EXIT_NO_MATCH = 10


def cmd_match(args: argparse.Namespace, conf: cfgmod.PipelineConfig, out: Output) -> int:
    """Bozorth3 score of two `.xyt` files and the decision against `bozorth3_match` (Req 9.11)."""
    from fpv.adapter import XytParseError
    from fpv.matcher import InsufficientMinutiae, MatcherError, decide, match
    from fpv.types import Status

    probe, gallery = Path(args.probe), Path(args.gallery)
    for p in (probe, gallery):
        if not p.is_file():
            raise CliInputError(f"no such file: {p}")
    threshold = conf.thresholds.bozorth3_match
    try:
        result = match(probe, gallery, tool_version=conf.tools.get("nbis", "unknown"))
    except (XytParseError, OSError) as exc:
        raise CliInputError(f"cannot read template: {exc}") from exc
    except InsufficientMinutiae as exc:
        status = Status.REJECTED_MINUTIAE if exc.which == "probe" else Status.ERROR
        out(f"match: {status.value} ({exc.which}_below_bozorth_min): {exc}")
        return EXIT_REJECTED_MINUTIAE if exc.which == "probe" else EXIT_ERROR
    except MatcherError as exc:
        out(f"match: ERROR (bozorth3_tool_error): {exc}")
        return EXIT_ERROR

    decision = decide(result.score, threshold)
    out(f"probe     {probe} ({result.probe_count} minutiae)")
    out(f"gallery   {gallery} ({result.gallery_count} minutiae)")
    out(f"score     {result.score}")
    out(f"decision  {decision.value} (bozorth3_match={threshold})")
    return EXIT_OK if decision is Status.MATCH else EXIT_NO_MATCH


# --------------------------------------------------------------------------- db (hidden)


def cmd_db_templates(args: argparse.Namespace, conf: cfgmod.PipelineConfig, out: Output) -> int:
    """List every template version of one identity (hidden diagnostic, Task 8)."""
    from fpv.storage import Repository, format_templates
    from fpv.types import Identity

    try:
        identity = Identity.parse(args.user, args.hand, args.finger)
    except ValueError as exc:
        raise CliInputError(str(exc)) from exc
    db = Path(args.db) if args.db else conf.path("db")
    if not db.is_file():
        raise CliInputError(f"no database at {db}")
    with Repository(db, conf.path("templates"), conf.path("probes"), artifact_root=conf.root) as repo:
        repo.migrate()
        rows = repo.list_templates(identity)
    out(f"templates for {identity} in {db}")
    if not rows:
        out("none")
        return EXIT_OK
    for line in format_templates(rows):
        out(line)
    return EXIT_OK


# --------------------------------------------------------------------------- enroll / verify

EXIT_NOT_ENROLLED = 11
EXIT_LIMIT_REACHED = 22

# Design exit-code table (Req 3.7). Invalid non-interactive input is EXIT_USAGE (2).
STATUS_EXIT_CODES: dict[str, int] = {
    "ENROLLED": EXIT_OK,
    "MATCH": EXIT_OK,
    "NO_MATCH": EXIT_NO_MATCH,
    "NOT_ENROLLED": EXIT_NOT_ENROLLED,
    "REJECTED_QUALITY": EXIT_REJECTED_QUALITY,
    "REJECTED_MINUTIAE": EXIT_REJECTED_MINUTIAE,
    "ERROR": EXIT_ERROR,
}
REJECTED_STATUSES = frozenset({"REJECTED_QUALITY", "REJECTED_MINUTIAE"})

InputFn = Callable[[str], str]


class SessionController(Protocol):
    """The part of `PipelineController` the session loop uses (fakes implement this in tests)."""

    def enroll(self, sample_path: Path, identity: Any, *, session_id: str, presentation_no: int) -> Any: ...
    def verify(self, sample_path: Path, identity: Any, *, session_id: str, presentation_no: int) -> Any: ...


ControllerFactory = Callable[[cfgmod.PipelineConfig], AbstractContextManager[SessionController]]


@contextlib.contextmanager
def default_controller(conf: cfgmod.PipelineConfig) -> Iterator[SessionController]:
    """The real controller over the configured catalog, SQLite DB and artifact store."""
    from fpv.catalog import Catalog
    from fpv.pipeline import PipelineController
    from fpv.storage import Repository

    catalog = Catalog.load(conf.path("splits"), conf.path("exclusions"), conf.path("raw"))
    with Repository.from_config(conf) as repo:
        yield PipelineController(conf, repo, catalog)


class _Aborted(Exception):
    """The user ended input (EOF / Ctrl-C) at a prompt."""


def _ask(input_fn: InputFn, prompt: str) -> str:
    try:
        return input_fn(prompt)
    except (EOFError, KeyboardInterrupt) as exc:
        raise _Aborted from exc


def _clean_path(text: str) -> Path:
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
        text = text[1:-1]
    return Path(text).expanduser()


def check_image(path: Path) -> str | None:
    """None when `path` is a decodable image, else the reason it cannot be used (Req 3.3)."""
    from fpv.imaging import InputImageError, load_gray8

    if not path.is_file():
        return f"no such file: {path}"
    try:
        load_gray8(path)
    except InputImageError as exc:
        return f"cannot decode image: {exc}"
    return None


def _prompt_image(input_fn: InputFn, out: Output, prompt: str = "Image path: ") -> Path:
    while True:
        text = _ask(input_fn, prompt)
        if not text.strip():
            out("Please enter the path of a fingerprint image.")
            continue
        path = _clean_path(text)
        problem = check_image(path)
        if problem is None:
            return path
        out(f"{problem}; please enter another image path.")


def _prompt_value(input_fn: InputFn, out: Output, prompt: str, parse: Callable[[str], Any]) -> Any:
    while True:
        text = _ask(input_fn, prompt)
        try:
            return parse(text)
        except ValueError as exc:
            out(f"{exc}")


def _gather_claim(args: argparse.Namespace, interactive: bool, input_fn: InputFn,
                  out: Output) -> tuple[Path, Any]:
    """Image path and identity from flags and/or prompts (Req 3.1-3.3, 3.5, 3.6).

    Non-interactive: any invalid value raises CliInputError (exit 2, nothing processed).
    Interactive: supplied valid flags are used as is; missing or invalid ones are prompted for.
    """
    from fpv.types import Finger, Hand, Identity, parse_finger, parse_hand, parse_user_id

    hands = ", ".join(h.value for h in Hand)
    fingers = ", ".join(f.value for f in Finger)

    image: Path | None = None
    if args.image is not None:
        candidate = _clean_path(args.image)
        problem = check_image(candidate)
        if problem is None:
            image = candidate
        elif not interactive:
            raise CliInputError(problem)
        else:
            out(f"{problem}")
    if image is None:
        image = _prompt_image(input_fn, out)

    def field(value: str | None, parse: Callable[[str], Any], prompt: str) -> Any:
        if value is not None:
            try:
                return parse(value)
            except ValueError as exc:
                if not interactive:
                    raise CliInputError(str(exc)) from exc
                out(f"{exc}")
        return _prompt_value(input_fn, out, prompt, parse)

    user = field(args.user, parse_user_id, "User ID (1-600): ")
    hand = field(args.hand, parse_hand, f"Hand ({hands}): ")
    finger = field(args.finger, parse_finger, f"Finger ({fingers}): ")
    return image, Identity(user, hand, finger)


def _report(outcome: Any, out: Output) -> None:
    status = outcome.status.value
    out(f"run {outcome.run_id}: {status}" + (f" ({outcome.reason})" if outcome.reason else ""))
    if getattr(outcome, "score", None) is not None:
        out(f"score {outcome.score} (bozorth3_match={outcome.threshold})")
    out(outcome.message)


def run_session(operation: str, args: argparse.Namespace, conf: cfgmod.PipelineConfig, out: Output, *,
                input_fn: InputFn, controller_factory: ControllerFactory) -> int:
    """One CLI session: claim, confirmation, then up to `max_presentations` runs (Req 3, 10)."""
    interactive = not (args.yes and None not in (args.image, args.user, args.hand, args.finger))
    try:
        image, identity = _gather_claim(args, interactive, input_fn, out)
        out(f"Claim: {operation} {identity} with image {image}")
        if not args.yes:
            answer = _ask(input_fn, "Proceed? [y/N]: ")
            if answer.strip().lower() not in ("y", "yes"):
                out("Cancelled; no sample was processed.")
                return EXIT_USAGE
    except _Aborted:
        out("Cancelled; no sample was processed.")
        return EXIT_USAGE

    limit = conf.max_presentations
    session_id = str(uuid.uuid4())
    with contextlib.ExitStack() as stack:
        try:
            controller = stack.enter_context(controller_factory(conf))
        except (OSError, ValueError) as exc:
            out(f"{operation}: cannot open the catalog or store: {exc}")
            return EXIT_ERROR
        run = controller.enroll if operation == "enroll" else controller.verify
        out(f"session {session_id}")
        presentation = 1
        while True:
            out(f"Presentation {presentation} of {limit}: {image}")
            outcome = run(image, identity, session_id=session_id, presentation_no=presentation)
            _report(outcome, out)
            status = outcome.status.value
            if status not in REJECTED_STATUSES:  # terminal status ends the session (Req 10.4)
                return STATUS_EXIT_CODES[status]
            if presentation >= limit:  # Req 10.3
                out(f"Presentation limit reached ({limit} of {limit}); session ended.")
                return EXIT_LIMIT_REACHED
            if not interactive:  # no prompts in non-interactive mode: report the rejection
                return STATUS_EXIT_CODES[status]
            presentation += 1
            out(f"Please present a new sample for {identity} (presentation {presentation} of {limit}).")
            try:
                image = _prompt_image(input_fn, out, "New image path: ")
            except _Aborted:
                out("Session ended by the user.")
                return STATUS_EXIT_CODES[status]


def _session_command(operation: str) -> Callable[..., int]:
    def cmd(args: argparse.Namespace, conf: cfgmod.PipelineConfig, out: Output) -> int:
        return run_session(operation, args, conf, out, input_fn=args.input_fn,
                           controller_factory=args.controller_factory)

    cmd.__name__ = f"cmd_{operation}"
    return cmd


cmd_enroll = _session_command("enroll")
cmd_verify = _session_command("verify")


def _add_claim_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--image", default=None, help="sample image path (prompted for when omitted)")
    p.add_argument("--user", default=None, help="claimed user ID, 1-600")
    p.add_argument("--hand", default=None, help="Left or Right")
    p.add_argument("--finger", default=None, help="thumb, index, middle, ring or little")
    p.add_argument("--yes", action="store_true",
                   help="skip the confirmation; with all claim flags, run without any prompt")


# --------------------------------------------------------------------------- evaluate


def check_test_split(conf: cfgmod.PipelineConfig, args: argparse.Namespace) -> dict[str, Any] | None:
    """Guard for `fpv evaluate --split test` (Req 14.6).

    Requires a frozen configuration, and refuses when `test_report_<config_set_id>.json` already
    exists unless `--force` is given. Returns the report a forced run replaces (else None).
    """
    from fpv.calibrate import TestSplitGuardError, guard_test_split

    try:
        return guard_test_split(conf, force=bool(getattr(args, "force", False)))
    except TestSplitGuardError as exc:
        raise CliInputError(str(exc)) from exc


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid integer {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def cmd_evaluate(args: argparse.Namespace, conf: cfgmod.PipelineConfig, out: Output) -> int:
    """Batch evaluation of one split through the pipeline controller (Req 13)."""
    from fpv.catalog import Catalog
    from fpv.evaluate import evaluate

    test_split = args.split == "test"
    previous = check_test_split(conf, args) if test_split else None
    if test_split and args.subjects is not None:
        raise CliInputError("--subjects is not allowed for the test split (the baseline covers every subject)")
    try:
        catalog = Catalog.load(conf.path("splits"), conf.path("exclusions"), conf.path("raw"))
    except (OSError, ValueError) as exc:
        out(f"evaluate: cannot load split/exclusion CSVs: {exc}")
        return EXIT_ERROR
    version = cfgmod.pipeline_version()
    report = evaluate(conf, catalog, args.split, seed=args.seed, subjects=args.subjects, workers=args.workers,
                      pipeline_version=version, progress=out)
    m = report.overall

    def pct(x: float) -> str:
        return "n/a" if x != x else f"{x * 100:.2f}%"

    out(f"FTE {pct(report.fte)}  FTA {pct(m.fta)}  FMR {pct(m.fmr)} ({m.impostor} impostor)  "
        f"FNMR {pct(m.fnmr)} ({m.genuine} genuine)  gFMR {pct(m.gfmr)}  gFNMR {pct(m.gfnmr)}  "
        f"at bozorth3_match={conf.thresholds.bozorth3_match}")
    out(f"wrote {report.out_dir / 'scores.csv'}, summary.md, det.png")
    if test_split:  # Req 14.6, 14.7
        from fpv.calibrate import write_baseline, write_test_report

        forced = previous is not None
        rp = write_test_report(conf, report, forced=forced, previous=previous, pipeline_version=version)
        bp = write_baseline(conf, report, pipeline_version=version, forced=forced, report_path=rp)
        out(f"wrote {rp}" + (" (forced: override recorded)" if forced else ""))
        out(f"wrote {bp}")
    return EXIT_OK


# --------------------------------------------------------------------------- calibrate


def cmd_calibrate(args: argparse.Namespace, conf: cfgmod.PipelineConfig, out: Output) -> int:
    """Calibrate thresholds on train + validation scores and freeze the config (Req 14.1-14.5)."""
    from fpv.calibrate import CalibrationInputError, run_calibration

    try:
        res, frozen, report = run_calibration(
            conf, out_config=Path(args.out) if args.out else None, primary_target_fmr=args.primary_fmr,
            nfiq2_min=args.nfiq2_min, minutiae_min=args.minutiae_min, force=args.force)
    except CalibrationInputError as exc:
        raise CliInputError(str(exc)) from exc
    for op in res.pooled:
        flag = " LOW CONFIDENCE" if op.low_confidence else ""
        primary = " (primary)" if op.target_fmr == res.primary_target_fmr else ""
        out(f"target FMR {op.target_fmr:g}{primary}: threshold {op.threshold}, FMR {op.fmr * 100:.3f}% "
            f"({op.false_matches}/{op.impostor}), FNMR {op.fnmr * 100:.2f}% "
            f"({op.false_non_matches}/{op.genuine}){flag}")
    out(f"wrote {report} and calibration.json")
    out(f"wrote {frozen.source_path} (frozen, config_set_id {frozen.config_set_id}; "
        f"bozorth3_match {frozen.thresholds.bozorth3_match}, nfiq2_min {frozen.thresholds.nfiq2_min}, "
        f"minutiae_min {frozen.thresholds.minutiae_min})")
    return EXIT_OK


# --------------------------------------------------------------------------- parser / main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fpv", description="SOCOFing fingerprint pipeline (phase 1)")
    parser.add_argument("--version", action="version", version=f"fpv {__version__}")
    parser.add_argument(
        "--config",
        default=None,
        help="pipeline configuration YAML (default: configs/pipeline.v1.frozen.yaml if present, "
        "else configs/pipeline.v1.yaml)",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    p = sub.add_parser("doctor", help="report tool/library versions and check the runtime")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("catalog", help="print a sample's SOCOFing labels, split and exclusion flag")
    p.add_argument("path", help="sample image path")
    p.set_defaults(func=cmd_catalog)

    p = sub.add_parser("quality", help="score an image with NFIQ 2 and apply the nfiq2_min gate")
    p.add_argument("image", help="sample image path")
    p.set_defaults(func=cmd_quality)

    p = sub.add_parser("preprocess", help="run one preprocessing attempt and score quality_after")
    p.add_argument("image", help="sample image path")
    p.add_argument("--attempt", type=int, required=True, help="attempt number (1..n from the config)")
    p.add_argument("--out", required=True, help="output PNG for the enhanced image")
    p.set_defaults(func=cmd_preprocess)

    p = sub.add_parser("extract", help="run the attempt policy and write the adapted .xyt template")
    p.add_argument("image", help="sample image path")
    p.add_argument("--out", required=True, help="output .xyt template path")
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("match", help="Bozorth3 score of two .xyt templates and the match decision")
    p.add_argument("probe", help="probe .xyt template")
    p.add_argument("gallery", help="gallery (reference) .xyt template")
    p.set_defaults(func=cmd_match)

    p = sub.add_parser("enroll", help="enroll a finger (interactive, or non-interactive with all flags and --yes)")
    _add_claim_args(p)
    p.set_defaults(func=cmd_enroll)

    p = sub.add_parser("verify", help="verify a claimed finger (interactive, or non-interactive with all flags "
                                      "and --yes)")
    _add_claim_args(p)
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("evaluate", help="batch genuine/impostor evaluation of one split; writes scores.csv, "
                                        "summary.md and det.png under <reports>/eval/<split>/<config_set_id>/")
    p.add_argument("--split", required=True, choices=("train", "validation", "test"))
    p.add_argument("--subjects", type=_positive_int, default=None,
                   help="evaluate a seeded subset of N subjects (default: all)")
    p.add_argument("--workers", type=_positive_int, default=min(4, os.cpu_count() or 1),
                   help="worker processes (default: min(4, CPUs))")
    p.add_argument("--seed", type=int, default=None, help="default: evaluation.seed from the config")
    p.add_argument("--force", action="store_true",
                   help="test split only: run again although test_report_<config_set_id>.json exists "
                        "(recorded in the report)")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("calibrate", help="choose thresholds from train + validation scores; writes the calibration "
                                         "report and configs/pipeline.v1.frozen.yaml")
    p.add_argument("--primary-fmr", type=float, default=None,
                   help="primary target FMR (default: calibration.primary_target_fmr)")
    p.add_argument("--nfiq2-min", type=int, default=None, help="frozen nfiq2_min (default: unchanged)")
    p.add_argument("--minutiae-min", type=int, default=None, help="frozen minutiae_min (default: unchanged)")
    p.add_argument("--out", default=None, help="frozen config path (default: configs/pipeline.v1.frozen.yaml)")
    p.add_argument("--force", action="store_true", help="replace an existing frozen config")
    p.set_defaults(func=cmd_calibrate)

    # Hidden: no `help=` keeps it out of the subcommand listing.
    p = sub.add_parser("db", description="inspect the pipeline database")
    db_sub = p.add_subparsers(dest="db_command", required=True, metavar="DB_COMMAND")
    q = db_sub.add_parser("templates", help="list all template versions of one identity")
    q.add_argument("--user", required=True, help="user ID (1-600)")
    q.add_argument("--hand", required=True, help="Left or Right")
    q.add_argument("--finger", required=True, help="thumb, index, middle, ring or little")
    q.add_argument("--db", default=None, help="SQLite path (default: paths.db from the config)")
    q.set_defaults(func=cmd_db_templates)
    return parser


def main(argv: Sequence[str] | None = None, *, out: Output = print, err: Output | None = None,
         input_fn: InputFn = input, controller_factory: ControllerFactory = default_controller) -> int:
    """Entry point. `out`/`input_fn` are the output and input functions of `enroll`/`verify`;
    `controller_factory` builds the pipeline controller (tests inject fakes)."""
    err = err or (lambda s: print(s, file=sys.stderr))
    args = build_parser().parse_args(argv)
    args.input_fn = input_fn
    args.controller_factory = controller_factory
    try:
        conf = cfgmod.load(args.config)
    except cfgmod.ConfigError as exc:
        err(f"fpv: configuration error: {exc}")
        return EXIT_USAGE
    err(f"fpv: config {conf.source_path} ({conf.config_set_id}{', frozen' if conf.frozen else ''})")
    try:
        return int(args.func(args, conf, out))
    except CliInputError as exc:
        err(f"fpv {args.command}: {exc}")
        return EXIT_USAGE


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
