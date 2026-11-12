"""Emit hash-pinned lock lines for a `pip install --dry-run --report` result.

Keeps only wheels the containers can install: CPython 3.12 (cp312, abi3 or py3-none) on manylinux
aarch64 and x86_64, or pure `*-none-any`. Warns on stderr when a package lacks wheels for both
architectures. Needs network access to pypi.org.

Toolchain additions (resolved against the existing pins, so protobuf etc. stay at the TensorFlow pins):

    docker run --rm -v "$PWD/scripts:/w" fpv-toolchain:phase1 bash -c '
      pip freeze | grep "==" > /tmp/c.txt
      pip install --dry-run --quiet --report /tmp/r.json -c /tmp/c.txt "a2a-sdk[http-server]==1.3.0" uvicorn httpx scikit-learn
      python /w/lockgen.py /tmp/r.json' > /tmp/lock_add.txt
    # then merge the blocks into requirements-pipeline.lock in name order and rebuild the image

Orchestrator lock (from scratch):

    docker run --rm -v "$PWD/scripts:/w" python:3.12-slim bash -c '
      pip install --dry-run --quiet --report /tmp/r.json "google-adk==2.13.0" "a2a-sdk==1.3.0" httpx numpy pyyaml pytest==9.1.1
      python /w/lockgen.py /tmp/r.json' > /tmp/orch_lock.txt
    # prepend the header comment of requirements-orchestrator.lock, then `docker compose build orchestrator`

Usage: python lockgen.py REPORT.json [PACKAGE_TO_SKIP ...]
"""

import json
import sys
import urllib.request


def installable(filename: str) -> bool:
    if filename.endswith("-none-any.whl"):
        return True
    if not filename.endswith(".whl"):
        return False
    py, abi, plat = filename[:-4].split("-")[-3:]
    tag_ok = (py == "cp312" and abi == "cp312") or abi == "abi3" or (py == "py3" and abi == "none")
    return tag_ok and "musllinux" not in plat and "manylinux" in plat and ("aarch64" in plat or "x86_64" in plat)


def main() -> None:
    report = json.load(open(sys.argv[1]))
    skip = {a.lower() for a in sys.argv[2:]}
    for item in sorted(report["install"], key=lambda i: i["metadata"]["name"].lower()):
        name, ver = item["metadata"]["name"], item["metadata"]["version"]
        if name.lower() in skip:
            continue
        files = [u for u in json.load(urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{ver}/json"))["urls"]
                 if installable(u["filename"])]
        archs = {a for u in files for a in ("aarch64", "x86_64", "any") if a in u["filename"]}
        if not files or not ("any" in archs or {"aarch64", "x86_64"} <= archs):
            print(f"# WARNING no complete wheel set for {name}=={ver}: {sorted(archs)}", file=sys.stderr)
        hashes = sorted({u["digests"]["sha256"] for u in files})
        lines = [f"{name}=={ver} \\"] + [f"    --hash=sha256:{h} \\" for h in hashes]
        lines[-1] = lines[-1].rstrip(" \\")
        print("\n".join(lines))


if __name__ == "__main__":
    main()
