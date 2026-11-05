#!/usr/bin/env python3
"""Select native/Python Metal checks; rebuild first and retain per-step evidence."""

import argparse
from datetime import datetime, timezone
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests/native_tests.json"


def inventory():
    data = json.loads(MANIFEST.read_text())
    tests = data["tests"]
    ids = [test["id"] for test in tests]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate native test IDs")
    for test in tests:
        if (
            test["profile"] not in data["profiles"]
            or not (ROOT / test["source"]).is_file()
        ):
            raise ValueError(f"Invalid native test entry: {test['id']}")
    expected = {
        str(p.relative_to(ROOT))
        for p in (ROOT / "tests").glob("metal_*")
        if p.suffix in {".cc", ".mm"}
    }
    # Federated transport has its own optional build and runner.
    expected.remove("tests/metal_federated_api.cc")
    actual = {test["source"] for test in tests}
    if expected != actual:
        raise ValueError(
            f"Native inventory mismatch: missing={expected - actual}, extra={actual - expected}"
        )
    return data


def capture(command):
    return subprocess.check_output(command, cwd=ROOT, text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list", action="store_true", help="List native IDs without building"
    )
    parser.add_argument(
        "--native", nargs="+", metavar="ID", help="Native IDs or quoted glob patterns"
    )
    parser.add_argument(
        "--python",
        nargs="*",
        metavar="PATH",
        help="Pytest files/node IDs; no paths means all tests",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="All native and Python checks (default without selectors)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without building or testing",
    )
    parser.add_argument(
        "--log-dir", type=Path, help="New directory for logs and summary.json"
    )
    args = parser.parse_args()
    data = inventory()
    if args.full and (args.native is not None or args.python is not None):
        parser.error("Use --full or explicit native/Python selections")
    full = args.full or (args.native is None and args.python is None)
    selected = data["tests"] if full else []
    if args.native:
        matched = set()
        for pattern in args.native:
            found = {
                t["id"] for t in data["tests"] if fnmatch.fnmatchcase(t["id"], pattern)
            }
            if not found:
                parser.error(f"Unknown native test selector: {pattern}")
            matched.update(found)
        selected = [t for t in data["tests"] if t["id"] in matched]
    if args.list:
        for test in selected:
            print(f"{test['id']}: {test['description']}")
        print(
            f"{len(selected)} native groups; optional federated checks use scripts/test_federated.sh"
        )
        return 0

    env = os.environ.copy()
    values = {
        "root": str(ROOT),
        "pythonpath_suffix": ":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "",
    }
    for name in ("libomp", "snappy"):
        needed = name == "libomp" or any(
            "{" + name + "}" in part
            for test in selected
            for part in data["profiles"][test["profile"]]
        )
        if needed:
            values[name] = env.get(name.upper() + "_ROOT") or (
                "<brew-prefix-" + name + ">"
                if args.dry_run
                else capture(["brew", "--prefix", name])
            )
    env["DYLD_LIBRARY_PATH"] = (
        values["libomp"]
        + "/lib"
        + (":" + env["DYLD_LIBRARY_PATH"] if env.get("DYLD_LIBRARY_PATH") else "")
    )
    steps = [("build-metal", ["bash", "scripts/build.sh"], {})]
    for test in selected:
        context = dict(values, source=test["source"], output=test["output"])
        command = [part.format(**context) for part in data["profiles"][test["profile"]]]
        steps.append(("compile-" + test["id"], command, {}))
        overrides = {
            key: value.format(**values) for key, value in test.get("env", {}).items()
        }
        steps.append((test["id"], test["run"], overrides))
    if full or args.python is not None:
        steps.append(
            ("linear-reference", ["bash", "tests/build_linear_reference.sh"], {})
        )
        steps.append(
            (
                "pytest",
                [
                    ".venv/bin/python",
                    "-m",
                    "pytest",
                    "-q",
                    *(args.python or ["tests"]),
                    "--require-metal",
                ],
                # pytest.ini updates only the parent interpreter's sys.path.
                # Distributed workers run scripts directly and inherit this env.
                {"PYTHONPATH": str(ROOT / "xgboost/python-package") + values["pythonpath_suffix"]},
            )
        )
    if args.dry_run:
        for label, command, overrides in steps:
            print(
                label
                + ": "
                + shlex.join(
                    [f"{key}={value}" for key, value in overrides.items()] + command
                )
            )
        print(f"DRY RUN: {len(selected)} native groups; no commands executed")
        return 0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_dir = (
        args.log_dir or ROOT / "results/test-runs" / f"{stamp}-{os.getpid()}"
    ).resolve()
    log_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "started_utc": stamp,
        "revision": capture(["git", "rev-parse", "HEAD"]),
        "working_tree": capture(["git", "status", "--short"]),
        "diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=ROOT)
        ).hexdigest(),
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "untracked_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in subprocess.check_output(
                ["git", "ls-files", "--others", "--exclude-standard", "-z"],
                cwd=ROOT,
                text=True,
            ).split("\0")
            if name and (ROOT / name).is_file()
        },
        "native_groups": [test["id"] for test in selected],
        "python_selection": (
            (args.python or ["tests"]) if full or args.python is not None else []
        ),
        "steps": [],
    }

    def save():
        (log_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    print(f"Logs: {log_dir}", flush=True)
    code = 0
    try:
        for number, (label, command, overrides) in enumerate(steps, 1):
            path = log_dir / f"{number:03}-{label}.log"
            print(f"[{number}/{len(steps)}] {label}: {shlex.join(command)}", flush=True)
            start = time.monotonic()
            item = {
                "name": label,
                "command": command,
                "environment": overrides,
                "log": path.name,
                "status": "running",
            }
            report["steps"].append(item)
            save()
            with path.open("w") as log:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=env | overrides,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
                try:
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                        log.flush()
                    code = process.wait()
                except BaseException:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise
            item.update(
                exit_code=code,
                seconds=time.monotonic() - start,
                status="passed" if code == 0 else "failed",
            )
            if label == "build-metal" and code == 0:
                library = ROOT / "xgboost/lib/libxgboost.dylib"
                report["library_sha256"] = hashlib.sha256(
                    library.read_bytes()
                ).hexdigest()
            save()
            print(f"[{number}/{len(steps)}] {label}: exit={code}", flush=True)
            if code:
                break
        report["status"] = "passed" if code == 0 else "failed"
    except KeyboardInterrupt:
        code, report["status"] = 130, "interrupted"
    except Exception as error:
        code, report["status"] = 1, "failed"
        report["error"] = str(error)
        print(str(error), file=sys.stderr)
    finally:
        if report["steps"] and report["steps"][-1]["status"] == "running":
            report["steps"][-1].update(status=report["status"], exit_code=code)
        report["exit_code"] = code
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()
        print(
            f"{report['status'].upper()}: exit={code}; {log_dir / 'summary.json'}",
            flush=True,
        )
    return code if code >= 0 else 128 - code


if __name__ == "__main__":
    sys.exit(main())
