"""Validate frozen CUDA caller-audit bookkeeping, not semantic or GPU parity.

Source and test references are resolved at the audit's frozen Git revision, never
against newer working-tree code. Supplemental validation/docs/results evidence is checked in the working tree;
new audit artifacts can postdate the frozen implementation.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SITES = 1078
EXPECTED_FAMILIES = 42
EXPECTED_CUDA_ROOTS = 109
DISPOSITIONS = {"implemented", "source-host", "platform-adaptation", "missing"}
ROLES = {"consumer", "support", "non-operation"}
REFERENCE = re.compile(r"(?P<path>[^:\s]+)(?::(?P<line>[1-9][0-9]*))?\Z")
EMBEDDED_REFERENCE = re.compile(
    r"(?<![\w./])((?:xgboost|tests|scripts|validation|docs|results)/"
    r"[\w./+-]+\.(?:cuh|metal|json|cc|cu|mm|py|sh|md|txt|log|h))(?![\w.])"
    r"(?::([0-9]+)(?:-([0-9]+))?)?"
)


class AuditError(Exception):
    """Invalid audit, inventory, or frozen reference."""


def require(condition, message):
    if not condition:
        raise AuditError(message)


def git(*args):
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True)
    if result.returncode:
        raise AuditError(f"git {' '.join(args)}: {result.stderr.decode().strip()}")
    return result.stdout


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as err:
        raise AuditError(f"Cannot read {path}: {err}") from err


def nonempty(value, context):
    require(
        isinstance(value, str) and bool(value.strip()),
        f"{context}: expected nonempty string",
    )


def strings(value, context, allow_empty=False):
    require(isinstance(value, list), f"{context}: expected list")
    require(allow_empty or bool(value), f"{context}: must not be empty")
    for item in value:
        nonempty(item, context)
    require(len(value) == len(set(value)), f"{context}: duplicate values")
    return set(value)


def records(value, context, key="id"):
    require(isinstance(value, list), f"{context}: expected list")
    output = {}
    for item in value:
        require(isinstance(item, dict), f"{context}: expected object")
        nonempty(item.get(key), f"{context}.{key}")
        identifier = item[key]
        require(identifier not in output, f"{context}: duplicate {key} {identifier}")
        output[identifier] = item
    return output


def prose(value, context):
    if isinstance(value, list):
        require(bool(value), f"{context}: must not be empty")
        for item in value:
            nonempty(item, context)
    else:
        nonempty(value, context)


class FrozenFiles:
    def __init__(self, revision):
        self.revision = revision
        self.cache = {}
        self.references = set()

    def content(self, path):
        parsed = PurePosixPath(path)
        require(
            not parsed.is_absolute() and ".." not in parsed.parts,
            f"Unsafe reference path: {path}",
        )
        if path not in self.cache:
            self.cache[path] = git("show", f"{self.revision}:{path}")
        return self.cache[path]

    def reference(self, value, context, generated=False):
        nonempty(value, context)
        match = REFERENCE.fullmatch(value)
        require(match is not None, f"{context}: expected path[:line], got {value!r}")
        path, line = match.group("path", "line")
        parsed = PurePosixPath(path)
        require(
            not parsed.is_absolute() and ".." not in parsed.parts,
            f"{context}: unsafe reference {value}",
        )
        try:
            if generated and path.startswith(("results/", "validation/", "docs/")):
                content = (ROOT / path).read_bytes()
            else:
                content = self.content(path)
        except (AuditError, OSError) as err:
            raise AuditError(f"{context}: invalid reference {value}: {err}") from err
        if line:
            require(
                int(line) <= len(content.splitlines()),
                f"{context}: line {line} exceeds frozen file {path}",
            )
        self.references.add(value)

    def embedded(self, value, context):
        if isinstance(value, dict):
            for key, item in value.items():
                self.embedded(item, f"{context}.{key}")
        elif isinstance(value, list):
            for item in value:
                self.embedded(item, context)
        elif isinstance(value, str):
            for match in EMBEDDED_REFERENCE.finditer(value):
                path, first, last = match.groups()
                self.reference(
                    path + (f":{first}" if first else ""), context, generated=True
                )
                if last:
                    require(
                        int(last) >= int(first),
                        f"{context}: reversed line range {match[0]}",
                    )
                    self.reference(f"{path}:{last}", context, generated=True)


def task_references(record, tasks, context):
    refs = []
    if record.get("task") is not None:
        refs.append(record["task"])
    if "tasks" in record:
        refs.extend(strings(record["tasks"], f"{context}.tasks", allow_empty=True))
    for ref in refs:
        require(
            isinstance(ref, str) and ref in tasks, f"{context}: unknown task {ref!r}"
        )
    return set(refs)


def validate(audit, inventory):
    require(
        isinstance(audit, dict) and isinstance(inventory, dict), "Expected JSON objects"
    )
    required = {
        "revision",
        "upstream_import",
        "families",
        "operations",
        "tasks",
        "numerical_contracts",
        "callsite_assignments",
        "evidence_scope",
    }
    require(
        required <= audit.keys(),
        f"Audit missing fields: {sorted(required - audit.keys())}",
    )
    for key in ("revision", "upstream_import"):
        value = audit[key]
        require(
            isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value),
            f"{key}: expected full lowercase Git commit ID",
        )
        require(
            git("rev-parse", f"{value}^{{commit}}").decode().strip() == value,
            f"{key}: must identify a commit",
        )
        require(inventory.get(key) == value, f"Audit/inventory {key} mismatch")
    files = FrozenFiles(audit["revision"])
    mapping = json.loads(files.content("validation/cuda-operation-map.json"))
    require(
        mapping["upstream_import"] == audit["upstream_import"],
        "Operation-map import mismatch",
    )
    expected_families = {group["id"] for group in mapping["groups"]}
    families = strings(audit["families"], "families")
    require(
        len(families) == EXPECTED_FAMILIES and families == expected_families,
        "families must equal the 42 frozen operation-map IDs",
    )
    scope = audit["evidence_scope"]
    if isinstance(scope, dict):
        for key in ("method", "execution", "coverage", "limits"):
            nonempty(scope.get(key), f"evidence_scope.{key}")
        if "reviews" in scope:
            strings(scope["reviews"], "evidence_scope.reviews")
    else:
        prose(scope, "evidence_scope")
    files.embedded(scope, "evidence_scope")

    sources = records(inventory.get("sources"), "inventory.sources", "path")
    sites = records(inventory.get("sites"), "inventory.sites")
    require(
        len(sites) == EXPECTED_SITES,
        f"Expected {EXPECTED_SITES} inventory sites, got {len(sites)}",
    )
    roots = {
        path
        for path in git(
            "ls-tree",
            "-r",
            "--name-only",
            audit["revision"],
            "xgboost/src",
            "xgboost/plugin",
        )
        .decode()
        .splitlines()
        if path.endswith((".cu", ".cuh"))
    }
    upstream_roots = {
        path
        for path in git(
            "ls-tree",
            "-r",
            "--name-only",
            audit["upstream_import"],
            "xgboost/src",
            "xgboost/plugin",
        )
        .decode()
        .splitlines()
        if path.endswith((".cu", ".cuh"))
    }
    mapped_roots = {
        path for group in mapping["groups"] for path in group["cuda_sources"]
    }
    require(
        len(roots) == EXPECTED_CUDA_ROOTS and roots == upstream_roots == mapped_roots,
        "Frozen/upstream/operation-map CUDA roots must equal the same 109 paths",
    )
    require(
        roots <= sources.keys(),
        f"Inventory lacks CUDA roots: {sorted(roots - sources.keys())}",
    )
    site_counts = Counter()
    for site_id, site in sites.items():
        path, line = site.get("path"), site.get("line")
        require(
            path in sources and type(line) is int and line > 0,
            f"{site_id}: invalid inventory path/line",
        )
        require(
            site_id == f"{path}:{line}",
            f"{site_id}: does not match inventory path/line",
        )
        files.reference(site_id, "inventory.sites")
        require(
            site.get("source")
            == files.content(path).decode().splitlines()[line - 1].strip(),
            f"{site_id}: inventory source text differs from frozen revision",
        )
        require(
            strings(site.get("families"), site_id, allow_empty=True) <= families,
            f"{site_id}: unknown family",
        )
        site_counts[path] += 1
    for path, source in sources.items():
        require(
            hashlib.sha256(files.content(path)).hexdigest() == source.get("sha256"),
            f"{path}: inventory hash differs from frozen revision",
        )
        require(
            type(source.get("sites")) is int and source["sites"] == site_counts[path],
            f"{path}: inventory site count mismatch",
        )
        require(
            strings(source.get("families"), path, allow_empty=True) <= families,
            f"{path}: unknown source family",
        )

    operations = records(audit["operations"], "operations")
    tasks = records(audit["tasks"], "tasks")
    contracts = records(audit["numerical_contracts"], "numerical_contracts")
    all_ids = list(operations) + list(tasks) + list(contracts)
    require(
        len(all_ids) == len(set(all_ids)),
        "IDs must be unique across operations/tasks/contracts",
    )
    referenced_tasks, covered_families = set(), set()
    for identifier, task in tasks.items():
        nonempty(task.get("title"), f"{identifier}.title")
        prose(task.get("evidence"), f"{identifier}.evidence")
        prose(task.get("acceptance"), f"{identifier}.acceptance")
        files.embedded(task, identifier)
    for identifier, operation in operations.items():
        required_op = {
            "families",
            "operation",
            "cuda",
            "metal",
            "disposition",
            "tests",
            "evidence",
            "limits",
            "task",
        }
        require(
            required_op <= operation.keys(),
            f"{identifier}: missing fields {sorted(required_op - operation.keys())}",
        )
        group = strings(operation["families"], f"{identifier}.families")
        require(
            group <= families,
            f"{identifier}: unknown operation families {sorted(group - families)}",
        )
        covered_families.update(group)
        nonempty(operation["operation"], f"{identifier}.operation")
        require(
            operation["disposition"] in DISPOSITIONS,
            f"{identifier}: invalid disposition",
        )
        prose(operation["evidence"], f"{identifier}.evidence")
        prose(operation["limits"], f"{identifier}.limits")
        for key in ("cuda", "metal", "tests"):
            for ref in strings(
                operation[key], f"{identifier}.{key}", allow_empty=key != "cuda"
            ):
                files.reference(ref, f"{identifier}.{key}")
        if operation["disposition"] == "implemented":
            require(
                bool(operation["metal"]) and bool(operation["tests"]),
                f"{identifier}: implemented operations need Metal path and specific test",
            )
        if operation["disposition"] == "missing":
            require(
                operation["task"] is not None,
                f"{identifier}: missing operation needs task",
            )
        referenced_tasks.update(task_references(operation, tasks, identifier))
        files.embedded(operation, identifier)
    require(
        covered_families == families,
        f"Families without logical operations: {sorted(families - covered_families)}",
    )
    for identifier, contract in contracts.items():
        nonempty(contract.get("disposition"), f"{identifier}.disposition")
        prose(contract.get("evidence"), f"{identifier}.evidence")
        referenced_tasks.update(task_references(contract, tasks, identifier))
        if "operations" in contract:
            refs = strings(
                contract["operations"], f"{identifier}.operations", allow_empty=True
            )
            require(
                refs <= operations.keys(), f"{identifier}: unknown operation references"
            )
        files.embedded(contract, identifier)
    require(
        referenced_tasks == tasks.keys(),
        f"Unreferenced tasks: {sorted(tasks.keys() - referenced_tasks)}",
    )

    assignments = records(audit["callsite_assignments"], "callsite_assignments", "site")
    require(
        assignments.keys() == sites.keys(),
        f"Callsite assignment mismatch: {len(sites.keys() - assignments.keys())} missing, "
        f"{len(assignments.keys() - sites.keys())} extra",
    )
    for site_id, assignment in assignments.items():
        role = assignment.get("role")
        require(role in ROLES, f"{site_id}: invalid assignment role {role!r}")
        refs = strings(
            assignment.get("operations"), site_id, allow_empty=role == "non-operation"
        )
        require(
            refs <= operations.keys(),
            f"{site_id}: unknown operations {sorted(refs - operations.keys())}",
        )
        nonempty(assignment.get("reason"), f"{site_id}.reason")
    return {
        "families": len(families),
        "roots": len(roots),
        "sites": len(sites),
        "operations": len(operations),
        "tasks": len(tasks),
        "contracts": len(contracts),
        "references": len(files.references),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit", type=Path, default=ROOT / "validation/cuda-caller-audit.json"
    )
    parser.add_argument(
        "--inventory", type=Path, default=ROOT / "validation/cuda-calls-inventory.json"
    )
    parser.add_argument(
        "--check-inventory",
        action="store_true",
        help="also regenerate and compare lexical inventory with inventory_cuda_calls.py --check",
    )
    args = parser.parse_args()
    try:
        audit, inventory = read_json(args.audit), read_json(args.inventory)
        counts = validate(audit, inventory)
        if args.check_inventory:
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/inventory_cuda_calls.py"),
                    "--revision",
                    audit["revision"],
                    "--output",
                    str(args.inventory.resolve()),
                    "--check",
                ],
                cwd=ROOT,
                check=True,
            )
    except (AuditError, subprocess.CalledProcessError) as err:
        print(f"FAIL: {err}", file=sys.stderr)
        return 1
    print(
        "PASS frozen CUDA audit structure: "
        + ", ".join(f"{n} {name}" for name, n in counts.items())
    )
    print(
        "Structural closure only; this does not prove semantic completeness, GPU execution, "
        "numerical equivalence, NVIDIA runtime parity, or performance."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
