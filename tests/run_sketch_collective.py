"""Ordered Metal variable reductions and distributed sketch source references.

Learner remains guarded until distributed histogram/leaf consumers are ported.
"""
import json
import os
from pathlib import Path
import subprocess
import time
import sys

from xgboost import RabitTracker
from xgboost.core import XGBoostError


def main():
    root = Path(__file__).resolve().parents[1]
    binary = root / "results/metal_sketch_collective_api_test"
    failure_only = "--failure-only" in sys.argv
    if not failure_only:
        subprocess.run([str(binary), "{}"], check=True, timeout=240)
    runs = [] if failure_only else [(2, ""), (3, ""), (5, "")]
    runs += [(2, "--disconnect-child"), (2, "--disconnect-root")]
    for world, failure in runs:
        tracker = RabitTracker(n_workers=world, host_ip="127.0.0.1", timeout=5 if failure else 180)
        tracker.start()
        workers = []
        logs = []
        try:
            for task in range(world):
                config = dict(tracker.worker_args(), dmlc_task_id=str(task), dmlc_timeout=5 if failure else 180)
                log = open(root / f"results/sketch-collective-worker-{world}-{task}{failure}.log", "w+")
                logs.append(log)
                workers.append(subprocess.Popen([str(binary), json.dumps(config)] + ([failure] if failure else []),
                                                stdout=log, stderr=subprocess.STDOUT,
                                                env=dict(os.environ, MTL_DEBUG_LAYER="1")))
            deadline = time.monotonic() + 240
            for worker in workers:
                code = worker.wait(timeout=max(1, deadline - time.monotonic()))
                if code:
                    raise RuntimeError(f"sketch collective worker exited {code}")
            if failure:
                try:
                    tracker.wait_for(timeout=8)
                except XGBoostError as error:
                    if "Poll timeout" not in str(error):
                        raise
                finally:
                    try:
                        tracker.free()
                    except XGBoostError as error:
                        if "Poll timeout" not in str(error):
                            raise
            else:
                tracker.wait_for(timeout=180)
        finally:
            for worker in workers:
                if worker.poll() is None:
                    worker.kill()
                    worker.wait()
            for log in logs:
                log.seek(0)
                print(log.read(), end="", flush=True)
                log.close()
        print(f"PASS {world} local sketch collective workers {failure}", flush=True)


if __name__ == "__main__":
    main()
