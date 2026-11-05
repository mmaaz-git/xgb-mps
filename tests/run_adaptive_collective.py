"""Run actual objectives with 1/2/3 local workers, including empty workers.

This exercises objectives directly; public learners have a separate test matrix.
"""
import json
import os
from pathlib import Path
import subprocess
import time

from xgboost import RabitTracker


def main():
    root = Path(__file__).resolve().parents[1]
    binary = root / "results/metal_adaptive_api_test"
    subprocess.run([str(binary), "{}"], check=True, timeout=60)
    for world in (2, 3):
        tracker = RabitTracker(n_workers=world, host_ip="127.0.0.1", timeout=45)
        tracker.start()
        workers = []
        logs = []
        try:
            for task in range(world):
                config = dict(tracker.worker_args(), dmlc_task_id=str(task), dmlc_timeout=45)
                log = open(root / f"results/adaptive-worker-{world}-{task}.log", "w+")
                logs.append(log)
                workers.append(subprocess.Popen([str(binary), json.dumps(config)],
                                                stdout=log, stderr=subprocess.STDOUT,
                                                env=dict(os.environ, MTL_DEBUG_LAYER="1")))
            deadline = time.monotonic() + 60
            for worker in workers:
                code = worker.wait(timeout=max(1, deadline - time.monotonic()))
                if code:
                    raise RuntimeError(f"Adaptive worker exited {code}")
            tracker.wait_for(timeout=45)
        finally:
            for worker in workers:
                if worker.poll() is None:
                    worker.kill()
                    worker.wait()
            for log in logs:
                log.seek(0)
                print(log.read(), end="", flush=True)
                log.close()
        print(f"PASS {world} local objective workers", flush=True)


if __name__ == "__main__":
    main()
