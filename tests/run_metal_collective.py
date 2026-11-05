"""Exercise Metal collective arithmetic/transport with 1/2/3/4 localhost workers.

Distributed training is still guarded pending sketch/histogram/leaf integration.
"""
import json
import os
from pathlib import Path
import subprocess
import time

from xgboost import RabitTracker
from xgboost.core import XGBoostError


def main():
    root = Path(__file__).resolve().parents[1]
    binary = root / "results/metal_collective_api_test"
    subprocess.run([str(binary), "{}"], check=True, timeout=180)
    for world, disconnect in ((2, False), (3, False), (4, False), (2, True)):
        tracker = RabitTracker(n_workers=world, host_ip="127.0.0.1", timeout=5 if disconnect else 120)
        tracker.start()
        workers = []
        logs = []
        try:
            for task in range(world):
                config = dict(tracker.worker_args(), dmlc_task_id=str(task), dmlc_timeout=5 if disconnect else 120)
                log = open(root / f"results/metal-collective-worker-{world}-{task}{'-disconnect' if disconnect else ''}.log", "w+")
                logs.append(log)
                workers.append(subprocess.Popen([str(binary), json.dumps(config)] + (["--disconnect"] if disconnect else []),
                                                stdout=log, stderr=subprocess.STDOUT,
                                                env=dict(os.environ, MTL_DEBUG_LAYER="1")))
            deadline = time.monotonic() + 180
            for worker in workers:
                code = worker.wait(timeout=max(1, deadline - time.monotonic()))
                if code:
                    raise RuntimeError(f"Metal collective worker exited {code}")
            if disconnect:
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
                tracker.wait_for(timeout=120)
        finally:
            for worker in workers:
                if worker.poll() is None:
                    worker.kill()
                    worker.wait()
            for log in logs:
                log.seek(0)
                print(log.read(), end="", flush=True)
                log.close()
        print(f"PASS {world} local Metal collective workers (disconnect={disconnect})", flush=True)


if __name__ == "__main__":
    main()
