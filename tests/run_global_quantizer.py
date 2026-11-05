"""Global quantizers, production sampling, stump sums and sample means across workers."""
import json
import os
from pathlib import Path
import subprocess
import time

from xgboost import RabitTracker


def main():
    root = Path(__file__).resolve().parents[1]
    binary = root / "results/metal_global_quantizer_api_test"
    subprocess.run([str(binary), "{}"], check=True, timeout=300)
    for world in (2, 3, 5):
        tracker = RabitTracker(n_workers=world, host_ip="127.0.0.1", timeout=180)
        tracker.start()
        workers, logs = [], []
        try:
            for task in range(world):
                config = dict(tracker.worker_args(), dmlc_task_id=str(task), dmlc_timeout=180)
                log = open(root / f"results/global-quantizer-worker-{world}-{task}.log", "w+")
                logs.append(log)
                workers.append(subprocess.Popen([str(binary), json.dumps(config)], stdout=log,
                    stderr=subprocess.STDOUT, env=dict(os.environ, MTL_DEBUG_LAYER="1")))
            deadline = time.monotonic() + 300
            for worker in workers:
                code = worker.wait(timeout=max(1, deadline - time.monotonic()))
                if code:
                    raise RuntimeError(f"Global quantizer worker exited {code}")
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
        print(f"PASS {world} global quantizer/initialization workers", flush=True)


if __name__ == "__main__":
    main()
