"""Real local federated servers for Metal transport, sketch and learner tests.

Run with the isolated package/library prepared by scripts/test_federated.sh.
"""
import argparse
import contextlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import xgboost as xgb
from xgboost.federated import FederatedTracker


def serve(world, connection, certificates):
    options = {}
    if certificates:
        options = dict(server_key_path=str(certificates / "server-key.pem"),
                       server_cert_path=str(certificates / "server-cert.pem"),
                       client_cert_path=str(certificates / "client-cert.pem"))
    tracker = FederatedTracker(n_workers=world, port=0, secure=bool(certificates),
                               timeout=180, **options)
    tracker.start()
    connection.send(tracker.worker_args()["dmlc_tracker_port"])
    connection.recv()
    tracker.free()
    connection.close()


def run(stage, world, root, certificates=None):
    context = multiprocessing.get_context("spawn")
    connection, child = context.Pipe()
    server = context.Process(target=serve, args=(world, child, certificates))
    server.start()
    child.close()
    security = "-tls" if certificates else ""
    directory = root / f"results/federated-{stage}-{world}{security}"
    directory.mkdir(exist_ok=True)
    processes, logs = [], []
    passed = False
    try:
        deadline = time.monotonic() + 30
        while not connection.poll(.05):
            if not server.is_alive() or time.monotonic() > deadline:
                raise RuntimeError(f"Federated server did not start: {server.exitcode}")
        port = connection.recv()
        for rank in range(world):
            config = dict(dmlc_communicator="federated", federated_world_size=world,
                          federated_rank=rank, federated_server_address=f"127.0.0.1:{port}",
                          dmlc_timeout=180)
            if certificates:
                config.update(federated_server_cert_path=str(certificates / "server-cert.pem"),
                              federated_client_key_path=str(certificates / "client-key.pem"),
                              federated_client_cert_path=str(certificates / "client-cert.pem"))
            if stage == "learner":
                args = [sys.executable, str(root / "tests/test_metal_distributed_training.py"),
                        json.dumps(config), str(directory)]
            else:
                binary = "metal_federated_api_test" if stage == "transport" else "metal_federated_sketch_api_test"
                args = [str(root / "results" / binary), json.dumps(config)]
            log = open(directory / f"worker-{rank}.log", "w+")
            logs.append(log)
            processes.append(subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT,
                cwd=root, env=dict(os.environ, MTL_DEBUG_LAYER="1")))
        deadline = time.monotonic() + 300
        while True:
            codes = [process.poll() for process in processes]
            if any(code is not None and code != 0 for code in codes):
                raise RuntimeError(f"Federated {stage} world{world} failed: {codes}")
            if all(code == 0 for code in codes):
                break
            if time.monotonic() > deadline:
                raise TimeoutError(f"Federated {stage} world{world}: {codes}")
            time.sleep(.05)
        passed = True
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
        if passed:
            connection.send("stop")
            server.join(timeout=10)
        # An incomplete collective can block the upstream handler's condition
        # variable even after its RPC client disconnects. Isolate the server so
        # failed tests can always clean up without waiting for a missing worker.
        if server.is_alive():
            server.terminate()
            server.join(timeout=5)
        if server.is_alive():
            server.kill()
            server.join()
        connection.close()
        for log in logs:
            log.seek(0)
            print(log.read(), end="", flush=True)
            log.close()
    assert server.exitcode == 0, server.exitcode
    print(f"PASS federated {stage}{security}: {world} workers", flush=True)


def make_certificates(directory):
    config = directory / "certificate.conf"
    config.write_text("[req]\ndistinguished_name=dn\nx509_extensions=extensions\nprompt=no\n"
                      "[dn]\nCN=localhost\n[extensions]\nbasicConstraints=CA:TRUE\n"
                      "subjectAltName=DNS:localhost,IP:127.0.0.1\n")
    for name in ("server", "client"):
        subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-days", "1", "-nodes",
                        "-keyout", str(directory / f"{name}-key.pem"),
                        "-out", str(directory / f"{name}-cert.pem"), "-config", str(config)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["transport", "sketch", "learner", "all"], default="all")
    parser.add_argument("--worlds", type=int, nargs="+")
    parser.add_argument("--secure", action="store_true", help="Use mutual TLS with temporary test certificates")
    args = parser.parse_args()
    assert xgb.build_info()["USE_FEDERATED"] and xgb.build_info()["USE_METAL"]
    root = Path(__file__).resolve().parents[1]
    temporary = (tempfile.TemporaryDirectory(prefix="federated-tls-", dir=root / "results")
                 if args.secure else contextlib.nullcontext(None))
    with temporary as directory:
        certificates = Path(directory) if directory else None
        if certificates:
            make_certificates(certificates)
        for stage in (["transport", "sketch", "learner"] if args.stage == "all" else [args.stage]):
            for world in args.worlds or ([1, 2, 3] if stage == "learner" else [1, 2, 3, 5]):
                run(stage, world, root, certificates)


if __name__ == "__main__":
    main()
