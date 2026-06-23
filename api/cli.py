"""Default mat entrypoint: gateway + live dashboard."""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time

import httpx

from connectors.dotenv import load_env


def _wait_for_health(*, timeout_sec: float = 30.0) -> bool:
    host = os.environ.get("MAT_HOST", "127.0.0.1")
    port = os.environ.get("MAT_PORT", "8080")
    url = f"http://{host}:{port}/health"
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            with httpx.Client(timeout=2.0) as client:
                r = client.get(url)
                if r.status_code == 200 and r.json().get("pool_size", 0) > 0:
                    return True
        except httpx.HTTPError:
            pass
        time.sleep(0.25)
    return False


def _start_server_thread() -> threading.Thread:
    import uvicorn

    from api.server import create_app

    host = os.environ.get("MAT_HOST", "127.0.0.1")
    port = int(os.environ.get("MAT_PORT", "8080"))
    config = uvicorn.Config(create_app(), host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)

    def _run() -> None:
        server.run()

    thread = threading.Thread(target=_run, name="mat-serve", daemon=True)
    thread.start()
    return thread


def main(argv: list[str] | None = None) -> None:
    load_env()
    parser = argparse.ArgumentParser(
        prog="mat",
        description="mat — capability-routed LLM gateway (server + dashboard by default)",
    )
    parser.add_argument(
        "--no-tui",
        action="store_true",
        help="run gateway only (no terminal dashboard)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="dashboard refresh interval (seconds)",
    )
    args = parser.parse_args(argv)

    if args.no_tui:
        from api.server import main as serve_main

        serve_main()
        return

    _start_server_thread()
    if not _wait_for_health():
        print("mat: gateway failed to start or pool is empty — check active.yaml", file=sys.stderr)
        raise SystemExit(1)

    from api.dashboard import run_dashboard

    try:
        run_dashboard(interval=args.interval)
    except KeyboardInterrupt:
        pass
