"""Live terminal dashboard for mat pool + gateway status."""

from __future__ import annotations

import argparse
import os
import time
from datetime import UTC, datetime

import httpx
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from api.pool_health import build_pool_health
from connectors.dotenv import load_env
from connectors.pool_resolver import resolve_pool


def _gateway_base() -> str:
    host = os.environ.get("MAT_HOST", "127.0.0.1")
    port = os.environ.get("MAT_PORT", "8080")
    return f"http://{host}:{port}"


def _fetch_gateway(path: str) -> dict | None:
    key = os.environ.get("MAT_GATEWAY_KEY", "local-dev-key")
    url = f"{_gateway_base()}{path}"
    try:
        with httpx.Client(timeout=3.0) as client:
            r = client.get(url, headers={"Authorization": f"Bearer {key}"})
            if r.status_code == 200:
                return r.json()
    except httpx.HTTPError:
        return None
    return None


def _connector_table(health: dict) -> Table:
    t = Table(title="Active pool", expand=True)
    t.add_column("Connector", style="cyan", no_wrap=True)
    t.add_column("Model")
    t.add_column("Locality")
    t.add_column("Coding")
    t.add_column("$/1k out")
    t.add_column("Status")

    for c in health.get("connectors") or []:
        served = c.get("served")
        if served is True:
            status = Text("served", style="green")
        elif served is False:
            status = Text("not loaded", style="red")
        else:
            status = Text("api", style="blue")
        out_price = c.get("output_per_1k")
        price_s = f"{out_price:.5f}" if out_price is not None else "—"
        t.add_row(
            c["id"],
            c.get("model_name") or "",
            c.get("locality") or "",
            f"{c.get('coding_score', 0):.2f}",
            price_s,
            status,
        )
    return t


def _recent_table(recent: list[dict]) -> Table:
    t = Table(title="Recent requests", expand=True)
    t.add_column("Time", style="dim")
    t.add_column("Tier")
    t.add_column("Connector", style="cyan")
    t.add_column("Model")
    t.add_column("ms", justify="right")
    t.add_column("Cost", justify="right")

    for row in recent[:8]:
        ts = datetime.fromtimestamp(row.get("ts", 0), tz=UTC).strftime("%H:%M:%S")
        t.add_row(
            ts,
            row.get("tier") or "",
            (row.get("connector_id") or "")[:36],
            (row.get("model_id") or "")[:24],
            str(int(row.get("latency_ms") or 0)),
            f"${row.get('cost_usd', 0):.4f}",
        )
    if not recent:
        t.add_row("—", "—", "(no requests yet)", "—", "—", "—")
    return t


def build_layout(*, gateway: dict | None, local_health: dict, pool_source: str) -> Panel:
    metrics = (gateway or {}).get("metrics") or {}
    issues = local_health.get("issues") or []
    status = local_health.get("status", "unknown")
    gw_status = (gateway or {}).get("status", "offline")

    header = Text()
    header.append("mat dashboard  ", style="bold")
    header.append(f"pool: {status}  ", style="green" if status == "ok" else "yellow")
    header.append(f"gateway: {gw_status}  ", style="green" if gw_status == "ok" else "red")
    header.append(f"source: {pool_source}", style="dim")

    if issues:
        header.append("\n")
        for issue in issues[:3]:
            header.append(f"⚠ {issue}\n", style="yellow")

    parts = [header, _connector_table(local_health)]
    if gateway:
        mline = (
            f"Requests: {metrics.get('requests_total', 0)}  "
            f"avg {metrics.get('latency_ms_avg', 0)}ms  "
            f"cost ${metrics.get('cost_usd_total', 0):.4f}  "
            f"uptime {metrics.get('uptime_sec', 0)}s"
        )
        parts.append(Text(mline, style="dim"))
        parts.append(_recent_table(gateway.get("recent") or []))
    else:
        parts.append(
            Text(
                f"Waiting for gateway at {_gateway_base()}…",
                style="dim italic",
            )
        )

    return Panel(Group(*parts), title="mat", border_style="blue")


def run_dashboard(*, interval: float = 2.0) -> None:
    """Blocking live dashboard loop."""
    console = Console()
    try:
        with Live(console=console, refresh_per_second=4, screen=True) as live:
            while True:
                try:
                    res = resolve_pool()
                    local_health = build_pool_health(res.pool)
                    gw = _fetch_gateway("/v1/mat/status")
                    recent = _fetch_gateway("/v1/mat/recent?limit=8")
                    if gw and recent:
                        gw["recent"] = recent.get("recent", [])
                    live.update(
                        build_layout(
                            gateway=gw,
                            local_health=local_health,
                            pool_source=res.source,
                        )
                    )
                except (OSError, ValueError) as exc:
                    live.update(Panel(str(exc), title="mat error", border_style="red"))
                time.sleep(interval)
    except KeyboardInterrupt:
        console.print("\n[dim]mat stopped[/dim]")


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(description="Live mat pool + routing dashboard")
    parser.add_argument("--interval", type=float, default=2.0, help="refresh seconds")
    args = parser.parse_args()
    run_dashboard(interval=args.interval)


if __name__ == "__main__":
    main()
