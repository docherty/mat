from __future__ import annotations

from connectors.schema import Connector
from eval.oracle import Task


def encode_task(task: Task) -> list[float]:
    return [
        task.difficulty,
        len(task.prompt) / 500.0,
        float(len(task.required_tags)),
    ]


def encode_connector(connector: Connector) -> list[float]:
    return connector.capability_vector()


def encode_pool(pool: list[Connector]) -> list[float]:
    """Flatten pool profiles: [conn1_vec, conn2_vec, ...] padded to fixed width elsewhere."""
    out: list[float] = []
    for connector in pool:
        out.extend(encode_connector(connector))
    return out
