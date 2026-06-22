from __future__ import annotations

from pathlib import Path

import yaml

from connectors.schema import Connector, compute_integrity_hash, validate_connector


def load_connector(path: str | Path, *, check_hash: bool = True) -> Connector:
    data = yaml.safe_load(Path(path).read_text())
    connector = Connector.model_validate(data)
    errors = validate_connector(connector, check_hash=check_hash)
    if errors:
        raise ValueError(f"invalid connector {path}: {'; '.join(errors)}")
    return connector


def load_connectors_dir(directory: str | Path) -> list[Connector]:
    directory = Path(directory)
    connectors: list[Connector] = []
    for path in sorted(directory.glob("*.yaml")):
        connectors.append(load_connector(path))
    return connectors


def dump_connector(connector: Connector, path: str | Path) -> None:
    connector.profile.integrity_sha256 = compute_integrity_hash(connector)
    payload = connector.model_dump(mode="json")
    Path(path).write_text(yaml.safe_dump(payload, sort_keys=False))
