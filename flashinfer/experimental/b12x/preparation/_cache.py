"""Versioned tuning decisions indexed by candidate program identities."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from filelock import FileLock

from .types import FrozenMapping


def _json(value):
    if isinstance(value, FrozenMapping):
        value = value.to_dict()
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def cache_identity(
    namespace: Mapping[str, object], device_ordinal: int, *, cache_policy: str = "cold"
):
    import torch

    if cache_policy not in ("cold", "warm"):
        raise ValueError("cache_policy must be cold or warm")
    raw_version = os.environ.get("B12X_TUNING_CACHE_VERSION", "1")
    try:
        version = int(raw_version)
    except ValueError:
        raise ValueError(
            "B12X_TUNING_CACHE_VERSION must be a positive integer"
        ) from None
    if version <= 0:
        raise ValueError("B12X_TUNING_CACHE_VERSION must be a positive integer")
    with torch.cuda.device(device_ordinal):
        name = torch.cuda.get_device_name(device_ordinal).strip()
        if not name:
            raise RuntimeError("CUDA device name is unavailable")
        properties = torch.cuda.get_device_properties(device_ordinal)
        # Key on the silicon, not the product label. One part ships under
        # several names (for example RTX PRO 6000 Blackwell "Workstation
        # Edition" and "Max-Q Workstation Edition"), so a tensor-parallel group
        # that mixes those labels would otherwise fail reconcile() across ranks
        # even though every rank can reuse the same decisions. Reported total
        # memory is excluded for the same reason: those two labels report it
        # several MiB apart. The name is still read above so an unusable device
        # fails closed.
        return {
            "schema_version": 23,
            "tuning_cache_version": version,
            "measurement": "stream_gated_events_v2"
            if cache_policy == "cold"
            else "stream_gated_events_v2_warm",
            "namespace": dict(namespace),
            "compute_capability": [int(properties.major), int(properties.minor)],
            "sm_count": int(properties.multi_processor_count),
        }


class SelectionCache:
    """Locked reload/merge/atomic publication prevents concurrent lost updates."""

    def __init__(self, root: str | Path, identity: Mapping[str, object]):
        self.identity = json.loads(_json(identity))
        if self.identity.get("schema_version") != 23:
            raise ValueError("preparation selection cache requires schema 23")
        version = self.identity.get("tuning_cache_version")
        if type(version) is not int or version <= 0:
            raise ValueError("selection cache requires a positive tuning_cache_version")
        self.path = Path(root) / f"{digest(self.identity)}.json"
        self.records = self._read()
        self._agreed_records = None
        self._pending = {}
        # Deferred saves accumulate in memory and reach the file in one locked
        # merge per flush instead of one per record.
        self.deferred = False

    def _read(self):
        if not self.path.exists():
            return {}
        payload = json.loads(self.path.read_text())
        if not isinstance(payload, dict) or set(payload) != {"identity", "records"}:
            raise ValueError("invalid preparation selection cache")
        if payload["identity"] != self.identity:
            raise ValueError("selection identity differs from its cache key")
        return payload["records"]

    def get(self, key):
        records = self.records if self._agreed_records is None else self._agreed_records
        record = records.get(key)
        return None if record is None else FrozenMapping(record)

    def reconcile(self, snapshots):
        """Use the first completed record in rank order for this session.

        Executable artifacts remain local. Only selection metadata is shared;
        preparation still compiles and primes each selected configuration locally.
        """
        agreed = {}
        for snapshot in snapshots:
            if snapshot.identity.to_dict() != self.identity:
                raise ValueError("tuning cache identities differ across ranks")
            records = snapshot.records.to_dict()
            for key, record in records.items():
                agreed.setdefault(key, record)
        self._agreed_records = agreed

    def flush(self):
        """Merge pending records into the file under its lock."""
        if not self._pending:
            return
        pending, self._pending = self._pending, {}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with FileLock(self.path.with_suffix(".lock")):
                records = self._read()
                records.update(pending)
                temporary = None
                try:
                    with tempfile.NamedTemporaryFile(
                        mode="w",
                        dir=self.path.parent,
                        suffix=".tmp",
                        delete=False,
                    ) as stream:
                        temporary = stream.name
                        stream.write(
                            _json({"identity": self.identity, "records": records})
                        )
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(temporary, self.path)
                finally:
                    if temporary is not None and os.path.exists(temporary):
                        os.unlink(temporary)
                self.records = records
        except BaseException:
            pending.update(self._pending)
            self._pending = pending
            raise

    def save(self, key, *, assignment, config, coverage, programs, finalists=None):
        """Record a completed selection; it reaches the file at the next flush."""
        record = {
            "assignment": FrozenMapping(assignment).to_dict(),
            "config": FrozenMapping(config).to_dict(),
            "coverage": dict(coverage),
            "programs": [
                list(item)
                for item in sorted(
                    {(program.dialect, program.key) for program in programs}
                )
            ],
        }
        if finalists is not None:
            record["finalists"] = [
                {
                    "assignment": item.assignment.to_dict(),
                    "family": item.family.to_dict(),
                    "candidate_index": item.candidate_index,
                    "latency_us": item.latency_us,
                }
                for item in finalists
            ]
        update = {key: record}
        self.records.update(update)
        if self._agreed_records is not None:
            self._agreed_records.update(update)
        self._pending.update(update)
        if not self.deferred or len(self._pending) >= 64:
            self.flush()
