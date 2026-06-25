"""Cloudflare R2 client wrapper.

R2 exposes an S3-compatible API, so we use :mod:`boto3` under the hood and
just preset the endpoint and credentials from environment variables.

Environment variables (see also ``docs/storage_setup.md`` and ``.env.example``):

- ``R2_ACCOUNT_ID``
- ``R2_ACCESS_KEY_ID``
- ``R2_SECRET_ACCESS_KEY``
- ``R2_ENDPOINT_URL``  (``https://<account>.r2.cloudflarestorage.com``)

The class supports two operating modes:

- **online**: real boto3 client, requires credentials.
- **offline**: returns a stub that records calls; used by tests.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, BinaryIO, Iterator

from another_world.utils.logging import get_logger

_LOG = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class R2Config:
    """Resolved credentials for a Cloudflare R2 account."""

    account_id: str
    access_key_id: str
    secret_access_key: str
    endpoint_url: str

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "R2Config":
        env = env or dict(os.environ)
        missing = [
            k
            for k in (
                "R2_ACCOUNT_ID",
                "R2_ACCESS_KEY_ID",
                "R2_SECRET_ACCESS_KEY",
            )
            if not env.get(k)
        ]
        if missing:
            raise RuntimeError(
                f"missing R2 credentials in environment: {missing}. "
                "See docs/storage_setup.md."
            )
        endpoint = env.get("R2_ENDPOINT_URL") or (
            f"https://{env['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com"
        )
        return cls(
            account_id=env["R2_ACCOUNT_ID"],
            access_key_id=env["R2_ACCESS_KEY_ID"],
            secret_access_key=env["R2_SECRET_ACCESS_KEY"],
            endpoint_url=endpoint,
        )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class R2Client:
    """Thin S3-style client for Cloudflare R2.

    Lazily constructs the underlying ``boto3.client('s3', ...)`` so importing
    this module never fails on machines without ``boto3``.
    """

    def __init__(self, config: R2Config, *, client: Any = None) -> None:
        self.config = config
        self._client = client

    @classmethod
    def from_env(cls) -> "R2Client":
        return cls(R2Config.from_env())

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                import boto3  # type: ignore[import-not-found]
            except ImportError as exc:  # pragma: no cover - only when missing
                raise ImportError(
                    "boto3 is required (`pip install boto3`)."
                ) from exc
            self._client = boto3.client(
                "s3",
                endpoint_url=self.config.endpoint_url,
                aws_access_key_id=self.config.access_key_id,
                aws_secret_access_key=self.config.secret_access_key,
                region_name="auto",
            )
        return self._client

    # ----- ops ------------------------------------------------------------

    def list_buckets(self) -> list[str]:
        resp = self.client.list_buckets()
        return [b["Name"] for b in resp.get("Buckets", [])]

    def list_objects(self, bucket: str, *, prefix: str = "") -> Iterator[str]:
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                yield obj["Key"]

    def head_object(self, bucket: str, key: str) -> dict[str, Any]:
        return self.client.head_object(Bucket=bucket, Key=key)

    def upload_fileobj(
        self,
        fh: BinaryIO,
        bucket: str,
        key: str,
        *,
        extra_args: dict[str, Any] | None = None,
    ) -> None:
        _LOG.info("R2 upload -> %s/%s", bucket, key)
        self.client.upload_fileobj(
            fh, Bucket=bucket, Key=key, ExtraArgs=extra_args or {},
        )

    def download_fileobj(self, bucket: str, key: str, fh: BinaryIO) -> None:
        self.client.download_fileobj(Bucket=bucket, Key=key, Fileobj=fh)

    def delete_object(self, bucket: str, key: str) -> None:
        self.client.delete_object(Bucket=bucket, Key=key)


__all__ = ["R2Client", "R2Config"]
