"""Tests for the Cloudflare R2 client wrapper."""

from __future__ import annotations

import io
from unittest.mock import MagicMock

import pytest

from another_world.data.storage_r2 import R2Client, R2Config


def test_r2_config_from_env_happy_path() -> None:
    env = {
        "R2_ACCOUNT_ID": "acct",
        "R2_ACCESS_KEY_ID": "ak",
        "R2_SECRET_ACCESS_KEY": "sk",
        "R2_ENDPOINT_URL": "https://x.r2.cloudflarestorage.com",
    }
    cfg = R2Config.from_env(env)
    assert cfg.account_id == "acct"
    assert cfg.access_key_id == "ak"
    assert cfg.endpoint_url.endswith("cloudflarestorage.com")


def test_r2_config_endpoint_defaults_from_account() -> None:
    env = {
        "R2_ACCOUNT_ID": "acct",
        "R2_ACCESS_KEY_ID": "ak",
        "R2_SECRET_ACCESS_KEY": "sk",
    }
    cfg = R2Config.from_env(env)
    assert cfg.endpoint_url == "https://acct.r2.cloudflarestorage.com"


def test_r2_config_missing_raises() -> None:
    with pytest.raises(RuntimeError, match="missing R2 credentials"):
        R2Config.from_env({})


def _fake_client(*, with_buckets: bool = False) -> MagicMock:
    client = MagicMock()
    if with_buckets:
        client.list_buckets.return_value = {
            "Buckets": [
                {"Name": "another-world-raw"},
                {"Name": "another-world-shards"},
            ]
        }
    return client


def _cfg() -> R2Config:
    return R2Config(
        account_id="a", access_key_id="k", secret_access_key="s",
        endpoint_url="https://a.r2.cloudflarestorage.com",
    )


def test_list_buckets_uses_injected_client() -> None:
    fake = _fake_client(with_buckets=True)
    r2 = R2Client(_cfg(), client=fake)
    assert r2.list_buckets() == ["another-world-raw", "another-world-shards"]
    fake.list_buckets.assert_called_once()


def test_list_objects_paginates() -> None:
    fake = _fake_client()
    paginator = MagicMock()
    paginator.paginate.return_value = iter(
        [
            {"Contents": [{"Key": "a"}, {"Key": "b"}]},
            {"Contents": [{"Key": "c"}]},
            {},
        ]
    )
    fake.get_paginator.return_value = paginator
    r2 = R2Client(_cfg(), client=fake)
    assert list(r2.list_objects("bucket")) == ["a", "b", "c"]
    fake.get_paginator.assert_called_with("list_objects_v2")


def test_upload_and_download() -> None:
    fake = _fake_client()
    r2 = R2Client(_cfg(), client=fake)
    fh = io.BytesIO(b"payload")
    r2.upload_fileobj(fh, "bucket", "key", extra_args={"ContentType": "x"})
    fake.upload_fileobj.assert_called_once()
    out = io.BytesIO()
    r2.download_fileobj("bucket", "key", out)
    fake.download_fileobj.assert_called_once()


def test_delete_and_head_object() -> None:
    fake = _fake_client()
    fake.head_object.return_value = {"ContentLength": 12}
    r2 = R2Client(_cfg(), client=fake)
    assert r2.head_object("b", "k")["ContentLength"] == 12
    r2.delete_object("b", "k")
    fake.delete_object.assert_called_once_with(Bucket="b", Key="k")
