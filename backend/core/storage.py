from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import IO, Optional


@dataclass(frozen=True, slots=True)
class S3Config:
    bucket: str
    region: str | None


def _s3_region() -> str | None:
    return (os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "").strip() or None


def s3_config_for_bucket(bucket: str | None) -> S3Config | None:
    """An S3Config for an explicit bucket (used to read/delete an artifact from
    whichever bucket it was written to)."""
    bucket = (bucket or "").strip()
    if not bucket:
        return None
    return S3Config(bucket=bucket, region=_s3_region())


def get_s3_config() -> S3Config | None:
    """Default ("temp") bucket: the transient upload hand-off and anonymous
    report files (which auto-expire). Resolves S3_TEMP_BUCKET, falling back to
    S3_BUCKET so a single-bucket deployment keeps working unchanged."""
    return s3_config_for_bucket(os.environ.get("S3_TEMP_BUCKET") or os.environ.get("S3_BUCKET"))


def get_persist_s3_config() -> S3Config | None:
    """Persistent bucket: signed-in users' report files, kept until deletion.
    Resolves S3_PERSIST_BUCKET, falling back to S3_BUCKET (single-bucket setup)."""
    return s3_config_for_bucket(os.environ.get("S3_PERSIST_BUCKET") or os.environ.get("S3_BUCKET"))


@lru_cache(maxsize=4)
def _s3_client(region: str | None):
    # Cached per region: boto3 clients are thread-safe and creating one is
    # non-trivial (session + endpoint + credential resolution), so a worker
    # touching several keys reuses one client instead of rebuilding it each call.
    import boto3

    kwargs = {}
    if region:
        kwargs["region_name"] = region
    return boto3.client("s3", **kwargs)


def s3_upload_fileobj(
    cfg: S3Config,
    *,
    key: str,
    fileobj: IO[bytes],
    content_type: str | None = None,
) -> None:
    client = _s3_client(cfg.region)
    extra_args = {}
    if content_type:
        extra_args["ContentType"] = content_type
    if extra_args:
        client.upload_fileobj(fileobj, cfg.bucket, key, ExtraArgs=extra_args)
    else:
        client.upload_fileobj(fileobj, cfg.bucket, key)


def s3_put_bytes(
    cfg: S3Config,
    *,
    key: str,
    data: bytes,
    content_type: str | None = None,
) -> None:
    client = _s3_client(cfg.region)
    kwargs = {}
    if content_type:
        kwargs["ContentType"] = content_type
    client.put_object(Bucket=cfg.bucket, Key=key, Body=data, **kwargs)


def s3_get_bytes(cfg: S3Config, *, key: str) -> bytes:
    client = _s3_client(cfg.region)
    response = client.get_object(Bucket=cfg.bucket, Key=key)
    body = response["Body"]
    try:
        return body.read()
    finally:
        try:
            body.close()
        except Exception:
            pass


def s3_download_to_path(cfg: S3Config, *, key: str, path: str) -> None:
    client = _s3_client(cfg.region)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(cfg.bucket, key, str(target))


def s3_delete_key(cfg: S3Config, *, key: str) -> None:
    client = _s3_client(cfg.region)
    client.delete_object(Bucket=cfg.bucket, Key=key)


def s3_copy_object(src: S3Config, dst: S3Config, *, key: str) -> None:
    """Server-side copy of ``key`` from ``src`` bucket to ``dst`` bucket (same key).

    Used when an anonymous report is claimed on login: its evidence is moved from
    the temp (auto-expiring) bucket to the persistent one. Server-side copy avoids
    round-tripping the bytes through the dyno and preserves content-type/metadata.
    """
    client = _s3_client(dst.region or src.region)
    client.copy_object(
        Bucket=dst.bucket,
        Key=key,
        CopySource={"Bucket": src.bucket, "Key": key},
        MetadataDirective="COPY",
    )


def guess_content_type(path: str) -> Optional[str]:
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        return "application/pdf"
    if suffix == ".docx":
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    if suffix == ".txt":
        return "text/plain"
    if suffix == ".csv":
        return "text/csv"
    return None
