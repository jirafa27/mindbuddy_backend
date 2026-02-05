"""Одна реализация: MinIOStorage реализует BlobStorage и FileStorage (два бакета)."""
import asyncio
import io
import logging
import pickle
import uuid
from datetime import datetime
from typing import Any, Optional

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from app.core.config import settings
from app.domain.protocols import BlobStorage, FileStorage

logger = logging.getLogger(__name__)

# Константа для временного бакета blob'ов (Claim Check pattern)
TEMP_BLOB_BUCKET = settings.MINIO_BUCKET_TEMP_BLOBS  # "temp-blobs"


class MinIOStorage(BlobStorage, FileStorage):
    def __init__(
        self,
        bucket: Optional[str] = None,
        blob_bucket: Optional[str] = None,
    ):
        self.client = boto3.client(
            "s3",
            endpoint_url=f"http://{settings.MINIO_ENDPOINT}" if not settings.MINIO_SECURE else f"https://{settings.MINIO_ENDPOINT}",
            aws_access_key_id=settings.MINIO_ACCESS_KEY,
            aws_secret_access_key=settings.MINIO_SECRET_KEY,
            config=Config(signature_version="s3v4"),
            region_name="us-east-1",
        )
        self.bucket_name = bucket or settings.MINIO_BUCKET_NAME
        self._blob_bucket_name = blob_bucket or TEMP_BLOB_BUCKET
        self._ensure_bucket_exists(self.bucket_name)
        self._ensure_bucket_exists(self._blob_bucket_name)
        self._setup_blob_bucket_lifecycle()

    @property
    def blob_bucket_name(self) -> str:
        """Имя бакета для blob'ов (Claim Check)."""
        return self._blob_bucket_name

    def _ensure_bucket_exists(self, bucket_name: str) -> None:
        try:
            self.client.head_bucket(Bucket=bucket_name)
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            if error_code == "404":
                try:
                    self.client.create_bucket(Bucket=bucket_name)
                    logger.info("Bucket created: %s", bucket_name)
                except ClientError:
                    pass

    def _setup_blob_bucket_lifecycle(self) -> None:
        """Настраивает auto-cleanup для temp-blobs: удаление через 1 день."""
        lifecycle_config = {
            "Rules": [
                {
                    "ID": "DeleteTempBlobsAfter1Day",
                    "Status": "Enabled",
                    "Filter": {"Prefix": ""},
                    "Expiration": {"Days": 1},
                }
            ]
        }
        try:
            self.client.put_bucket_lifecycle_configuration(
                Bucket=self.blob_bucket_name,
                LifecycleConfiguration=lifecycle_config,
            )
            logger.debug("Lifecycle policy set for bucket=%s", self.blob_bucket_name)
        except ClientError as e:
            # MinIO может не поддерживать lifecycle в некоторых конфигурациях
            logger.warning("Failed to set lifecycle for bucket=%s: %s", self.blob_bucket_name, e)

    # --- BlobStorage (Claim Check, бакет blob_bucket) ---

    async def put_blob(self, data: Any) -> str:
        key = uuid.uuid4().hex
        raw = pickle.dumps(data)
        logger.info("put_blob: uploading key=%s to bucket=%s", key, self.blob_bucket_name)
        await asyncio.to_thread(
            self.client.upload_fileobj,
            io.BytesIO(raw),
            self.blob_bucket_name,
            key,
        )
        # Убеждаемся, что объект виден (eventual consistency): иначе DBAgent получит NoSuchKey
        for attempt in range(5):
            try:
                await asyncio.to_thread(
                    self.client.head_object,
                    Bucket=self.blob_bucket_name,
                    Key=key,
                )
                logger.info("put_blob: key=%s verified in bucket=%s", key, self.blob_bucket_name)
                return key
            except ClientError as e:
                if e.response.get("Error", {}).get("Code") != "404":
                    raise
                if attempt < 4:
                    wait_time = 0.5 * (attempt + 1)
                    logger.warning(
                        "put_blob: key=%s not visible yet in bucket=%s, retry %s/5, waiting %.1fs",
                        key, self.blob_bucket_name, attempt + 1, wait_time,
                    )
                    await asyncio.sleep(wait_time)
                    continue
                raise RuntimeError(
                    f"put_blob: object {key} not visible in {self.blob_bucket_name} after 5 checks"
                )
        # Не должны сюда попасть, но на всякий случай
        raise RuntimeError(f"put_blob: object {key} not visible in {self.blob_bucket_name}")

    async def get_blob(self, key: str) -> Any:
        logger.info("get_blob: fetching key=%s from bucket=%s", key, self.blob_bucket_name)

        def _get():
            response = self.client.get_object(Bucket=self.blob_bucket_name, Key=key)
            try:
                raw = response["Body"].read()
                if not raw:
                    raise ValueError(f"get_blob: empty content for key={key}")
                return pickle.loads(raw)
            finally:
                response["Body"].close()

        # Retry с 5 попытками и интервалом 1 секунда
        last_error: Optional[Exception] = None
        for attempt in range(5):
            try:
                result = await asyncio.to_thread(_get)
                logger.info("get_blob: successfully fetched key=%s", key)
                return result
            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code")
                if error_code == "NoSuchKey":
                    last_error = e
                    if attempt < 4:
                        logger.warning(
                            "get_blob: NoSuchKey for key=%s bucket=%s, retry %s/5, waiting 1s",
                            key, self.blob_bucket_name, attempt + 1,
                        )
                        await asyncio.sleep(1.0)
                        continue
                raise
            except Exception as e:
                # Для других ошибок (ValueError, pickle и т.д.) — не ретраим
                raise

        # Если дошли сюда — все 5 попыток NoSuchKey
        raise last_error or RuntimeError(f"get_blob: key={key} not found after 5 retries")

    async def delete_blob(self, key: str) -> None:
        await asyncio.to_thread(
            self.client.delete_object,
            Bucket=self.blob_bucket_name,
            Key=key,
        )

    # --- FileStorage (постоянные файлы, бакет bucket) ---

    @staticmethod
    def generate_object_name(
        user_id: Optional[int] = None,
        namespace_id: Optional[int] = None,
        filename: Optional[str] = None,
    ) -> str:
        if user_id is None:
            return f"temp/{uuid.uuid4().hex}.pkl"
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        ns = namespace_id if namespace_id is not None else 0
        name = filename if filename else "file"
        return f"users/{user_id}/namespaces/{ns}/{timestamp}_{name}"

    async def upload_file(
        self,
        file_content: bytes,
        object_name: str,
        content_type: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        extra_args = {}
        if content_type:
            extra_args["ContentType"] = content_type
        if metadata:
            extra_args["Metadata"] = metadata
        await asyncio.to_thread(
            self.client.upload_fileobj,
            io.BytesIO(file_content),
            self.bucket_name,
            object_name,
            extra_args or None,
        )
        return object_name

    async def download_file(self, object_name: str) -> bytes:
        def _get():
            response = self.client.get_object(Bucket=self.bucket_name, Key=object_name)
            try:
                return response["Body"].read()
            finally:
                response["Body"].close()

        return await asyncio.to_thread(_get)

    async def delete_file(self, object_name: str) -> None:
        await asyncio.to_thread(
            self.client.delete_object,
            Bucket=self.bucket_name,
            Key=object_name,
        )

    def get_file_url(self, object_name: str, expires_in: int = 3600) -> str:
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket_name, "Key": object_name},
            ExpiresIn=expires_in,
        )
