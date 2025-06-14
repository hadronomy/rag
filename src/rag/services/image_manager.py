import asyncio
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Union

from aiobotocore.session import get_session
from botocore.exceptions import ClientError, NoCredentialsError
from loguru import logger
from PIL import Image
from pydantic import UUID4


class S3ImageError(Exception):
    """Base exception for S3 image operations."""

    pass


class S3ImageNotFoundError(S3ImageError):
    """Raised when an image is not found in S3."""

    pass


class S3ImageUploadError(S3ImageError):
    """Raised when image upload fails."""

    pass


class S3ImageDownloadError(S3ImageError):
    """Raised when image download fails."""

    pass


class S3JPEGManager:
    """
    Asynchronous S3 manager for handling JPEG image uploads and downloads.

    This class provides methods to upload PIL Images as JPEG files to S3 and download
    images from S3 as bytes or instructor.Image objects. All operations are performed
    asynchronously for better performance.

    Supports context manager usage for automatic resource cleanup.

    Attributes:
        bucket_name (str): The S3 bucket name to operate on
        session: Aiobotocore session for S3 operations
        config (dict): AWS configuration including credentials and region
    """

    def __init__(
        self,
        bucket_name: str,
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
        region_name: str = "us-east-1",
        endpoint_url: Optional[str] = None,
        jpeg_quality: int = 95,
        max_retries: int = 3,
    ):
        """
        Initialize the S3JPEGManager.

        Args:
            bucket_name (str): The name of the S3 bucket to use
            access_key_id (Optional[str]): AWS access key ID. If None, uses default credentials
            secret_access_key (Optional[str]): AWS secret access key. If None, uses default credentials
            region_name (str): AWS region name. Defaults to "us-east-1"
            endpoint_url (Optional[str]): Custom S3 endpoint URL for S3-compatible services
            jpeg_quality (int): JPEG compression quality (1-100). Defaults to 95
            max_retries (int): Maximum number of retry attempts for failed operations. Defaults to 3

        Raises:
            ValueError: If bucket_name is empty or jpeg_quality is out of range
        """
        if not bucket_name or not bucket_name.strip():
            raise ValueError("bucket_name cannot be empty")
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")

        self.bucket_name = bucket_name.strip()
        self.jpeg_quality = jpeg_quality
        self.max_retries = max_retries
        self.session = get_session()
        self.config = {
            "region_name": region_name,
            "aws_access_key_id": access_key_id,
            "aws_secret_access_key": secret_access_key,
            "endpoint_url": endpoint_url,
        }
        # Remove None values
        self.config = {k: v for k, v in self.config.items() if v is not None}
        self._client = None
        self._client_context = None

    async def __aenter__(self):
        """Async context manager entry."""
        logger.debug("Entering S3JPEGManager context manager")
        await self._ensure_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        logger.debug("Exiting S3JPEGManager context manager")
        await self.close()

    async def _ensure_client(self):
        """Ensure S3 client is created and available."""
        if self._client is None:
            try:
                logger.debug("Creating S3 client with bucket: {}", self.bucket_name)
                client_context = self.session.create_client("s3", **self.config)
                self._client = await client_context.__aenter__()
                self._client_context = client_context
                logger.info(
                    "S3 client created successfully for bucket: {}", self.bucket_name
                )
            except NoCredentialsError as e:
                logger.error("AWS credentials not found: {}", e)
                raise S3ImageError(f"AWS credentials not found: {e}")

    async def close(self):
        """Close the S3 client and clean up resources."""
        if hasattr(self, "_client_context") and self._client_context:
            logger.debug("Closing S3 client")
            await self._client_context.__aexit__(None, None, None)
            self._client = None
            self._client_context = None
            logger.debug("S3 client closed successfully")

    @asynccontextmanager
    async def _get_client(self):
        """Get S3 client as context manager."""
        if self._client:
            yield self._client
        else:
            async with self.session.create_client("s3", **self.config) as client:
                yield client

    async def _retry_operation(self, operation, *args, **kwargs):
        """Retry an operation with exponential backoff."""
        last_exception = None
        for attempt in range(self.max_retries + 1):
            try:
                return await operation(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries:
                    wait_time = 2**attempt
                    logger.warning(
                        f"Operation failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    break
        raise last_exception

    # Download methods
    async def download_image(self, filename: str) -> bytes:
        """
        Download a single image from S3 as bytes.

        Args:
            filename (str): The S3 key/filename of the image to download

        Returns:
            bytes: The raw image data as bytes

        Raises:
            S3ImageNotFoundError: If the image doesn't exist
            S3ImageDownloadError: If download fails for other reasons
        """
        if not filename or not filename.strip():
            raise ValueError("filename cannot be empty")

        async def _download():
            async with self._get_client() as s3_client:
                try:
                    logger.debug("Downloading image: {}", filename.strip())
                    response = await s3_client.get_object(
                        Bucket=self.bucket_name, Key=filename.strip()
                    )
                    image_data = await response["Body"].read()
                    logger.info(
                        "Successfully downloaded image: {} ({} bytes)",
                        filename.strip(),
                        len(image_data),
                    )
                    return image_data
                except ClientError as e:
                    error_code = e.response["Error"]["Code"]
                    if error_code == "NoSuchKey":
                        logger.warning("Image not found in S3: {}", filename)
                        raise S3ImageNotFoundError(f"Image not found: {filename}")
                    else:
                        logger.error("Failed to download image {}: {}", filename, e)
                        raise S3ImageDownloadError(
                            f"Failed to download {filename}: {e}"
                        )

        return await self._retry_operation(_download)

    async def download_images(
        self, paths: List[str], ignore_missing: bool = False
    ) -> List[Optional[bytes]]:
        """
        Download multiple images from S3 concurrently.

        Args:
            paths (List[str]): List of S3 keys/filenames to download
            ignore_missing (bool): If True, return None for missing images instead of raising error

        Returns:
            List[Optional[bytes]]: List of raw image data as bytes or None for missing images

        Raises:
            S3ImageDownloadError: If any download fails and ignore_missing is False
        """
        if not paths:
            logger.debug("No paths provided for batch download")
            return []

        logger.info("Starting batch download of {} images", len(paths))

        async def _safe_download(path: str) -> Optional[bytes]:
            try:
                return await self.download_image(path)
            except S3ImageNotFoundError:
                if ignore_missing:
                    logger.warning(f"Image not found, skipping: {path}")
                    return None
                raise

        tasks = [_safe_download(path) for path in paths]
        results = await asyncio.gather(*tasks)
        successful_downloads = sum(1 for result in results if result is not None)
        logger.info(
            "Batch download completed: {}/{} images downloaded successfully",
            successful_downloads,
            len(paths),
        )
        return results

    async def download_pydantic_ai_images(
        self, filenames: List[str], ignore_missing: bool = False
    ) -> List[bytes]:
        """
        Download multiple images from S3 as bytes for pydantic-ai.

        Args:
            filenames (List[str]): List of S3 keys/filenames to download
            ignore_missing (bool): If True, skip missing images instead of raising error

        Returns:
            List[bytes]: List of image bytes ready for use with pydantic-ai BinaryContent

        Raises:
            S3ImageDownloadError: If any download fails and ignore_missing is False
        """
        images_bytes = await self.download_images(
            paths=filenames, ignore_missing=ignore_missing
        )
        filtered_images = [
            img_bytes for img_bytes in images_bytes if img_bytes is not None
        ]
        logger.info(
            "Prepared {} images for pydantic-ai from {} requested",
            len(filtered_images),
            len(filenames),
        )
        return filtered_images

    # Upload methods
    async def _upload_image(
        self, key: str, image: Image.Image, metadata: Optional[Dict[str, str]] = None
    ):
        """
        Internal method to upload a single PIL Image to S3 as JPEG.

        Args:
            key (str): S3 key for the image
            image (Image.Image): PIL Image object to upload
            metadata (Optional[Dict[str, str]]): Optional metadata to attach to the object

        Raises:
            S3ImageUploadError: If upload fails
        """

        async def _upload():
            async with self._get_client() as s3_client:
                try:
                    logger.debug("Uploading image to S3 key: {}", key)
                    with BytesIO() as buffer:
                        # Convert to RGB if necessary (for JPEG compatibility)
                        if image.mode in ("RGBA", "LA", "P"):
                            logger.debug(
                                "Converting image from {} to RGB for JPEG compatibility",
                                image.mode,
                            )
                            converted_image = image.convert("RGB")
                        else:
                            converted_image = image

                        converted_image.save(
                            buffer,
                            format="JPEG",
                            quality=self.jpeg_quality,
                            optimize=True,
                        )
                        buffer.seek(0)
                        image_size = len(buffer.getvalue())

                        put_args = {
                            "Bucket": self.bucket_name,
                            "Key": key,
                            "Body": buffer.getvalue(),
                            "ContentType": "image/jpeg",
                        }

                        if metadata:
                            put_args["Metadata"] = metadata
                            logger.debug("Adding metadata to upload: {}", metadata)

                        await s3_client.put_object(**put_args)
                        logger.info(
                            "Successfully uploaded image to {} ({} bytes, quality: {})",
                            key,
                            image_size,
                            self.jpeg_quality,
                        )
                except Exception as e:
                    logger.error("Failed to upload image to {}: {}", key, e)
                    raise S3ImageUploadError(f"Failed to upload image to {key}: {e}")

        await self._retry_operation(_upload)

    async def upload_images(
        self,
        session_id: UUID4,
        file_name: str,
        images: List[Image.Image],
        start: int = 1,
        metadata: Optional[Dict[str, str]] = None,
    ) -> List[str]:
        """
        Upload multiple PIL Images to S3 concurrently as JPEG files.

        Args:
            session_id (UUID4): Unique session identifier for organizing uploads
            file_name (str): Name of the file/document the images belong to
            images (List[Image.Image]): List of PIL Image objects to upload
            start (int): Starting page number for the images. Defaults to 1
            metadata (Optional[Dict[str, str]]): Optional metadata to attach to all images

        Returns:
            List[str]: List of S3 keys for the uploaded images

        Note:
            Images are uploaded with sequential page numbers starting from 'start'.
            All uploads are performed concurrently for better performance.
        """
        if not images:
            logger.warning("No images to upload")
            return []

        if not file_name or not file_name.strip():
            raise ValueError("file_name cannot be empty")

        logger.info(
            "Attempting to upload {n} images for {f} in session {s}",
            n=len(images),
            f=file_name,
            s=session_id,
        )

        keys = [
            f"{session_id}/{file_name.strip()}/{page}.jpeg"
            for page in range(start, start + len(images))
        ]

        tasks = [
            self._upload_image(key=key, image=image, metadata=metadata)
            for key, image in zip(keys, images)
        ]

        logger.debug("Starting concurrent upload of {} images", len(tasks))
        await asyncio.gather(*tasks)
        logger.success("Uploaded {n} images", n=len(images))
        return keys

    async def upload_single_image(
        self,
        key: str,
        image: Union[Image.Image, bytes, str, Path],
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Upload a single image to S3 with a custom key.

        Args:
            key (str): S3 key for the image
            image (Union[Image.Image, bytes, str, Path]): Image to upload
            metadata (Optional[Dict[str, str]]): Optional metadata to attach

        Returns:
            str: The S3 key of the uploaded image

        Raises:
            S3ImageUploadError: If upload fails
            ValueError: If image format is not supported
        """
        if not key or not key.strip():
            raise ValueError("key cannot be empty")

        # Convert input to PIL Image
        if isinstance(image, (str, Path)):
            logger.debug("Loading image from file path: {}", image)
            pil_image = Image.open(image)
        elif isinstance(image, bytes):
            logger.debug("Loading image from bytes ({} bytes)", len(image))
            pil_image = Image.open(BytesIO(image))
        elif isinstance(image, Image.Image):
            logger.debug("Using provided PIL Image object")
            pil_image = image
        else:
            logger.error("Unsupported image type: {}", type(image))
            raise ValueError(f"Unsupported image type: {type(image)}")

        await self._upload_image(key=key.strip(), image=pil_image, metadata=metadata)
        logger.info("Single image uploaded successfully to key: {}", key.strip())
        return key.strip()

    # Utility methods
    async def image_exists(self, key: str) -> bool:
        """
        Check if an image exists in S3.

        Args:
            key (str): S3 key to check

        Returns:
            bool: True if the image exists, False otherwise
        """
        try:
            async with self._get_client() as s3_client:
                logger.debug("Checking if image exists: {}", key)
                await s3_client.head_object(Bucket=self.bucket_name, Key=key)
                logger.debug("Image exists: {}", key)
                return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                logger.debug("Image does not exist: {}", key)
                return False
            logger.error("Failed to check if image exists {}: {}", key, e)
            raise S3ImageDownloadError(f"Failed to check if image exists: {e}")

    async def delete_image(self, key: str) -> bool:
        """
        Delete an image from S3.

        Args:
            key (str): S3 key of the image to delete

        Returns:
            bool: True if deleted successfully, False if image didn't exist

        Raises:
            S3ImageError: If deletion fails
        """
        try:
            async with self._get_client() as s3_client:
                logger.debug("Deleting image: {}", key)
                await s3_client.delete_object(Bucket=self.bucket_name, Key=key)
                logger.info("Successfully deleted image: {}", key)
                return True
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "NoSuchKey":
                logger.warning("Attempted to delete non-existent image: {}", key)
                return False
            logger.error("Failed to delete image {}: {}", key, e)
            raise S3ImageError(f"Failed to delete image {key}: {e}")

    async def list_images(self, prefix: str = "", max_keys: int = 1000) -> List[str]:
        """
        List images in the bucket with optional prefix filtering.

        Args:
            prefix (str): Prefix to filter by
            max_keys (int): Maximum number of keys to return

        Returns:
            List[str]: List of S3 keys matching the criteria
        """
        try:
            async with self._get_client() as s3_client:
                logger.debug(
                    "Listing images with prefix: '{}', max_keys: {}", prefix, max_keys
                )
                response = await s3_client.list_objects_v2(
                    Bucket=self.bucket_name, Prefix=prefix, MaxKeys=max_keys
                )
                keys = [obj["Key"] for obj in response.get("Contents", [])]
                logger.info("Found {} images with prefix '{}'", len(keys), prefix)
                return keys
        except ClientError as e:
            logger.error("Failed to list images with prefix '{}': {}", prefix, e)
            raise S3ImageError(f"Failed to list images: {e}")
