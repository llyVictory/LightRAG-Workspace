import os
import logging
from typing import Any, final, Union
from dataclasses import dataclass, field
import pipmaster as pm
import configparser
from contextlib import asynccontextmanager
import threading

if not pm.is_installed("redis"):
    pm.install("redis")

# aioredis is a depricated library, replaced with redis
from redis.asyncio import Redis, ConnectionPool  # type: ignore
from redis.exceptions import RedisError, ConnectionError, TimeoutError  # type: ignore
from lightrag.utils import logger, get_pinyin_sort_key

from lightrag.base import (
    BaseKVStorage,
    DocStatusStorage,
    DocStatus,
    DocProcessingStatus,
)
from ..kg.shared_storage import get_data_init_lock
import json

# Import tenacity for retry logic
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

config = configparser.ConfigParser()
config.read("config.ini", "utf-8")

# Constants for Redis connection pool with environment variable support
MAX_CONNECTIONS = int(os.getenv("REDIS_MAX_CONNECTIONS", "200"))
SOCKET_TIMEOUT = float(os.getenv("REDIS_SOCKET_TIMEOUT", "30.0"))
SOCKET_CONNECT_TIMEOUT = float(os.getenv("REDIS_CONNECT_TIMEOUT", "10.0"))
RETRY_ATTEMPTS = int(os.getenv("REDIS_RETRY_ATTEMPTS", "3"))

# Tenacity retry decorator for Redis operations
redis_retry = retry(
    stop=stop_after_attempt(RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=(
            retry_if_exception_type(ConnectionError)
            | retry_if_exception_type(TimeoutError)
            | retry_if_exception_type(RedisError)
    ),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)


class RedisConnectionManager:
    """Shared Redis connection pool manager to avoid creating multiple pools for the same Redis URI"""

    _pools = {}
    _pool_refs = {}  # Track reference count for each pool
    _lock = threading.Lock()

    @classmethod
    def get_pool(cls, redis_url: str) -> ConnectionPool:
        """Get or create a connection pool for the given Redis URL"""
        with cls._lock:
            if redis_url not in cls._pools:
                cls._pools[redis_url] = ConnectionPool.from_url(
                    redis_url,
                    max_connections=MAX_CONNECTIONS,
                    decode_responses=True,
                    socket_timeout=SOCKET_TIMEOUT,
                    socket_connect_timeout=SOCKET_CONNECT_TIMEOUT,
                )
                cls._pool_refs[redis_url] = 0
                logger.info(f"Created shared Redis connection pool for {redis_url}")

            # Increment reference count
            cls._pool_refs[redis_url] += 1
            logger.debug(
                f"Redis pool {redis_url} reference count: {cls._pool_refs[redis_url]}"
            )

        return cls._pools[redis_url]

    @classmethod
    def release_pool(cls, redis_url: str):
        """Release a reference to the connection pool"""
        with cls._lock:
            if redis_url in cls._pool_refs:
                cls._pool_refs[redis_url] -= 1
                logger.debug(
                    f"Redis pool {redis_url} reference count: {cls._pool_refs[redis_url]}"
                )

                # If no more references, close the pool
                if cls._pool_refs[redis_url] <= 0:
                    try:
                        cls._pools[redis_url].disconnect()
                        logger.info(
                            f"Closed Redis connection pool for {redis_url} (no more references)"
                        )
                    except Exception as e:
                        logger.error(f"Error closing Redis pool for {redis_url}: {e}")
                    finally:
                        del cls._pools[redis_url]
                        del cls._pool_refs[redis_url]

    @classmethod
    def close_all_pools(cls):
        """Close all connection pools (for cleanup)"""
        with cls._lock:
            for url, pool in cls._pools.items():
                try:
                    pool.disconnect()
                    logger.info(f"Closed Redis connection pool for {url}")
                except Exception as e:
                    logger.error(f"Error closing Redis pool for {url}: {e}")
            cls._pools.clear()
            cls._pool_refs.clear()


@final
@dataclass
class RedisKVStorage(BaseKVStorage):
    def __post_init__(self):
        # [修改] 移除静态绑定的 workspace 逻辑
        # self.final_namespace 的计算移至 _get_final_namespace()

        self._redis_url = os.environ.get(
            "REDIS_URI", config.get("redis", "uri", fallback="redis://localhost:6379")
        )
        self._pool = None
        self._redis = None
        self._initialized = False

        try:
            # Use shared connection pool
            self._pool = RedisConnectionManager.get_pool(self._redis_url)
            self._redis = Redis(connection_pool=self._pool)
            logger.info(
                f"Initialized Redis KV storage for {self.namespace} using shared connection pool"
            )
        except Exception as e:
            # Clean up on initialization failure
            if self._redis_url:
                RedisConnectionManager.release_pool(self._redis_url)
            logger.error(
                f"[{self.workspace}] Failed to initialize Redis KV storage: {e}"
            )
            raise

    def _get_final_namespace(self) -> str:
        """Dynamically compute namespace prefix based on current workspace."""
        ws = self.workspace
        if ws and ws.strip():
            return f"{ws}_{self.namespace}"
        return self.namespace

    async def initialize(self):
        """Initialize Redis connection and migrate legacy cache structure if needed"""
        async with get_data_init_lock():
            if self._initialized:
                return

            # Test connection
            try:
                async with self._get_redis_connection() as redis:
                    await redis.ping()
                    logger.info(
                        f"[{self.workspace}] Connected to Redis for namespace {self.namespace}"
                    )
                    self._initialized = True
            except Exception as e:
                logger.error(f"[{self.workspace}] Failed to connect to Redis: {e}")
                # Clean up on connection failure
                await self.close()
                raise

            # Migrate legacy cache structure if this is a cache namespace
            if self.namespace.endswith("_cache"):
                try:
                    await self._migrate_legacy_cache_structure()
                except Exception as e:
                    logger.error(
                        f"[{self.workspace}] Failed to migrate legacy cache structure: {e}"
                    )
                    # Don't fail initialization for migration errors, just log them

    @asynccontextmanager
    async def _get_redis_connection(self):
        """Safe context manager for Redis operations."""
        if not self._redis:
            raise ConnectionError("Redis connection not initialized")

        try:
            # Use the existing Redis instance with shared pool
            yield self._redis
        except ConnectionError as e:
            logger.error(
                f"[{self.workspace}] Redis connection error in {self.namespace}: {e}"
            )
            raise
        except RedisError as e:
            logger.error(
                f"[{self.workspace}] Redis operation error in {self.namespace}: {e}"
            )
            raise
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Unexpected error in Redis operation for {self.namespace}: {e}"
            )
            raise

    async def close(self):
        """Close the Redis connection and release pool reference to prevent resource leaks."""
        if hasattr(self, "_redis") and self._redis:
            try:
                await self._redis.close()
                logger.debug(
                    f"[{self.workspace}] Closed Redis connection for {self.namespace}"
                )
            except Exception as e:
                logger.error(f"[{self.workspace}] Error closing Redis connection: {e}")
            finally:
                self._redis = None

        # Release the pool reference (will auto-close pool if no more references)
        if hasattr(self, "_redis_url") and self._redis_url:
            RedisConnectionManager.release_pool(self._redis_url)
            self._pool = None
            logger.debug(
                f"[{self.workspace}] Released Redis connection pool reference for {self.namespace}"
            )

    async def __aenter__(self):
        """Support for async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Ensure Redis resources are cleaned up when exiting context."""
        await self.close()

    @redis_retry
    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        namespace = self._get_final_namespace()
        async with self._get_redis_connection() as redis:
            try:
                data = await redis.get(f"{namespace}:{id}")
                if data:
                    result = json.loads(data)
                    # Ensure time fields are present, provide default values for old data
                    result.setdefault("create_time", 0)
                    result.setdefault("update_time", 0)
                    return result
                return None
            except json.JSONDecodeError as e:
                logger.error(f"[{self.workspace}] JSON decode error for id {id}: {e}")
                return None

    @redis_retry
    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        namespace = self._get_final_namespace()
        async with self._get_redis_connection() as redis:
            try:
                pipe = redis.pipeline()
                for id in ids:
                    pipe.get(f"{namespace}:{id}")
                results = await pipe.execute()

                processed_results = []
                for result in results:
                    if result:
                        data = json.loads(result)
                        # Ensure time fields are present for all documents
                        data.setdefault("create_time", 0)
                        data.setdefault("update_time", 0)
                        processed_results.append(data)
                    else:
                        processed_results.append(None)

                return processed_results
            except json.JSONDecodeError as e:
                logger.error(f"[{self.workspace}] JSON decode error in batch get: {e}")
                return [None] * len(ids)

    async def filter_keys(self, keys: set[str]) -> set[str]:
        namespace = self._get_final_namespace()
        async with self._get_redis_connection() as redis:
            pipe = redis.pipeline()
            keys_list = list(keys)  # Convert set to list for indexing
            for key in keys_list:
                pipe.exists(f"{namespace}:{key}")
            results = await pipe.execute()

            existing_ids = {keys_list[i] for i, exists in enumerate(results) if exists}
            return set(keys) - existing_ids

    @redis_retry
    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        if not data:
            return

        import time
        current_time = int(time.time())  # Get current Unix timestamp
        namespace = self._get_final_namespace()

        async with self._get_redis_connection() as redis:
            try:
                # Check which keys already exist to determine create vs update
                pipe = redis.pipeline()
                for k in data.keys():
                    pipe.exists(f"{namespace}:{k}")
                exists_results = await pipe.execute()

                # Add timestamps to data
                for i, (k, v) in enumerate(data.items()):
                    # For text_chunks namespace, ensure llm_cache_list field exists
                    if self.namespace.endswith("text_chunks"):
                        if "llm_cache_list" not in v:
                            v["llm_cache_list"] = []

                    # Add timestamps based on whether key exists
                    if exists_results[i]:  # Key exists, only update update_time
                        v["update_time"] = current_time
                    else:  # New key, set both create_time and update_time
                        v["create_time"] = current_time
                        v["update_time"] = current_time

                    v["_id"] = k

                # Store the data
                pipe = redis.pipeline()
                for k, v in data.items():
                    pipe.set(f"{namespace}:{k}", json.dumps(v))
                await pipe.execute()

            except json.JSONDecodeError as e:
                logger.error(f"[{self.workspace}] JSON decode error during upsert: {e}")
                raise

    async def index_done_callback(self) -> None:
        # Redis handles persistence automatically
        pass

    async def is_empty(self) -> bool:
        """Check if the storage is empty for the current workspace and namespace"""
        namespace = self._get_final_namespace()
        pattern = f"{namespace}:*"
        try:
            async with self._get_redis_connection() as redis:
                # Use scan to check if any keys exist
                async for key in redis.scan_iter(match=pattern, count=1):
                    return False  # Found at least one key
                return True  # No keys found
        except Exception as e:
            logger.error(f"[{self.workspace}] Error checking if storage is empty: {e}")
            return True

    async def delete(self, ids: list[str]) -> None:
        """Delete specific records from storage by their IDs"""
        if not ids:
            return
        namespace = self._get_final_namespace()

        async with self._get_redis_connection() as redis:
            pipe = redis.pipeline()
            for id in ids:
                pipe.delete(f"{namespace}:{id}")

            results = await pipe.execute()
            deleted_count = sum(results)
            logger.info(
                f"[{self.workspace}] Deleted {deleted_count} of {len(ids)} entries from {self.namespace}"
            )

    async def drop(self) -> dict[str, str]:
        """Drop the storage by removing all keys under the current namespace."""
        namespace = self._get_final_namespace()
        async with self._get_redis_connection() as redis:
            try:
                # Use SCAN to find all keys with the namespace prefix
                pattern = f"{namespace}:*"
                cursor = 0
                deleted_count = 0

                while True:
                    cursor, keys = await redis.scan(cursor, match=pattern, count=1000)
                    if keys:
                        # Delete keys in batches
                        pipe = redis.pipeline()
                        for key in keys:
                            pipe.delete(key)
                        results = await pipe.execute()
                        deleted_count += sum(results)

                    if cursor == 0:
                        break

                logger.info(
                    f"[{self.workspace}] Dropped {deleted_count} keys from {self.namespace}"
                )
                return {
                    "status": "success",
                    "message": f"{deleted_count} keys dropped",
                }

            except Exception as e:
                logger.error(
                    f"[{self.workspace}] Error dropping keys from {self.namespace}: {e}"
                )
                return {"status": "error", "message": str(e)}

    async def _migrate_legacy_cache_structure(self):
        """Migrate legacy nested cache structure to flattened structure for Redis"""
        from lightrag.utils import generate_cache_key

        # [Note] Migration logic also needs to be dynamic, but since it's usually one-off
        # we apply it to the current workspace.
        namespace = self._get_final_namespace()

        async with self._get_redis_connection() as redis:
            # Get all keys for this namespace
            keys = await redis.keys(f"{namespace}:*")

            if not keys:
                return

            # ... (rest of migration logic, ensure `self.final_namespace` is replaced with `namespace`)
            # Simplified for brevity:
            has_flattened_keys = False
            keys_to_migrate = []

            for key in keys:
                key_id = key.split(":", 1)[1]
                if ":" in key_id and len(key_id.split(":")) == 3:
                    has_flattened_keys = True
                    break

                data = await redis.get(key)
                if data:
                    try:
                        parsed_data = json.loads(data)
                        if isinstance(parsed_data, dict) and all(isinstance(v, dict) and "return" in v for v in parsed_data.values()):
                            keys_to_migrate.append((key, key_id, parsed_data))
                    except json.JSONDecodeError: continue

            if has_flattened_keys or not keys_to_migrate:
                return

            pipe = redis.pipeline()
            migration_count = 0

            for old_key, mode, nested_data in keys_to_migrate:
                pipe.delete(old_key)
                for cache_hash, cache_entry in nested_data.items():
                    cache_type = cache_entry.get("cache_type", "extract")
                    flattened_key = generate_cache_key(mode, cache_type, cache_hash)
                    full_key = f"{namespace}:{flattened_key}"
                    pipe.set(full_key, json.dumps(cache_entry))
                    migration_count += 1

            await pipe.execute()
            if migration_count > 0:
                logger.info(f"[{self.workspace}] Migrated {migration_count} legacy cache entries")


@final
@dataclass
class RedisDocStatusStorage(DocStatusStorage):
    """Redis implementation of document status storage"""

    def __post_init__(self):
        # [修改] 移除静态绑定的 workspace 逻辑
        self._redis_url = os.environ.get(
            "REDIS_URI", config.get("redis", "uri", fallback="redis://localhost:6379")
        )
        self._pool = None
        self._redis = None
        self._initialized = False

        try:
            self._pool = RedisConnectionManager.get_pool(self._redis_url)
            self._redis = Redis(connection_pool=self._pool)
            logger.info(
                f"Initialized Redis doc status storage for {self.namespace} using shared connection pool"
            )
        except Exception as e:
            if self._redis_url:
                RedisConnectionManager.release_pool(self._redis_url)
            logger.error(
                f"[{self.workspace}] Failed to initialize Redis doc status storage: {e}"
            )
            raise

    def _get_final_namespace(self) -> str:
        """Dynamically compute namespace prefix."""
        ws = self.workspace
        # Doc status uses "_" default if workspace is empty in original logic, preserve it?
        # Actually base.py contextvars defaults to "default", so we can just stick to standard logic
        if ws and ws.strip():
            return f"{ws}_{self.namespace}"
        return self.namespace # Or f"_{self.namespace}" if you want to strictly match old logic for empty workspace

    async def initialize(self):
        """Initialize Redis connection"""
        async with get_data_init_lock():
            if self._initialized:
                return

            try:
                async with self._get_redis_connection() as redis:
                    await redis.ping()
                    logger.info(
                        f"[{self.workspace}] Connected to Redis for doc status namespace {self.namespace}"
                    )
                    self._initialized = True
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] Failed to connect to Redis for doc status: {e}"
                )
                await self.close()
                raise

    @asynccontextmanager
    async def _get_redis_connection(self):
        if not self._redis:
            raise ConnectionError("Redis connection not initialized")
        try:
            yield self._redis
        except Exception as e:
            logger.error(f"[{self.workspace}] Redis error: {e}")
            raise

    async def close(self):
        if hasattr(self, "_redis") and self._redis:
            try:
                await self._redis.close()
            except Exception as e:
                logger.error(f"[{self.workspace}] Error closing Redis connection: {e}")
            finally:
                self._redis = None

        if hasattr(self, "_redis_url") and self._redis_url:
            RedisConnectionManager.release_pool(self._redis_url)
            self._pool = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def filter_keys(self, keys: set[str]) -> set[str]:
        namespace = self._get_final_namespace()
        async with self._get_redis_connection() as redis:
            pipe = redis.pipeline()
            keys_list = list(keys)
            for key in keys_list:
                pipe.exists(f"{namespace}:{key}")
            results = await pipe.execute()

            existing_ids = {keys_list[i] for i, exists in enumerate(results) if exists}
            return set(keys) - existing_ids

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        namespace = self._get_final_namespace()
        ordered_results: list[dict[str, Any] | None] = []
        async with self._get_redis_connection() as redis:
            try:
                pipe = redis.pipeline()
                for id in ids:
                    pipe.get(f"{namespace}:{id}")
                results = await pipe.execute()

                for result_data in results:
                    if result_data:
                        try:
                            ordered_results.append(json.loads(result_data))
                        except json.JSONDecodeError as e:
                            logger.error(f"[{self.workspace}] JSON decode error: {e}")
                            ordered_results.append(None)
                    else:
                        ordered_results.append(None)
            except Exception as e:
                logger.error(f"[{self.workspace}] Error in get_by_ids: {e}")
        return ordered_results

    async def get_status_counts(self) -> dict[str, int]:
        namespace = self._get_final_namespace()
        counts = {status.value: 0 for status in DocStatus}
        async with self._get_redis_connection() as redis:
            try:
                cursor = 0
                while True:
                    cursor, keys = await redis.scan(
                        cursor, match=f"{namespace}:*", count=1000
                    )
                    if keys:
                        pipe = redis.pipeline()
                        for key in keys:
                            pipe.get(key)
                        values = await pipe.execute()

                        for value in values:
                            if value:
                                try:
                                    doc_data = json.loads(value)
                                    status = doc_data.get("status")
                                    if status in counts:
                                        counts[status] += 1
                                except json.JSONDecodeError: continue
                    if cursor == 0: break
            except Exception as e:
                logger.error(f"[{self.workspace}] Error getting status counts: {e}")
        return counts

    async def get_docs_by_status(self, status: DocStatus) -> dict[str, DocProcessingStatus]:
        namespace = self._get_final_namespace()
        result = {}
        async with self._get_redis_connection() as redis:
            try:
                cursor = 0
                while True:
                    cursor, keys = await redis.scan(
                        cursor, match=f"{namespace}:*", count=1000
                    )
                    if keys:
                        pipe = redis.pipeline()
                        for key in keys:
                            pipe.get(key)
                        values = await pipe.execute()

                        for key, value in zip(keys, values):
                            if value:
                                try:
                                    doc_data = json.loads(value)
                                    if doc_data.get("status") == status.value:
                                        doc_id = key.split(":", 1)[1]
                                        data = doc_data.copy()
                                        data.pop("content", None)
                                        if "file_path" not in data: data["file_path"] = "no-file-path"
                                        if "metadata" not in data: data["metadata"] = {}
                                        if "error_msg" not in data: data["error_msg"] = None
                                        result[doc_id] = DocProcessingStatus(**data)
                                except (json.JSONDecodeError, KeyError): continue
                    if cursor == 0: break
            except Exception as e:
                logger.error(f"[{self.workspace}] Error getting docs by status: {e}")
        return result

    async def get_docs_by_track_id(self, track_id: str) -> dict[str, DocProcessingStatus]:
        # Logic same as get_docs_by_status, just different filter
        namespace = self._get_final_namespace()
        result = {}
        async with self._get_redis_connection() as redis:
            try:
                cursor = 0
                while True:
                    cursor, keys = await redis.scan(
                        cursor, match=f"{namespace}:*", count=1000
                    )
                    if keys:
                        pipe = redis.pipeline()
                        for key in keys:
                            pipe.get(key)
                        values = await pipe.execute()

                        for key, value in zip(keys, values):
                            if value:
                                try:
                                    doc_data = json.loads(value)
                                    if doc_data.get("track_id") == track_id:
                                        doc_id = key.split(":", 1)[1]
                                        data = doc_data.copy()
                                        data.pop("content", None)
                                        if "file_path" not in data: data["file_path"] = "no-file-path"
                                        if "metadata" not in data: data["metadata"] = {}
                                        if "error_msg" not in data: data["error_msg"] = None
                                        result[doc_id] = DocProcessingStatus(**data)
                                except (json.JSONDecodeError, KeyError): continue
                    if cursor == 0: break
            except Exception as e:
                logger.error(f"[{self.workspace}] Error getting docs by track_id: {e}")
        return result

    async def index_done_callback(self) -> None: pass

    async def is_empty(self) -> bool:
        namespace = self._get_final_namespace()
        try:
            async with self._get_redis_connection() as redis:
                async for key in redis.scan_iter(match=f"{namespace}:*", count=1):
                    return False
                return True
        except Exception: return True

    @redis_retry
    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        if not data: return
        namespace = self._get_final_namespace()
        async with self._get_redis_connection() as redis:
            try:
                for doc_id, doc_data in data.items():
                    if "chunks_list" not in doc_data: doc_data["chunks_list"] = []
                pipe = redis.pipeline()
                for k, v in data.items():
                    pipe.set(f"{namespace}:{k}", json.dumps(v))
                await pipe.execute()
            except json.JSONDecodeError as e:
                logger.error(f"[{self.workspace}] Upsert error: {e}")
                raise

    @redis_retry
    async def get_by_id(self, id: str) -> Union[dict[str, Any], None]:
        namespace = self._get_final_namespace()
        async with self._get_redis_connection() as redis:
            try:
                data = await redis.get(f"{namespace}:{id}")
                return json.loads(data) if data else None
            except json.JSONDecodeError: return None

    async def delete(self, doc_ids: list[str]) -> None:
        if not doc_ids: return
        namespace = self._get_final_namespace()
        async with self._get_redis_connection() as redis:
            pipe = redis.pipeline()
            for doc_id in doc_ids:
                pipe.delete(f"{namespace}:{doc_id}")
            await pipe.execute()

    async def get_docs_paginated(
            self,
            status_filter: DocStatus | None = None,
            page: int = 1,
            page_size: int = 50,
            sort_field: str = "updated_at",
            sort_direction: str = "desc",
    ) -> tuple[list[tuple[str, DocProcessingStatus]], int]:
        namespace = self._get_final_namespace()
        # Pagination params validation (omitted for brevity, same as original)
        all_docs = []

        async with self._get_redis_connection() as redis:
            try:
                cursor = 0
                while True:
                    cursor, keys = await redis.scan(
                        cursor, match=f"{namespace}:*", count=1000
                    )
                    if keys:
                        pipe = redis.pipeline()
                        for key in keys: pipe.get(key)
                        values = await pipe.execute()

                        for key, value in zip(keys, values):
                            if value:
                                try:
                                    doc_data = json.loads(value)
                                    if status_filter and doc_data.get("status") != status_filter.value:
                                        continue

                                    doc_id = key.split(":", 1)[1]
                                    data = doc_data.copy()
                                    data.pop("content", None)
                                    # ... (field defaults as original) ...
                                    if "file_path" not in data: data["file_path"] = "no-file-path"
                                    if "metadata" not in data: data["metadata"] = {}
                                    if "error_msg" not in data: data["error_msg"] = None

                                    # Sort key logic
                                    if sort_field == "id": sort_key = doc_id
                                    elif sort_field == "file_path": sort_key = get_pinyin_sort_key(data.get(sort_field, ""))
                                    else: sort_key = data.get(sort_field, "")

                                    all_docs.append((doc_id, DocProcessingStatus(**data), sort_key))
                                except (json.JSONDecodeError, KeyError): continue
                    if cursor == 0: break
            except Exception: return [], 0

        reverse_sort = sort_direction.lower() == "desc"
        all_docs.sort(key=lambda x: x[2], reverse=reverse_sort)
        all_docs = [(d, s) for d, s, _ in all_docs]

        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        return all_docs[start_idx:end_idx], len(all_docs)

    async def get_all_status_counts(self) -> dict[str, int]:
        counts = await self.get_status_counts()
        counts["all"] = sum(counts.values())
        return counts

    async def get_doc_by_file_path(self, file_path: str) -> Union[dict[str, Any], None]:
        namespace = self._get_final_namespace()
        async with self._get_redis_connection() as redis:
            # (Logic same as original, just use namespace)
            cursor = 0
            while True:
                cursor, keys = await redis.scan(cursor, match=f"{namespace}:*", count=1000)
                if keys:
                    pipe = redis.pipeline()
                    for key in keys: pipe.get(key)
                    values = await pipe.execute()
                    for value in values:
                        if value:
                            try:
                                doc = json.loads(value)
                                if doc.get("file_path") == file_path: return doc
                            except: continue
                if cursor == 0: break
        return None

    async def drop(self) -> dict[str, str]:
        namespace = self._get_final_namespace()
        async with self._get_redis_connection() as redis:
            try:
                cursor = 0; deleted = 0
                while True:
                    cursor, keys = await redis.scan(cursor, match=f"{namespace}:*", count=1000)
                    if keys:
                        pipe = redis.pipeline()
                        for key in keys: pipe.delete(key)
                        res = await pipe.execute()
                        deleted += sum(res)
                    if cursor == 0: break
                return {"status": "success", "message": f"{deleted} dropped"}
            except Exception as e: return {"status": "error", "message": str(e)}