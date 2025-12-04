import os
import re
import time
from dataclasses import dataclass, field
import numpy as np
import configparser
import asyncio

from typing import Any, Union, final

from ..base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    DocProcessingStatus,
    DocStatus,
    DocStatusStorage,
)
from ..utils import logger, compute_mdhash_id
from ..types import KnowledgeGraph, KnowledgeGraphNode, KnowledgeGraphEdge
from ..constants import GRAPH_FIELD_SEP
from ..kg.shared_storage import get_data_init_lock

import pipmaster as pm

if not pm.is_installed("pymongo"):
    pm.install("pymongo")

from pymongo import AsyncMongoClient  # type: ignore
from pymongo import UpdateOne  # type: ignore
from pymongo.asynchronous.database import AsyncDatabase  # type: ignore
from pymongo.asynchronous.collection import AsyncCollection  # type: ignore
from pymongo.operations import SearchIndexModel  # type: ignore
from pymongo.errors import PyMongoError  # type: ignore

from pymongo.errors import CollectionInvalid, PyMongoError

config = configparser.ConfigParser()
config.read("config.ini", "utf-8")

GRAPH_BFS_MODE = os.getenv("MONGO_GRAPH_BFS_MODE", "bidirectional")


class ClientManager:
    _instances = {"db": None, "ref_count": 0}
    _lock = asyncio.Lock()

    @classmethod
    async def get_client(cls) -> AsyncMongoClient:
        async with cls._lock:
            if cls._instances["db"] is None:
                uri = os.environ.get(
                    "MONGO_URI",
                    config.get(
                        "mongodb",
                        "uri",
                        fallback="mongodb://root:root@localhost:27017/",
                    ),
                )
                database_name = os.environ.get(
                    "MONGO_DATABASE",
                    config.get("mongodb", "database", fallback="LightRAG"),
                )
                client = AsyncMongoClient(uri)
                db = client.get_database(database_name)
                cls._instances["db"] = db
                cls._instances["ref_count"] = 0
            cls._instances["ref_count"] += 1
            return cls._instances["db"]

    @classmethod
    async def release_client(cls, db: AsyncDatabase):
        async with cls._lock:
            if db is not None:
                if db is cls._instances["db"]:
                    cls._instances["ref_count"] -= 1
                    if cls._instances["ref_count"] == 0:
                        cls._instances["db"] = None


# --------------------------------------------------------------------------
# MongoKVStorage
# --------------------------------------------------------------------------
@final
@dataclass
class MongoKVStorage(BaseKVStorage):
    db: AsyncDatabase = field(default=None)
    # 缓存已加载的集合对象: {collection_name: AsyncCollection}
    _collections: dict = field(default_factory=dict, init=False)

    def __post_init__(self):
        # 移除静态 workspace 计算
        self._collections = {}

    def _get_collection_name(self) -> str:
        ws = self.workspace
        if ws and ws.strip():
            return f"{ws}_{self.namespace}"
        return self.namespace

    async def _get_collection(self) -> AsyncCollection:
        coll_name = self._get_collection_name()
        if coll_name in self._collections:
            return self._collections[coll_name]

        if self.db is None:
            self.db = await ClientManager.get_client()

        coll = await get_or_create_collection(self.db, coll_name)
        self._collections[coll_name] = coll
        return coll

    async def initialize(self):
        async with get_data_init_lock():
            if self.db is None:
                self.db = await ClientManager.get_client()
            # 不预加载集合，懒加载

    async def finalize(self):
        if self.db is not None:
            await ClientManager.release_client(self.db)
            self.db = None
            self._collections = {}

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        coll = await self._get_collection()
        doc = await coll.find_one({"_id": id})
        if doc:
            doc.setdefault("create_time", 0)
            doc.setdefault("update_time", 0)
        return doc

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        coll = await self._get_collection()
        cursor = coll.find({"_id": {"$in": ids}})
        docs = await cursor.to_list(length=None)
        doc_map = {str(d["_id"]): d for d in docs if d}

        for doc in doc_map.values():
            doc.setdefault("create_time", 0)
            doc.setdefault("update_time", 0)

        return [doc_map.get(str(i)) for i in ids]

    async def filter_keys(self, keys: set[str]) -> set[str]:
        coll = await self._get_collection()
        cursor = coll.find({"_id": {"$in": list(keys)}}, {"_id": 1})
        existing_ids = {str(x["_id"]) async for x in cursor}
        return keys - existing_ids

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        if not data: return
        coll = await self._get_collection()
        operations = []
        current_time = int(time.time())

        for k, v in data.items():
            if self.namespace.endswith("text_chunks") and "llm_cache_list" not in v:
                v["llm_cache_list"] = []

            v_for_set = v.copy()
            v_for_set["_id"] = k
            v_for_set["update_time"] = current_time
            v_for_set.pop("create_time", None)

            operations.append(UpdateOne(
                {"_id": k},
                {"$set": v_for_set, "$setOnInsert": {"create_time": current_time}},
                upsert=True
            ))

        if operations:
            await coll.bulk_write(operations)

    async def index_done_callback(self) -> None: pass

    async def is_empty(self) -> bool:
        try:
            coll = await self._get_collection()
            count = await coll.count_documents({}, limit=1)
            return count == 0
        except PyMongoError: return True

    async def delete(self, ids: list[str]) -> None:
        if not ids: return
        if isinstance(ids, set): ids = list(ids)
        coll = await self._get_collection()
        await coll.delete_many({"_id": {"$in": ids}})

    async def drop(self) -> dict[str, str]:
        coll = await self._get_collection()
        await coll.delete_many({})
        return {"status": "success", "message": "dropped"}

# --------------------------------------------------------------------------
# MongoDocStatusStorage
# --------------------------------------------------------------------------
@final
@dataclass
class MongoDocStatusStorage(DocStatusStorage):
    db: AsyncDatabase = field(default=None)
    _collections: dict = field(default_factory=dict, init=False)

    def _prepare_doc_status_data(self, doc: dict[str, Any]) -> dict[str, Any]:
        data = doc.copy()
        data.pop("content", None)
        data.pop("_id", None)
        if "file_path" not in data: data["file_path"] = "no-file-path"
        if "metadata" not in data: data["metadata"] = {}
        if "error_msg" not in data: data["error_msg"] = None
        if "error" in data:
            if not data.get("error_msg"): data["error_msg"] = data.pop("error")
            else: data.pop("error", None)
        return data

    def __post_init__(self):
        self._collections = {}

    def _get_collection_name(self) -> str:
        ws = self.workspace
        if ws and ws.strip():
            return f"{ws}_{self.namespace}"
        return self.namespace

    async def _get_collection(self) -> AsyncCollection:
        coll_name = self._get_collection_name()
        if coll_name in self._collections:
            return self._collections[coll_name]

        if self.db is None:
            self.db = await ClientManager.get_client()

        coll = await get_or_create_collection(self.db, coll_name)
        # Create indexes
        await self._create_indexes(coll, coll_name)
        self._collections[coll_name] = coll
        return coll

    async def _create_indexes(self, coll: AsyncCollection, coll_name: str):
        # 索引创建逻辑移到这里，针对具体 collection
        try:
            indexes = [
                {"keys": [("status", 1), ("updated_at", -1)]},
                {"keys": [("file_path", 1)], "collation": {"locale": "zh", "numericOrdering": True}}
                # ... 其他索引 ...
            ]
            for idx in indexes:
                kwargs = {"name": f"idx_{'_'.join(str(k[0]) for k in idx['keys'])}"}
                if "collation" in idx: kwargs["collation"] = idx["collation"]
                await coll.create_index(idx["keys"], **kwargs)
        except Exception as e:
            logger.warning(f"Index creation failed for {coll_name}: {e}")

    async def initialize(self):
        async with get_data_init_lock():
            if self.db is None:
                self.db = await ClientManager.get_client()

    async def get_by_id(self, id: str):
        coll = await self._get_collection()
        return await coll.find_one({"_id": id})

    async def get_by_ids(self, ids: list[str]):
        coll = await self._get_collection()
        cursor = coll.find({"_id": {"$in": ids}})
        docs = await cursor.to_list(length=None)
        doc_map = {str(d["_id"]): d for d in docs if d}
        return [doc_map.get(str(i)) for i in ids]

    async def filter_keys(self, data: set[str]) -> set[str]:
        coll = await self._get_collection()
        cursor = coll.find({"_id": {"$in": list(data)}}, {"_id": 1})
        existing = {str(x["_id"]) async for x in cursor}
        return data - existing

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        if not data: return
        coll = await self._get_collection()
        tasks = []
        for k, v in data.items():
            if "chunks_list" not in v: v["chunks_list"] = []
            v["_id"] = k
            tasks.append(coll.update_one({"_id": k}, {"$set": v}, upsert=True))
        await asyncio.gather(*tasks)


    async def get_status_counts(self) -> dict[str, int]:
        """Get counts of documents in each status"""
        pipeline = [{"$group": {"_id": "$status", "count": {"$sum": 1}}}]
        coll = await self._get_collection()
        cursor = await  coll.aggregate(pipeline, allowDiskUse=True)
        result = await cursor.to_list()
        counts = {}
        for doc in result:
            counts[doc["_id"]] = doc["count"]
        return counts

    async def get_docs_by_status(
        self, status: DocStatus
    ) -> dict[str, DocProcessingStatus]:
        """Get all documents with a specific status"""
        coll = await self._get_collection()
        cursor = coll.find({"status": status.value})
        result = await cursor.to_list()
        processed_result = {}
        for doc in result:
            try:
                data = self._prepare_doc_status_data(doc)
                processed_result[doc["_id"]] = DocProcessingStatus(**data)
            except KeyError as e:
                logger.error(
                    f"[{self.workspace}] Missing required field for document {doc['_id']}: {e}"
                )
                continue
        return processed_result

    async def get_docs_by_track_id(
        self, track_id: str
    ) -> dict[str, DocProcessingStatus]:
        """Get all documents with a specific track_id"""
        coll = await self._get_collection()
        cursor = coll.find({"track_id": track_id})
        result = await cursor.to_list()
        processed_result = {}
        for doc in result:
            try:
                data = self._prepare_doc_status_data(doc)
                processed_result[doc["_id"]] = DocProcessingStatus(**data)
            except KeyError as e:
                logger.error(
                    f"[{self.workspace}] Missing required field for document {doc['_id']}: {e}"
                )
                continue
        return processed_result

    async def index_done_callback(self) -> None:
        # Mongo handles persistence automatically
        pass

    async def is_empty(self) -> bool:
        """Check if the storage is empty for the current workspace and namespace

        Returns:
            bool: True if storage is empty, False otherwise
        """
        try:
            # Use count_documents with limit 1 for efficiency
            coll = await self._get_collection()
            count = await coll.count_documents({}, limit=1)
            return count == 0
        except PyMongoError as e:
            logger.error(f"[{self.workspace}] Error checking if storage is empty: {e}")
            return True

    async def drop(self) -> dict[str, str]:
        """Drop the storage by removing all documents in the collection.

        Returns:
            dict[str, str]: Status of the operation with keys 'status' and 'message'
        """
        try:
            coll = await self._get_collection()
            result = await coll.delete_many({})
            deleted_count = result.deleted_count

            logger.info(
                f"[{self.workspace}] Dropped {deleted_count} documents from doc status {self._collection_name}"
            )
            return {
                "status": "success",
                "message": f"{deleted_count} documents dropped",
            }
        except PyMongoError as e:
            logger.error(
                f"[{self.workspace}] Error dropping doc status {self._collection_name}: {e}"
            )
            return {"status": "error", "message": str(e)}

    async def delete(self, ids: list[str]) -> None:
        coll = await self._get_collection()
        await coll.delete_many({"_id": {"$in": ids}})

    async def create_and_migrate_indexes_if_not_exists(self):
        """Create indexes to optimize pagination queries and migrate file_path indexes for Chinese collation"""
        try:
            # Get indexes for the current collection only
            coll = await self._get_collection()
            indexes_cursor = await coll.list_indexes()
            existing_indexes = await indexes_cursor.to_list(length=None)
            existing_index_names = {idx.get("name", "") for idx in existing_indexes}

            # Define collation configuration for Chinese pinyin sorting
            collation_config = {"locale": "zh", "numericOrdering": True}

            # Use workspace-specific index names to avoid cross-workspace conflicts
            workspace_prefix = f"{self.workspace}_" if self.workspace != "" else ""

            # 1. Define all indexes needed with workspace-specific names
            all_indexes = [
                # Original pagination indexes
                {
                    "name": f"{workspace_prefix}status_updated_at",
                    "keys": [("status", 1), ("updated_at", -1)],
                },
                {
                    "name": f"{workspace_prefix}status_created_at",
                    "keys": [("status", 1), ("created_at", -1)],
                },
                {"name": f"{workspace_prefix}updated_at", "keys": [("updated_at", -1)]},
                {"name": f"{workspace_prefix}created_at", "keys": [("created_at", -1)]},
                {"name": f"{workspace_prefix}id", "keys": [("_id", 1)]},
                {"name": f"{workspace_prefix}track_id", "keys": [("track_id", 1)]},
                # New file_path indexes with Chinese collation and workspace-specific names
                {
                    "name": f"{workspace_prefix}file_path_zh_collation",
                    "keys": [("file_path", 1)],
                    "collation": collation_config,
                },
                {
                    "name": f"{workspace_prefix}status_file_path_zh_collation",
                    "keys": [("status", 1), ("file_path", 1)],
                    "collation": collation_config,
                },
            ]

            # 2. Handle legacy index cleanup: only drop old indexes that exist in THIS collection
            legacy_index_names = [
                "file_path_zh_collation",
                "status_file_path_zh_collation",
                "status_updated_at",
                "status_created_at",
                "updated_at",
                "created_at",
                "id",
                "track_id",
            ]

            for legacy_name in legacy_index_names:
                if (
                    legacy_name in existing_index_names
                    and legacy_name
                    != f"{workspace_prefix}{legacy_name.replace(workspace_prefix, '')}"
                ):
                    try:
                        await coll.drop_index(legacy_name)
                        logger.debug(
                            f"[{self.workspace}] Migrated: dropped legacy index '{legacy_name}' from collection {self._collection_name}"
                        )
                        existing_index_names.discard(legacy_name)
                    except PyMongoError as drop_error:
                        logger.warning(
                            f"[{self.workspace}] Failed to drop legacy index '{legacy_name}' from collection {self._collection_name}: {drop_error}"
                        )

            # 3. Create all needed indexes with workspace-specific names
            for index_info in all_indexes:
                index_name = index_info["name"]
                if index_name not in existing_index_names:
                    create_kwargs = {"name": index_name}
                    if "collation" in index_info:
                        create_kwargs["collation"] = index_info["collation"]

                    try:
                        await coll.create_index(
                            index_info["keys"], **create_kwargs
                        )
                        logger.debug(
                            f"[{self.workspace}] Created index '{index_name}' for collection {self._collection_name}"
                        )
                    except PyMongoError as create_error:
                        # If creation still fails, log the error but continue with other indexes
                        logger.error(
                            f"[{self.workspace}] Failed to create index '{index_name}' for collection {self._collection_name}: {create_error}"
                        )
                else:
                    logger.debug(
                        f"[{self.workspace}] Index '{index_name}' already exists for collection {self._collection_name}"
                    )

        except PyMongoError as e:
            logger.error(
                f"[{self.workspace}] Error creating/migrating indexes for {self._collection_name}: {e}"
            )

    async def get_docs_paginated(
        self,
        status_filter: DocStatus | None = None,
        page: int = 1,
        page_size: int = 50,
        sort_field: str = "updated_at",
        sort_direction: str = "desc",
    ) -> tuple[list[tuple[str, DocProcessingStatus]], int]:
        """Get documents with pagination support

        Args:
            status_filter: Filter by document status, None for all statuses
            page: Page number (1-based)
            page_size: Number of documents per page (10-200)
            sort_field: Field to sort by ('created_at', 'updated_at', '_id')
            sort_direction: Sort direction ('asc' or 'desc')

        Returns:
            Tuple of (list of (doc_id, DocProcessingStatus) tuples, total_count)
        """
        # Validate parameters
        if page < 1:
            page = 1
        if page_size < 10:
            page_size = 10
        elif page_size > 200:
            page_size = 200

        if sort_field not in ["created_at", "updated_at", "_id", "file_path"]:
            sort_field = "updated_at"

        if sort_direction.lower() not in ["asc", "desc"]:
            sort_direction = "desc"

        # Build query filter
        query_filter = {}
        if status_filter is not None:
            query_filter["status"] = status_filter.value

        # Get total count
        coll = await self._get_collection()
        total_count = await coll.count_documents(query_filter)

        # Calculate skip value
        skip = (page - 1) * page_size

        # Build sort criteria
        sort_direction_value = 1 if sort_direction.lower() == "asc" else -1
        sort_criteria = [(sort_field, sort_direction_value)]

        # Query for paginated data with Chinese collation for file_path sorting
        if sort_field == "file_path":
            # Use Chinese collation for pinyin sorting
            cursor = (
                coll.find(query_filter)
                .sort(sort_criteria)
                .collation({"locale": "zh", "numericOrdering": True})
                .skip(skip)
                .limit(page_size)
            )
        else:
            # Use default sorting for other fields
            cursor = (
                 coll.find(query_filter)
                .sort(sort_criteria)
                .skip(skip)
                .limit(page_size)
            )
        result = await cursor.to_list(length=page_size)

        # Convert to (doc_id, DocProcessingStatus) tuples
        documents = []
        for doc in result:
            try:
                doc_id = doc["_id"]

                data = self._prepare_doc_status_data(doc)

                doc_status = DocProcessingStatus(**data)
                documents.append((doc_id, doc_status))
            except KeyError as e:
                logger.error(
                    f"[{self.workspace}] Missing required field for document {doc['_id']}: {e}"
                )
                continue

        return documents, total_count

    async def get_all_status_counts(self) -> dict[str, int]:
        """Get counts of documents in each status for all documents

        Returns:
            Dictionary mapping status names to counts, including 'all' field
        """
        pipeline = [{"$group": {"_id": "$status", "count": {"$sum": 1}}}]
        coll = await self._get_collection()
        cursor = await coll.aggregate(pipeline, allowDiskUse=True)
        result = await cursor.to_list()

        counts = {}
        total_count = 0
        for doc in result:
            counts[doc["_id"]] = doc["count"]
            total_count += doc["count"]

        # Add 'all' field with total count
        counts["all"] = total_count

        return counts

    async def get_doc_by_file_path(self, file_path: str) -> Union[dict[str, Any], None]:
        """Get document by file path

        Args:
            file_path: The file path to search for

        Returns:
            Union[dict[str, Any], None]: Document data if found, None otherwise
            Returns the same format as get_by_id method
        """
        coll = await self._get_collection()
        return await coll.find_one({"file_path": file_path})


@final
@dataclass
class MongoGraphStorage(BaseGraphStorage):
    """
    A concrete implementation using MongoDB's $graphLookup to demonstrate multi-hop queries.
    Refactored to support ContextVars for dynamic workspace isolation.
    """

    db: AsyncDatabase = field(default=None)
    # Cache for collections: { "workspace_namespace": AsyncCollection }
    _node_collections: dict = field(default_factory=dict, init=False)
    _edge_collections: dict = field(default_factory=dict, init=False)

    def __post_init__(self):
        # Initialize caches
        self._node_collections = {}
        self._edge_collections = {}

    async def _get_collections(self) -> tuple[AsyncCollection, AsyncCollection]:
        """
        Dynamically retrieve the node and edge collections for the current workspace.
        This method handles lazy loading and index creation.
        """
        # 1. Calculate dynamic collection names based on current ContextVar workspace
        ws = self.workspace
        if ws and ws.strip():
            node_coll_name = f"{ws}_{self.namespace}"
        else:
            node_coll_name = self.namespace

        edge_coll_name = f"{node_coll_name}_edges"

        # 2. Check cache
        if node_coll_name in self._node_collections:
            return self._node_collections[node_coll_name], self._edge_collections[node_coll_name]

        # 3. Initialize DB connection if needed
        if self.db is None:
            self.db = await ClientManager.get_client()

        # 4. Get or Create Collections
        node_coll = await get_or_create_collection(self.db, node_coll_name)
        edge_coll = await get_or_create_collection(self.db, edge_coll_name)

        # 5. Ensure Indexes exist for this specific node collection
        # We pass the collection instance specifically
        await self.create_search_index_if_not_exists(node_coll)

        # 6. Update cache
        self._node_collections[node_coll_name] = node_coll
        self._edge_collections[node_coll_name] = edge_coll

        return node_coll, edge_coll

    async def initialize(self):
        """
        Initialize DB connection only.
        Collections are lazy-loaded in _get_collections to support multi-tenancy.
        """
        async with get_data_init_lock():
            if self.db is None:
                self.db = await ClientManager.get_client()

    async def finalize(self):
        if self.db is not None:
            await ClientManager.release_client(self.db)
            self.db = None
            self._node_collections.clear()
            self._edge_collections.clear()

    # -------------------------------------------------------------------------
    # BASIC QUERIES
    # -------------------------------------------------------------------------

    async def has_node(self, node_id: str) -> bool:
        coll, _ = await self._get_collections()
        doc = await coll.find_one({"_id": node_id}, {"_id": 1})
        return doc is not None

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        _, edge_coll = await self._get_collections()
        doc = await edge_coll.find_one(
            {
                "$or": [
                    {
                        "source_node_id": source_node_id,
                        "target_node_id": target_node_id,
                    },
                    {
                        "source_node_id": target_node_id,
                        "target_node_id": source_node_id,
                    },
                ]
            },
            {"_id": 1},
        )
        return doc is not None

    # -------------------------------------------------------------------------
    # DEGREES
    # -------------------------------------------------------------------------

    async def node_degree(self, node_id: str) -> int:
        _, edge_coll = await self._get_collections()
        return await edge_coll.count_documents(
            {"$or": [{"source_node_id": node_id}, {"target_node_id": node_id}]}
        )

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        src_degree = await self.node_degree(src_id)
        trg_degree = await self.node_degree(tgt_id)
        return src_degree + trg_degree

    # -------------------------------------------------------------------------
    # GETTERS
    # -------------------------------------------------------------------------

    async def get_node(self, node_id: str) -> dict[str, str] | None:
        coll, _ = await self._get_collections()
        return await coll.find_one({"_id": node_id})

    async def get_edge(
            self, source_node_id: str, target_node_id: str
    ) -> dict[str, str] | None:
        _, edge_coll = await self._get_collections()
        return await edge_coll.find_one(
            {
                "$or": [
                    {
                        "source_node_id": source_node_id,
                        "target_node_id": target_node_id,
                    },
                    {
                        "source_node_id": target_node_id,
                        "target_node_id": source_node_id,
                    },
                ]
            }
        )

    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        _, edge_coll = await self._get_collections()
        cursor = edge_coll.find(
            {
                "$or": [
                    {"source_node_id": source_node_id},
                    {"target_node_id": source_node_id},
                ]
            },
            {"source_node_id": 1, "target_node_id": 1},
        )

        return [
            (e.get("source_node_id"), e.get("target_node_id")) async for e in cursor
        ]

    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, dict]:
        coll, _ = await self._get_collections()
        result = {}
        async for doc in coll.find({"_id": {"$in": node_ids}}):
            result[doc.get("_id")] = doc
        return result

    async def node_degrees_batch(self, node_ids: list[str]) -> dict[str, int]:
        _, edge_coll = await self._get_collections()
        merged_results = {}

        # Outbound degrees
        outbound_pipeline = [
            {"$match": {"source_node_id": {"$in": node_ids}}},
            {"$group": {"_id": "$source_node_id", "degree": {"$sum": 1}}},
        ]

        cursor = await edge_coll.aggregate(
            outbound_pipeline, allowDiskUse=True
        )
        async for doc in cursor:
            merged_results[doc.get("_id")] = doc.get("degree")

        # Inbound degrees
        inbound_pipeline = [
            {"$match": {"target_node_id": {"$in": node_ids}}},
            {"$group": {"_id": "$target_node_id", "degree": {"$sum": 1}}},
        ]

        cursor = await edge_coll.aggregate(
            inbound_pipeline, allowDiskUse=True
        )
        async for doc in cursor:
            merged_results[doc.get("_id")] = merged_results.get(
                doc.get("_id"), 0
            ) + doc.get("degree")

        return merged_results

    async def get_nodes_edges_batch(
            self, node_ids: list[str]
    ) -> dict[str, list[tuple[str, str]]]:
        _, edge_coll = await self._get_collections()
        result = {node_id: [] for node_id in node_ids}

        # Query outgoing edges
        outgoing_cursor = edge_coll.find(
            {"source_node_id": {"$in": node_ids}},
            {"source_node_id": 1, "target_node_id": 1},
        )
        async for edge in outgoing_cursor:
            source = edge["source_node_id"]
            target = edge["target_node_id"]
            result[source].append((source, target))

        # Query incoming edges
        incoming_cursor = edge_coll.find(
            {"target_node_id": {"$in": node_ids}},
            {"source_node_id": 1, "target_node_id": 1},
        )
        async for edge in incoming_cursor:
            source = edge["source_node_id"]
            target = edge["target_node_id"]
            result[target].append((source, target))

        return result

    # -------------------------------------------------------------------------
    # UPSERTS
    # -------------------------------------------------------------------------

    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        coll, _ = await self._get_collections()
        update_doc = {"$set": {**node_data}}
        if node_data.get("source_id", ""):
            update_doc["$set"]["source_ids"] = node_data["source_id"].split(
                GRAPH_FIELD_SEP
            )

        await coll.update_one({"_id": node_id}, update_doc, upsert=True)

    async def upsert_edge(
            self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ) -> None:
        coll, edge_coll = await self._get_collections()

        # Ensure source node exists in the correct collection
        # We manually call update_one on coll to avoid re-fetching collections via upsert_node
        await coll.update_one({"_id": source_node_id}, {"$set": {}}, upsert=True)

        update_doc = {"$set": edge_data}
        if edge_data.get("source_id", ""):
            update_doc["$set"]["source_ids"] = edge_data["source_id"].split(
                GRAPH_FIELD_SEP
            )

        edge_data["source_node_id"] = source_node_id
        edge_data["target_node_id"] = target_node_id

        await edge_coll.update_one(
            {
                "$or": [
                    {
                        "source_node_id": source_node_id,
                        "target_node_id": target_node_id,
                    },
                    {
                        "source_node_id": target_node_id,
                        "target_node_id": source_node_id,
                    },
                ]
            },
            update_doc,
            upsert=True,
        )

    # -------------------------------------------------------------------------
    # DELETION
    # -------------------------------------------------------------------------

    async def delete_node(self, node_id: str) -> None:
        coll, edge_coll = await self._get_collections()
        # Remove all edges
        await edge_coll.delete_many(
            {"$or": [{"source_node_id": node_id}, {"target_node_id": node_id}]}
        )
        # Remove the node doc
        await coll.delete_one({"_id": node_id})

    async def remove_nodes(self, nodes: list[str]) -> None:
        logger.info(f"[{self.workspace}] Deleting {len(nodes)} nodes")
        if not nodes:
            return
        coll, edge_coll = await self._get_collections()

        # 1. Remove all edges referencing these nodes
        await edge_coll.delete_many(
            {
                "$or": [
                    {"source_node_id": {"$in": nodes}},
                    {"target_node_id": {"$in": nodes}},
                ]
            }
        )
        # 2. Delete the node documents
        await coll.delete_many({"_id": {"$in": nodes}})
        logger.debug(f"[{self.workspace}] Successfully deleted nodes: {nodes}")

    async def remove_edges(self, edges: list[tuple[str, str]]) -> None:
        logger.info(f"[{self.workspace}] Deleting {len(edges)} edges")
        if not edges:
            return
        _, edge_coll = await self._get_collections()

        all_edge_pairs = []
        for source_id, target_id in edges:
            all_edge_pairs.append(
                {"source_node_id": source_id, "target_node_id": target_id}
            )
            all_edge_pairs.append(
                {"source_node_id": target_id, "target_node_id": source_id}
            )

        await edge_coll.delete_many({"$or": all_edge_pairs})
        logger.debug(f"[{self.workspace}] Successfully deleted edges: {edges}")

    # -------------------------------------------------------------------------
    # QUERY
    # -------------------------------------------------------------------------

    async def get_all_labels(self) -> list[str]:
        coll, _ = await self._get_collections()
        pipeline = [{"$project": {"_id": 1}}, {"$sort": {"_id": 1}}]
        cursor = await coll.aggregate(pipeline, allowDiskUse=True)
        labels = []
        async for doc in cursor:
            labels.append(doc["_id"])
        return labels

    def _construct_graph_node(
            self, node_id, node_data: dict[str, str]
    ) -> KnowledgeGraphNode:
        return KnowledgeGraphNode(
            id=node_id,
            labels=[node_id],
            properties={
                k: v
                for k, v in node_data.items()
                if k not in ["_id", "connected_edges", "source_ids", "edge_count"]
            },
        )

    def _construct_graph_edge(self, edge_id: str, edge: dict[str, str]):
        return KnowledgeGraphEdge(
            id=edge_id,
            type=edge.get("relationship", ""),
            source=edge["source_node_id"],
            target=edge["target_node_id"],
            properties={
                k: v
                for k, v in edge.items()
                if k not in ["_id", "source_node_id", "target_node_id", "relationship", "source_ids"]
            },
        )

    async def get_knowledge_graph_all_by_degree(
            self, max_depth: int, max_nodes: int
    ) -> KnowledgeGraph:
        coll, edge_coll = await self._get_collections()
        # [Crucial] We must use the dynamic name of the edge collection for aggregation lookups
        edge_coll_name = edge_coll.name

        total_node_count = await coll.count_documents({})
        result = KnowledgeGraph()
        seen_edges = set()

        result.is_truncated = total_node_count > max_nodes
        if result.is_truncated:
            pipeline = [
                {"$project": {"source_node_id": 1, "_id": 0}},
                {"$group": {"_id": "$source_node_id", "degree": {"$sum": 1}}},
                {
                    "$unionWith": {
                        "coll": edge_coll_name, # Dynamic name
                        "pipeline": [
                            {"$project": {"target_node_id": 1, "_id": 0}},
                            {
                                "$group": {
                                    "_id": "$target_node_id",
                                    "degree": {"$sum": 1},
                                }
                            },
                        ],
                    }
                },
                {"$group": {"_id": "$_id", "degree": {"$sum": "$degree"}}},
                {"$sort": {"degree": -1}},
                {"$limit": max_nodes},
            ]
            cursor = await edge_coll.aggregate(pipeline, allowDiskUse=True)

            node_ids = []
            async for doc in cursor:
                node_id = str(doc["_id"])
                node_ids.append(node_id)

            cursor = coll.find({"_id": {"$in": node_ids}}, {"source_ids": 0})
            async for doc in cursor:
                result.nodes.append(self._construct_graph_node(doc["_id"], doc))

            edge_cursor = edge_coll.find(
                {
                    "$and": [
                        {"source_node_id": {"$in": node_ids}},
                        {"target_node_id": {"$in": node_ids}},
                    ]
                }
            )
        else:
            cursor = coll.find({}, {"source_ids": 0})
            async for doc in cursor:
                node_id = str(doc["_id"])
                result.nodes.append(self._construct_graph_node(doc["_id"], doc))

            edge_cursor = edge_coll.find({})

        async for edge in edge_cursor:
            edge_id = f"{edge['source_node_id']}-{edge['target_node_id']}"
            if edge_id not in seen_edges:
                seen_edges.add(edge_id)
                result.edges.append(self._construct_graph_edge(edge_id, edge))

        return result

    async def _bidirectional_bfs_nodes(
            self,
            node_labels: list[str],
            seen_nodes: set[str],
            result: KnowledgeGraph,
            depth: int,
            max_depth: int,
            max_nodes: int,
    ) -> KnowledgeGraph:
        if depth > max_depth or len(result.nodes) > max_nodes:
            return result

        coll, edge_coll = await self._get_collections()
        cursor = coll.find({"_id": {"$in": node_labels}})

        async for node in cursor:
            node_id = node["_id"]
            if node_id not in seen_nodes:
                seen_nodes.add(node_id)
                result.nodes.append(self._construct_graph_node(node_id, node))
                if len(result.nodes) > max_nodes:
                    return result

        cursor = edge_coll.find(
            {
                "$or": [
                    {"source_node_id": {"$in": node_labels}},
                    {"target_node_id": {"$in": node_labels}},
                ]
            }
        )

        neighbor_nodes = []
        async for edge in cursor:
            if edge["source_node_id"] not in seen_nodes:
                neighbor_nodes.append(edge["source_node_id"])
            if edge["target_node_id"] not in seen_nodes:
                neighbor_nodes.append(edge["target_node_id"])

        if neighbor_nodes:
            result = await self._bidirectional_bfs_nodes(
                neighbor_nodes, seen_nodes, result, depth + 1, max_depth, max_nodes
            )

        return result

    async def get_knowledge_subgraph_bidirectional_bfs(
            self,
            node_label: str,
            depth: int,
            max_depth: int,
            max_nodes: int,
    ) -> KnowledgeGraph:
        seen_nodes = set()
        seen_edges = set()
        result = KnowledgeGraph()

        result = await self._bidirectional_bfs_nodes(
            [node_label], seen_nodes, result, depth, max_depth, max_nodes
        )

        all_node_ids = list(seen_nodes)
        _, edge_coll = await self._get_collections()

        cursor = edge_coll.find(
            {
                "$and": [
                    {"source_node_id": {"$in": all_node_ids}},
                    {"target_node_id": {"$in": all_node_ids}},
                ]
            }
        )

        async for edge in cursor:
            edge_id = f"{edge['source_node_id']}-{edge['target_node_id']}"
            if edge_id not in seen_edges:
                result.edges.append(self._construct_graph_edge(edge_id, edge))
                seen_edges.add(edge_id)

        return result

    async def get_knowledge_subgraph_in_out_bound_bfs(
            self, node_label: str, max_depth: int, max_nodes: int
    ) -> KnowledgeGraph:
        seen_nodes = set()
        seen_edges = set()
        result = KnowledgeGraph()

        coll, edge_coll = await self._get_collections()
        # [Crucial] Get names for aggregation
        coll_name = coll.name
        edge_coll_name = edge_coll.name

        project_doc = {
            "source_ids": 0,
            "created_at": 0,
            "entity_type": 0,
            "file_path": 0,
        }

        start_node = await coll.find_one({"_id": node_label})
        if not start_node:
            logger.warning(
                f"[{self.workspace}] Starting node with label {node_label} does not exist!"
            )
            return result

        seen_nodes.add(node_label)
        result.nodes.append(self._construct_graph_node(node_label, start_node))

        if max_depth == 0:
            return result

        max_depth = max_depth - 1

        pipeline = [
            {"$match": {"_id": node_label}},
            {"$project": project_doc},
            {
                "$graphLookup": {
                    "from": edge_coll_name, # Dynamic name
                    "startWith": "$_id",
                    "connectFromField": "target_node_id",
                    "connectToField": "source_node_id",
                    "maxDepth": max_depth,
                    "depthField": "depth",
                    "as": "connected_edges",
                },
            },
            {
                "$unionWith": {
                    "coll": coll_name, # Dynamic name
                    "pipeline": [
                        {"$match": {"_id": node_label}},
                        {"$project": project_doc},
                        {
                            "$graphLookup": {
                                "from": edge_coll_name, # Dynamic name
                                "startWith": "$_id",
                                "connectFromField": "source_node_id",
                                "connectToField": "target_node_id",
                                "maxDepth": max_depth,
                                "depthField": "depth",
                                "as": "connected_edges",
                            }
                        },
                    ],
                }
            },
        ]

        cursor = await coll.aggregate(pipeline, allowDiskUse=True)
        node_edges = []

        async for doc in cursor:
            if doc.get("connected_edges", []):
                node_edges.extend(doc.get("connected_edges"))

        node_edges = sorted(
            node_edges,
            key=lambda x: (x["depth"], -x["weight"]),
        )

        node_ids = []
        for edge in node_edges:
            if len(node_ids) < max_nodes and edge["source_node_id"] not in seen_nodes:
                node_ids.append(edge["source_node_id"])
                seen_nodes.add(edge["source_node_id"])

            if len(node_ids) < max_nodes and edge["target_node_id"] not in seen_nodes:
                node_ids.append(edge["target_node_id"])
                seen_nodes.add(edge["target_node_id"])

        cursor = coll.find({"_id": {"$in": node_ids}})

        async for doc in cursor:
            result.nodes.append(self._construct_graph_node(str(doc["_id"]), doc))

        for edge in node_edges:
            if (
                    edge["source_node_id"] not in seen_nodes
                    or edge["target_node_id"] not in seen_nodes
            ):
                continue

            edge_id = f"{edge['source_node_id']}-{edge['target_node_id']}"
            if edge_id not in seen_edges:
                result.edges.append(self._construct_graph_edge(edge_id, edge))
                seen_edges.add(edge_id)

        return result

    async def get_knowledge_graph(
            self,
            node_label: str,
            max_depth: int = 3,
            max_nodes: int = None,
    ) -> KnowledgeGraph:
        if max_nodes is None:
            max_nodes = self.global_config.get("max_graph_nodes", 1000)
        else:
            max_nodes = min(max_nodes, self.global_config.get("max_graph_nodes", 1000))

        result = KnowledgeGraph()
        start = time.perf_counter()

        try:
            if node_label == "*":
                result = await self.get_knowledge_graph_all_by_degree(
                    max_depth, max_nodes
                )
            elif GRAPH_BFS_MODE == "in_out_bound":
                result = await self.get_knowledge_subgraph_in_out_bound_bfs(
                    node_label, max_depth, max_nodes
                )
            else:
                result = await self.get_knowledge_subgraph_bidirectional_bfs(
                    node_label, 0, max_depth, max_nodes
                )

            duration = time.perf_counter() - start
            logger.info(
                f"[{self.workspace}] Subgraph query successful in {duration:.4f} seconds | Node count: {len(result.nodes)} | Edge count: {len(result.edges)} | Truncated: {result.is_truncated}"
            )

        except PyMongoError as e:
            coll, _ = await self._get_collections()
            if "memory limit" in str(e).lower() or "sort exceeded" in str(e).lower():
                logger.warning(
                    f"[{self.workspace}] MongoDB memory limit exceeded, falling back to simple query: {str(e)}"
                )
                try:
                    simple_cursor = coll.find({}).limit(max_nodes)
                    async for doc in simple_cursor:
                        result.nodes.append(
                            self._construct_graph_node(str(doc["_id"]), doc)
                        )
                    result.is_truncated = True
                    logger.info(
                        f"[{self.workspace}] Fallback query completed | Node count: {len(result.nodes)}"
                    )
                except PyMongoError as fallback_error:
                    logger.error(
                        f"[{self.workspace}] Fallback query also failed: {str(fallback_error)}"
                    )
            else:
                logger.error(f"[{self.workspace}] MongoDB query failed: {str(e)}")

        return result

    async def index_done_callback(self) -> None:
        pass

    async def get_all_nodes(self) -> list[dict]:
        coll, _ = await self._get_collections()
        cursor = coll.find({})
        nodes = []
        async for node in cursor:
            node_dict = dict(node)
            node_dict["id"] = node_dict.get("_id")
            nodes.append(node_dict)
        return nodes

    async def get_all_edges(self) -> list[dict]:
        _, edge_coll = await self._get_collections()
        cursor = edge_coll.find({})
        edges = []
        async for edge in cursor:
            edge_dict = dict(edge)
            edge_dict["source"] = edge_dict.get("source_node_id")
            edge_dict["target"] = edge_dict.get("target_node_id")
            edges.append(edge_dict)
        return edges

    async def get_popular_labels(self, limit: int = 300) -> list[str]:
        coll, edge_coll = await self._get_collections()
        edge_coll_name = edge_coll.name # Dynamic name for unionWith

        try:
            pipeline = [
                {"$group": {"_id": "$source_node_id", "out_degree": {"$sum": 1}}},
                {
                    "$unionWith": {
                        "coll": edge_coll_name,
                        "pipeline": [
                            {
                                "$group": {
                                    "_id": "$target_node_id",
                                    "in_degree": {"$sum": 1},
                                }
                            }
                        ],
                    }
                },
                {
                    "$group": {
                        "_id": "$_id",
                        "total_degree": {
                            "$sum": {
                                "$add": [
                                    {"$ifNull": ["$out_degree", 0]},
                                    {"$ifNull": ["$in_degree", 0]},
                                ]
                            }
                        },
                    }
                },
                {"$sort": {"total_degree": -1, "_id": 1}},
                {"$limit": limit},
                {"$project": {"_id": 1}},
            ]

            cursor = await edge_coll.aggregate(pipeline, allowDiskUse=True)
            labels = []
            async for doc in cursor:
                if doc.get("_id"):
                    labels.append(doc["_id"])

            logger.debug(
                f"[{self.workspace}] Retrieved {len(labels)} popular labels (limit: {limit})"
            )
            return labels
        except Exception as e:
            logger.error(f"[{self.workspace}] Error getting popular labels: {str(e)}")
            return []

    # -------------------------------------------------------------------------
    # SEARCH HELPER & METHODS
    # -------------------------------------------------------------------------

    async def _try_atlas_text_search(self, query_strip: str, limit: int) -> list[str]:
        try:
            coll, _ = await self._get_collections()
            pipeline = [
                {
                    "$search": {
                        "index": "entity_id_search_idx",
                        "text": {"query": query_strip, "path": "_id"},
                    }
                },
                {"$project": {"_id": 1, "score": {"$meta": "searchScore"}}},
                {"$limit": limit},
            ]
            cursor = await coll.aggregate(pipeline)
            labels = [doc["_id"] async for doc in cursor if doc.get("_id")]
            if labels:
                return labels
            return []
        except PyMongoError:
            return []

    async def _try_atlas_autocomplete_search(self, query_strip: str, limit: int) -> list[str]:
        try:
            coll, _ = await self._get_collections()
            pipeline = [
                {
                    "$search": {
                        "index": "entity_id_search_idx",
                        "autocomplete": {
                            "query": query_strip,
                            "path": "_id",
                            "fuzzy": {"maxEdits": 1, "prefixLength": 1},
                        },
                    }
                },
                {"$project": {"_id": 1, "score": {"$meta": "searchScore"}}},
                {"$limit": limit},
            ]
            cursor = await coll.aggregate(pipeline)
            labels = [doc["_id"] async for doc in cursor if doc.get("_id")]
            if labels:
                return labels
            return []
        except PyMongoError:
            return []

    async def _try_atlas_compound_search(self, query_strip: str, limit: int) -> list[str]:
        try:
            coll, _ = await self._get_collections()
            pipeline = [
                {
                    "$search": {
                        "index": "entity_id_search_idx",
                        "compound": {
                            "should": [
                                {
                                    "text": {
                                        "query": query_strip,
                                        "path": "_id",
                                        "score": {"boost": {"value": 10}},
                                    }
                                },
                                {
                                    "autocomplete": {
                                        "query": query_strip,
                                        "path": "_id",
                                        "score": {"boost": {"value": 5}},
                                        "fuzzy": {"maxEdits": 1, "prefixLength": 1},
                                    }
                                },
                                {
                                    "wildcard": {
                                        "query": f"*{query_strip}*",
                                        "path": "_id",
                                        "score": {"boost": {"value": 2}},
                                    }
                                },
                            ],
                            "minimumShouldMatch": 1,
                        },
                    }
                },
                {"$project": {"_id": 1, "score": {"$meta": "searchScore"}}},
                {"$sort": {"score": {"$meta": "searchScore"}}},
                {"$limit": limit},
            ]
            cursor = await coll.aggregate(pipeline)
            labels = [doc["_id"] async for doc in cursor if doc.get("_id")]
            if labels:
                return labels
            return []
        except PyMongoError:
            return []

    async def _fallback_regex_search(self, query_strip: str, limit: int) -> list[str]:
        try:
            logger.debug(
                f"[{self.workspace}] Using regex fallback search for: '{query_strip}'"
            )
            coll, _ = await self._get_collections()
            escaped_query = re.escape(query_strip)
            regex_condition = {"_id": {"$regex": escaped_query, "$options": "i"}}
            cursor = coll.find(regex_condition, {"_id": 1}).limit(limit * 2)
            docs = await cursor.to_list(length=limit * 2)

            labels = []
            for doc in docs:
                doc_id = doc.get("_id")
                if doc_id:
                    labels.append(doc_id)

            def sort_key(label):
                label_lower = label.lower()
                query_lower_strip = query_strip.lower()
                if label_lower == query_lower_strip:
                    return (0, label_lower)
                elif label_lower.startswith(query_lower_strip):
                    return (1, label_lower)
                else:
                    return (2, label_lower)

            labels.sort(key=sort_key)
            labels = labels[:limit]
            return labels

        except Exception as e:
            logger.error(f"[{self.workspace}] Regex fallback search failed: {e}")
            return []

    async def search_labels(self, query: str, limit: int = 50) -> list[str]:
        query_strip = query.strip()
        if not query_strip:
            return []

        coll, _ = await self._get_collections()
        try:
            node_count = await coll.count_documents({})
            if node_count == 0:
                return []
        except PyMongoError:
            return []

        search_methods = [
            ("text", self._try_atlas_text_search),
            ("autocomplete", self._try_atlas_autocomplete_search),
            ("compound", self._try_atlas_compound_search),
        ]

        for _, search_method in search_methods:
            try:
                labels = await search_method(query_strip, limit)
                if labels:
                    return labels
            except Exception:
                continue

        logger.info(
            f"[{self.workspace}] All Atlas Search methods failed, using regex fallback search for: '{query_strip}'"
        )
        return await self._fallback_regex_search(query_strip, limit)

    async def _check_if_index_needs_rebuild(self, indexes: list, index_name: str) -> bool:
        for index in indexes:
            if index["name"] == index_name:
                definition = index.get("latestDefinition", {})
                mappings = definition.get("mappings", {})
                fields = mappings.get("fields", {})
                id_field = fields.get("_id", {})

                if isinstance(id_field, dict) and id_field.get("type") == "autocomplete":
                    return True
                if not isinstance(id_field, list):
                    return True
                return False
        return True

    async def _safely_drop_old_index(self, index_name: str, collection: AsyncCollection):
        try:
            await collection.drop_search_index(index_name)
            logger.info(f"[{self.workspace}] Successfully dropped old search index '{index_name}'")
        except PyMongoError as e:
            logger.warning(f"[{self.workspace}] Could not drop old index '{index_name}': {e}")

    async def _create_improved_search_index(self, index_name: str, collection: AsyncCollection):
        search_index_model = SearchIndexModel(
            definition={
                "mappings": {
                    "dynamic": False,
                    "fields": {
                        "_id": [
                            {"type": "string"},
                            {"type": "token"},
                            {"type": "autocomplete", "maxGrams": 15, "minGrams": 2},
                        ]
                    },
                },
                "analyzer": "lucene.standard",
            },
            name=index_name,
            type="search",
        )
        await collection.create_search_index(search_index_model)
        logger.info(f"[{self.workspace}] Created improved Atlas Search index '{index_name}'")

    async def create_search_index_if_not_exists(self, collection: AsyncCollection = None):
        """Creates an improved Atlas Search index for entity search, rebuilding if necessary."""
        # Ensure we work on the provided collection instance
        if collection is None:
            return

        index_name = "entity_id_search_idx"
        try:
            indexes_cursor = await collection.list_search_indexes()
            indexes = await indexes_cursor.to_list(length=None)

            needs_rebuild = await self._check_if_index_needs_rebuild(indexes, index_name)

            if needs_rebuild:
                index_exists = any(idx["name"] == index_name for idx in indexes)
                if index_exists:
                    await self._safely_drop_old_index(index_name, collection)
                await self._create_improved_search_index(index_name, collection)
            else:
                logger.debug(f"[{self.workspace}] Atlas Search index '{index_name}' already exists")

        except PyMongoError as e:
            logger.debug(f"[{self.workspace}] Could not create Atlas Search index: {e}. Normal if not using Atlas.")
        except Exception as e:
            logger.warning(f"[{self.workspace}] Unexpected error creating Atlas Search index: {e}")

    async def drop(self) -> dict[str, str]:
        try:
            coll, edge_coll = await self._get_collections()

            result = await coll.delete_many({})
            deleted_count = result.deleted_count
            logger.info(f"[{self.workspace}] Dropped {deleted_count} documents from graph {coll.name}")

            result = await edge_coll.delete_many({})
            edge_count = result.deleted_count
            logger.info(f"[{self.workspace}] Dropped {edge_count} edges from graph {edge_coll.name}")

            # Clear cache for this workspace
            coll_name = coll.name
            if coll_name in self._node_collections:
                del self._node_collections[coll_name]
            if coll_name in self._edge_collections:
                del self._edge_collections[coll_name]

            return {
                "status": "success",
                "message": f"{deleted_count} documents and {edge_count} edges dropped",
            }
        except PyMongoError as e:
            logger.error(f"[{self.workspace}] Error dropping graph: {e}")
            return {"status": "error", "message": str(e)}


@final
@dataclass
class MongoVectorDBStorage(BaseVectorStorage):
    """
    A MongoDB implementation of VectorDBStorage that supports dynamic workspaces via ContextVars.
    """
    db: AsyncDatabase | None = field(default=None)
    # Cache for collections: { "collection_name": { "coll": AsyncCollection, "index_name": str } }
    _collections: dict = field(default_factory=dict, init=False)

    def __init__(
            self, namespace, global_config, embedding_func, workspace=None, meta_fields=None
    ):
        super().__init__(
            namespace=namespace,
            workspace=workspace or "",
            global_config=global_config,
            embedding_func=embedding_func,
            meta_fields=meta_fields or set(),
        )
        self.__post_init__()

    def __post_init__(self):
        # Initialize cache
        self._collections = {}

        kwargs = self.global_config.get("vector_db_storage_cls_kwargs", {})
        cosine_threshold = kwargs.get("cosine_better_than_threshold")
        if cosine_threshold is None:
            raise ValueError(
                "cosine_better_than_threshold must be specified in vector_db_storage_cls_kwargs"
            )
        self.cosine_better_than_threshold = cosine_threshold
        self._max_batch_size = self.global_config["embedding_batch_num"]

    def _get_collection_name(self) -> str:
        """Dynamically calculate collection name based on current workspace."""
        ws = self.workspace
        if ws and ws.strip():
            return f"{ws}_{self.namespace}"
        return self.namespace

    async def _get_collection_and_index(self) -> tuple[AsyncCollection, str]:
        """
        Get the collection and index name for the current workspace.
        Handles lazy loading and index creation.
        """
        coll_name = self._get_collection_name()

        # Check cache
        if coll_name in self._collections:
            return self._collections[coll_name]["coll"], self._collections[coll_name]["index_name"]

        # Initialize DB if needed
        if self.db is None:
            self.db = await ClientManager.get_client()

        # Get or Create Collection
        coll = await get_or_create_collection(self.db, coll_name)

        # Determine Index Name (collection-specific)
        index_name = f"vector_knn_index_{coll_name}"

        # Ensure Index Exists
        await self.create_vector_index_if_not_exists(coll, index_name)

        # Update Cache
        self._collections[coll_name] = {"coll": coll, "index_name": index_name}

        return coll, index_name

    async def initialize(self):
        """Initialize DB connection only."""
        async with get_data_init_lock():
            if self.db is None:
                self.db = await ClientManager.get_client()

    async def finalize(self):
        if self.db is not None:
            await ClientManager.release_client(self.db)
            self.db = None
            self._collections.clear()

    async def create_vector_index_if_not_exists(self, collection: AsyncCollection, index_name: str):
        """Creates an Atlas Vector Search index for a specific collection."""
        try:
            indexes_cursor = await collection.list_search_indexes()
            indexes = await indexes_cursor.to_list(length=None)
            for index in indexes:
                if index["name"] == index_name:
                    logger.debug(
                        f"[{self.workspace}] vector index {index_name} already exist"
                    )
                    return

            search_index_model = SearchIndexModel(
                definition={
                    "fields": [
                        {
                            "type": "vector",
                            "numDimensions": self.embedding_func.embedding_dim,
                            "path": "vector",
                            "similarity": "cosine",
                        }
                    ]
                },
                name=index_name,
                type="vectorSearch",
            )

            await collection.create_search_index(search_index_model)
            logger.info(
                f"[{self.workspace}] Vector index {index_name} created successfully."
            )

        except PyMongoError as e:
            error_msg = f"[{self.workspace}] Error creating vector index {index_name}: {e}"
            logger.error(error_msg)
            # We don't exit here to allow resilience in multi-tenant environments
            # raise SystemExit(...)

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        if not data:
            return

        # Get collection and index dynamically
        coll, _ = await self._get_collection_and_index()

        logger.debug(f"[{self.workspace}] Inserting {len(data)} to {self.namespace}")

        current_time = int(time.time())

        list_data = [
            {
                "_id": k,
                "created_at": current_time,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
            }
            for k, v in data.items()
        ]
        contents = [v["content"] for v in data.values()]
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]

        embedding_tasks = [self.embedding_func(batch) for batch in batches]
        embeddings_list = await asyncio.gather(*embedding_tasks)
        embeddings = np.concatenate(embeddings_list)
        for i, d in enumerate(list_data):
            d["vector"] = np.array(embeddings[i], dtype=np.float32).tolist()

        # Use the dynamic collection object 'coll'
        update_tasks = []
        for doc in list_data:
            update_tasks.append(
                coll.update_one({"_id": doc["_id"]}, {"$set": doc}, upsert=True)
            )
        await asyncio.gather(*update_tasks)

        return list_data

    async def query(
            self, query: str, top_k: int, query_embedding: list[float] = None
    ) -> list[dict[str, Any]]:
        """Queries the vector database using Atlas Vector Search."""

        # Get collection and index dynamically
        coll, index_name = await self._get_collection_and_index()

        if query_embedding is not None:
            if hasattr(query_embedding, "tolist"):
                query_vector = query_embedding.tolist()
            else:
                query_vector = list(query_embedding)
        else:
            embedding = await self.embedding_func(
                [query], _priority=5
            )
            query_vector = embedding[0].tolist()

        pipeline = [
            {
                "$vectorSearch": {
                    "index": index_name,  # Use dynamic index name
                    "path": "vector",
                    "queryVector": query_vector,
                    "numCandidates": 100,
                    "limit": top_k,
                }
            },
            {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
            {"$match": {"score": {"$gte": self.cosine_better_than_threshold}}},
            {"$project": {"vector": 0}},
        ]

        cursor = await coll.aggregate(pipeline, allowDiskUse=True)
        results = await cursor.to_list(length=None)

        return [
            {
                **doc,
                "id": doc["_id"],
                "distance": doc.get("score", None),
                "created_at": doc.get("created_at"),
            }
            for doc in results
        ]

    async def index_done_callback(self) -> None:
        pass

    async def delete(self, ids: list[str]) -> None:
        logger.debug(
            f"[{self.workspace}] Deleting {len(ids)} vectors from {self.namespace}"
        )
        if not ids:
            return

        coll, _ = await self._get_collection_and_index()

        if isinstance(ids, set):
            ids = list(ids)

        try:
            result = await coll.delete_many({"_id": {"$in": ids}})
            logger.debug(
                f"[{self.workspace}] Successfully deleted {result.deleted_count} vectors from {self.namespace}"
            )
        except PyMongoError as e:
            logger.error(
                f"[{self.workspace}] Error while deleting vectors from {self.namespace}: {str(e)}"
            )

    async def delete_entity(self, entity_name: str) -> None:
        coll, _ = await self._get_collection_and_index()
        try:
            entity_id = compute_mdhash_id(entity_name, prefix="ent-")
            logger.debug(
                f"[{self.workspace}] Attempting to delete entity {entity_name} with ID {entity_id}"
            )

            result = await coll.delete_one({"_id": entity_id})
            if result.deleted_count > 0:
                logger.debug(
                    f"[{self.workspace}] Successfully deleted entity {entity_name}"
                )
            else:
                logger.debug(
                    f"[{self.workspace}] Entity {entity_name} not found in storage"
                )
        except PyMongoError as e:
            logger.error(
                f"[{self.workspace}] Error deleting entity {entity_name}: {str(e)}"
            )

    async def delete_entity_relation(self, entity_name: str) -> None:
        coll, _ = await self._get_collection_and_index()
        try:
            relations_cursor = coll.find(
                {"$or": [{"src_id": entity_name}, {"tgt_id": entity_name}]}
            )
            relations = await relations_cursor.to_list(length=None)

            if not relations:
                logger.debug(
                    f"[{self.workspace}] No relations found for entity {entity_name}"
                )
                return

            relation_ids = [relation["_id"] for relation in relations]
            logger.debug(
                f"[{self.workspace}] Found {len(relation_ids)} relations for entity {entity_name}"
            )

            result = await coll.delete_many({"_id": {"$in": relation_ids}})
            logger.debug(
                f"[{self.workspace}] Deleted {result.deleted_count} relations for {entity_name}"
            )
        except PyMongoError as e:
            logger.error(
                f"[{self.workspace}] Error deleting relations for {entity_name}: {str(e)}"
            )

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        coll, _ = await self._get_collection_and_index()
        try:
            result = await coll.find_one({"_id": id})
            if result:
                result_dict = dict(result)
                if "_id" in result_dict and "id" not in result_dict:
                    result_dict["id"] = result_dict["_id"]
                return result_dict
            return None
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error retrieving vector data for ID {id}: {e}"
            )
            return None

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        if not ids:
            return []

        coll, _ = await self._get_collection_and_index()
        try:
            cursor = coll.find({"_id": {"$in": ids}})
            results = await cursor.to_list(length=None)

            formatted_map: dict[str, dict[str, Any]] = {}
            for result in results:
                result_dict = dict(result)
                if "_id" in result_dict and "id" not in result_dict:
                    result_dict["id"] = result_dict["_id"]
                key = str(result_dict.get("id", result_dict.get("_id")))
                formatted_map[key] = result_dict

            ordered_results: list[dict[str, Any] | None] = []
            for id_value in ids:
                ordered_results.append(formatted_map.get(str(id_value)))

            return ordered_results
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error retrieving vector data for IDs {ids}: {e}"
            )
            return []

    async def get_vectors_by_ids(self, ids: list[str]) -> dict[str, list[float]]:
        if not ids:
            return {}

        coll, _ = await self._get_collection_and_index()
        try:
            cursor = coll.find({"_id": {"$in": ids}}, {"vector": 1})
            results = await cursor.to_list(length=None)

            vectors_dict = {}
            for result in results:
                if result and "vector" in result and "_id" in result:
                    vectors_dict[result["_id"]] = result["vector"]

            return vectors_dict
        except PyMongoError as e:
            logger.error(
                f"[{self.workspace}] Error retrieving vectors by IDs from {self.namespace}: {e}"
            )
            return {}

    async def drop(self) -> dict[str, str]:
        """Drop the storage by removing all documents in the collection and recreating vector index."""
        try:
            # We get the collection but we don't necessarily need to ensure the index exists
            # if we are going to drop everything anyway, but for consistency we use the standard getter
            coll, index_name = await self._get_collection_and_index()
            coll_name = coll.name

            # Delete all documents
            result = await coll.delete_many({})
            deleted_count = result.deleted_count

            # Recreate vector index (ensure it exists after clearing data)
            await self.create_vector_index_if_not_exists(coll, index_name)

            logger.info(
                f"[{self.workspace}] Dropped {deleted_count} documents from vector storage {coll_name} and recreated vector index"
            )

            # Clear cache for this specific collection to be safe
            if coll_name in self._collections:
                del self._collections[coll_name]

            return {
                "status": "success",
                "message": f"{deleted_count} documents dropped and vector index recreated",
            }
        except PyMongoError as e:
            logger.error(
                f"[{self.workspace}] Error dropping vector storage: {e}"
            )
            return {"status": "error", "message": str(e)}




async def get_or_create_collection(db: AsyncDatabase, collection_name: str) -> AsyncCollection:
    """
    Get a collection, creating it if it doesn't exist.
    optimized for concurrency and performance (avoids listing all collections).
    """
    try:
        # 尝试直接创建。原子操作，如果已存在会立即抛出 CollectionInvalid
        collection = await db.create_collection(collection_name)
        logger.info(f"Created collection: {collection_name}")
        return collection
    except CollectionInvalid:
        # 集合已存在 (可能是此前创建的，也可能是并发请求刚刚创建的)
        # logger.debug(f"Collection '{collection_name}' already exists.")
        return db.get_collection(collection_name)
    except PyMongoError as e:
        # 处理其他真正的数据库错误 (如权限、连接断开)
        logger.error(f"Error getting/creating collection {collection_name}: {e}")
        raise e