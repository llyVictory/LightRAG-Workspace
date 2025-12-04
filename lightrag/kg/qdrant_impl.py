import asyncio
import configparser
import hashlib
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, List, final

import numpy as np
import pipmaster as pm

from ..base import BaseVectorStorage
from ..exceptions import QdrantMigrationError
from ..kg.shared_storage import get_data_init_lock
from ..utils import compute_mdhash_id, logger

if not pm.is_installed("qdrant-client"):
    pm.install("qdrant-client")

from qdrant_client import QdrantClient, models  # type: ignore

DEFAULT_WORKSPACE = "_"
WORKSPACE_ID_FIELD = "workspace_id"
ENTITY_PREFIX = "ent-"
CREATED_AT_FIELD = "created_at"
ID_FIELD = "id"

config = configparser.ConfigParser()
config.read("config.ini", "utf-8")


def compute_mdhash_id_for_qdrant(
        content: str, prefix: str = "", style: str = "simple"
) -> str:
    """
    Generate a UUID based on the content and support multiple formats.
    """
    if not content:
        raise ValueError("Content must not be empty.")

    hashed_content = hashlib.sha256((prefix + content).encode("utf-8")).digest()
    generated_uuid = uuid.UUID(bytes=hashed_content[:16], version=4)

    if style == "simple":
        return generated_uuid.hex
    elif style == "hyphenated":
        return str(generated_uuid)
    elif style == "urn":
        return f"urn:uuid:{generated_uuid}"
    else:
        raise ValueError("Invalid style. Choose from 'simple', 'hyphenated', or 'urn'.")


def workspace_filter_condition(workspace: str) -> models.FieldCondition:
    """
    Create a workspace filter condition for Qdrant queries.
    """
    # Ensure empty workspace becomes DEFAULT_WORKSPACE to match storage logic
    ws_value = workspace if workspace and workspace.strip() else DEFAULT_WORKSPACE
    return models.FieldCondition(
        key=WORKSPACE_ID_FIELD, match=models.MatchValue(value=ws_value)
    )


@final
@dataclass
class QdrantVectorDBStorage(BaseVectorStorage):
    _client: QdrantClient | None = field(default=None, init=False)
    _initialized: bool = field(default=False, init=False)

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
        # [修改] 移除静态 workspace 绑定逻辑

        # Use a shared collection with payload-based partitioning
        # The namespace is part of the collection name (e.g., lightrag_vdb_doc_status)
        self.final_namespace = f"lightrag_vdb_{self.namespace}"

        kwargs = self.global_config.get("vector_db_storage_cls_kwargs", {})
        cosine_threshold = kwargs.get("cosine_better_than_threshold")
        if cosine_threshold is None:
            raise ValueError(
                "cosine_better_than_threshold must be specified in vector_db_storage_cls_kwargs"
            )
        self.cosine_better_than_threshold = cosine_threshold

        self._client = None
        self._max_batch_size = self.global_config["embedding_batch_num"]
        self._initialized = False

    def _get_effective_workspace(self) -> str:
        """Helper to get current workspace or default value"""
        ws = self.workspace
        if ws and ws.strip():
            return ws
        return DEFAULT_WORKSPACE

    @staticmethod
    def setup_collection(
            client: QdrantClient,
            collection_name: str,
            **kwargs,
    ):
        """
        Setup Qdrant collection.
        Simplified: removed migration logic for clarity in this refactor.
        """
        if not client.collection_exists(collection_name):
            logger.info(f"Qdrant: Creating new collection '{collection_name}'")
            client.create_collection(collection_name, **kwargs)

            # Create payload index for workspace_id (Crucial for multi-tenancy performance)
            client.create_payload_index(
                collection_name=collection_name,
                field_name=WORKSPACE_ID_FIELD,
                field_schema=models.KeywordIndexParams(
                    type=models.KeywordIndexType.KEYWORD,
                    is_tenant=True, # Optimized for tenancy
                ),
            )
            logger.info(f"Qdrant: Collection '{collection_name}' created successfully")
        else:
            # Ensure index exists even if collection exists
            try:
                collection_info = client.get_collection(collection_name)
                # Check payload schema (simplified check)
                # In production, might want deeper check, but re-creating index is idempotent-ish or throws safe error
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name=WORKSPACE_ID_FIELD,
                    field_schema=models.KeywordIndexParams(
                        type=models.KeywordIndexType.KEYWORD,
                        is_tenant=True,
                    ),
                )
            except Exception:
                pass # Ignore if already exists

    async def initialize(self):
        """Initialize Qdrant collection"""
        async with get_data_init_lock():
            if self._initialized:
                return

            try:
                if self._client is None:
                    self._client = QdrantClient(
                        url=os.environ.get(
                            "QDRANT_URL", config.get("qdrant", "uri", fallback=None)
                        ),
                        api_key=os.environ.get(
                            "QDRANT_API_KEY",
                            config.get("qdrant", "apikey", fallback=None),
                        ),
                    )
                    logger.debug("QdrantClient created successfully")

                # Setup shared collection
                # We setup the collection once. Isolation is handled via payload.
                QdrantVectorDBStorage.setup_collection(
                    self._client,
                    self.final_namespace,
                    vectors_config=models.VectorParams(
                        size=self.embedding_func.embedding_dim,
                        distance=models.Distance.COSINE,
                    ),
                    hnsw_config=models.HnswConfigDiff(
                        payload_m=16,
                        m=0,
                    ),
                )

                self._initialized = True
                logger.info(
                    f"Qdrant collection '{self.final_namespace}' initialized successfully"
                )
            except Exception as e:
                logger.error(
                    f"Failed to initialize Qdrant collection '{self.namespace}': {e}"
                )
                raise

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        # [修改] 获取当前 workspace
        current_workspace = self._get_effective_workspace()

        logger.debug(f"[{current_workspace}] Inserting {len(data)} to {self.namespace}")
        if not data:
            return

        import time
        current_time = int(time.time())

        list_data = [
            {
                ID_FIELD: k,
                WORKSPACE_ID_FIELD: current_workspace, # [关键] 写入当前 workspace
                CREATED_AT_FIELD: current_time,
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

        list_points = []
        for i, d in enumerate(list_data):
            # [关键] ID 计算必须包含 workspace 前缀，确保不同 workspace 的同名 ID 不冲突
            list_points.append(
                models.PointStruct(
                    id=compute_mdhash_id_for_qdrant(
                        d[ID_FIELD], prefix=current_workspace
                    ),
                    vector=embeddings[i],
                    payload=d,
                )
            )

        results = self._client.upsert(
            collection_name=self.final_namespace, points=list_points, wait=True
        )
        return results

    async def query(
            self, query: str, top_k: int, query_embedding: list[float] = None
    ) -> list[dict[str, Any]]:
        current_workspace = self._get_effective_workspace()

        if query_embedding is not None:
            embedding = query_embedding
        else:
            embedding_result = await self.embedding_func(
                [query], _priority=5
            )
            embedding = embedding_result[0]

        results = self._client.query_points(
            collection_name=self.final_namespace,
            query=embedding,
            limit=top_k,
            with_payload=True,
            score_threshold=self.cosine_better_than_threshold,
            # [关键] 增加 workspace 过滤条件
            query_filter=models.Filter(
                must=[workspace_filter_condition(current_workspace)]
            ),
        ).points

        return [
            {
                **dp.payload,
                "distance": dp.score,
                CREATED_AT_FIELD: dp.payload.get(CREATED_AT_FIELD),
            }
            for dp in results
        ]

    async def index_done_callback(self) -> None:
        pass

    async def delete(self, ids: List[str]) -> None:
        try:
            if not ids:
                return

            current_workspace = self._get_effective_workspace()

            # Convert to Qdrant compatible ids (with workspace prefix)
            qdrant_ids = [
                compute_mdhash_id_for_qdrant(id, prefix=current_workspace)
                for id in ids
            ]

            self._client.delete(
                collection_name=self.final_namespace,
                points_selector=models.PointIdsList(points=qdrant_ids),
                wait=True,
            )
            logger.debug(
                f"[{current_workspace}] Successfully deleted {len(ids)} vectors from {self.namespace}"
            )
        except Exception as e:
            logger.error(f"[{self.workspace}] Error deleting vectors: {e}")

    async def delete_entity(self, entity_name: str) -> None:
        try:
            current_workspace = self._get_effective_workspace()

            entity_id = compute_mdhash_id(entity_name, prefix=ENTITY_PREFIX)
            qdrant_entity_id = compute_mdhash_id_for_qdrant(
                entity_id, prefix=current_workspace
            )

            self._client.delete(
                collection_name=self.final_namespace,
                points_selector=models.PointIdsList(points=[qdrant_entity_id]),
                wait=True,
            )
            logger.debug(f"[{current_workspace}] Deleted entity {entity_name}")
        except Exception as e:
            logger.error(f"[{self.workspace}] Error deleting entity {entity_name}: {e}")

    async def delete_entity_relation(self, entity_name: str) -> None:
        try:
            current_workspace = self._get_effective_workspace()

            # Find relations with workspace filtering
            results = self._client.scroll(
                collection_name=self.final_namespace,
                scroll_filter=models.Filter(
                    must=[workspace_filter_condition(current_workspace)],
                    should=[
                        models.FieldCondition(
                            key="src_id", match=models.MatchValue(value=entity_name)
                        ),
                        models.FieldCondition(
                            key="tgt_id", match=models.MatchValue(value=entity_name)
                        ),
                    ],
                ),
                with_payload=True,
                limit=1000,
            )

            relation_points = results[0]
            ids_to_delete = [point.id for point in relation_points]

            if ids_to_delete:
                self._client.delete(
                    collection_name=self.final_namespace,
                    points_selector=models.PointIdsList(points=ids_to_delete),
                    wait=True,
                )
                logger.debug(f"[{current_workspace}] Deleted {len(ids_to_delete)} relations")
        except Exception as e:
            logger.error(f"[{self.workspace}] Error deleting relations: {e}")

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        try:
            current_workspace = self._get_effective_workspace()

            qdrant_id = compute_mdhash_id_for_qdrant(
                id, prefix=current_workspace
            )

            result = self._client.retrieve(
                collection_name=self.final_namespace,
                ids=[qdrant_id],
                with_payload=True,
            )

            if not result:
                return None

            payload = result[0].payload
            if CREATED_AT_FIELD not in payload:
                payload[CREATED_AT_FIELD] = None

            # Optional: verify workspace matches (double check)
            if payload.get(WORKSPACE_ID_FIELD) != current_workspace:
                return None

            return payload
        except Exception as e:
            logger.error(f"[{self.workspace}] Error retrieving ID {id}: {e}")
            return None

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        if not ids:
            return []

        try:
            current_workspace = self._get_effective_workspace()

            qdrant_ids = [
                compute_mdhash_id_for_qdrant(id, prefix=current_workspace)
                for id in ids
            ]

            results = self._client.retrieve(
                collection_name=self.final_namespace,
                ids=qdrant_ids,
                with_payload=True,
            )

            # Map results
            payload_map = {}
            for point in results:
                payload = dict(point.payload or {})
                if payload.get(WORKSPACE_ID_FIELD) != current_workspace:
                    continue # Skip if cross-tenant leak (unlikely but safe)

                original_id = payload.get(ID_FIELD)
                if original_id:
                    payload_map[str(original_id)] = payload

            # Order results
            return [payload_map.get(str(i)) for i in ids]
        except Exception as e:
            logger.error(f"[{self.workspace}] Error retrieving IDs: {e}")
            return []

    async def get_vectors_by_ids(self, ids: list[str]) -> dict[str, list[float]]:
        if not ids:
            return {}

        try:
            current_workspace = self._get_effective_workspace()

            qdrant_ids = [
                compute_mdhash_id_for_qdrant(id, prefix=current_workspace)
                for id in ids
            ]

            results = self._client.retrieve(
                collection_name=self.final_namespace,
                ids=qdrant_ids,
                with_vectors=True,
                with_payload=True,
            )

            vectors_dict = {}
            for point in results:
                if point and point.vector is not None:
                    # Check workspace
                    if point.payload.get(WORKSPACE_ID_FIELD) != current_workspace:
                        continue

                    original_id = point.payload.get(ID_FIELD)
                    if original_id:
                        vector_data = point.vector
                        if isinstance(vector_data, np.ndarray):
                            vector_data = vector_data.tolist()
                        vectors_dict[original_id] = vector_data

            return vectors_dict
        except Exception as e:
            logger.error(f"[{self.workspace}] Error retrieving vectors: {e}")
            return {}

    async def drop(self) -> dict[str, str]:
        """Delete all data for the CURRENT workspace only."""
        try:
            current_workspace = self._get_effective_workspace()

            self._client.delete(
                collection_name=self.final_namespace,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[workspace_filter_condition(current_workspace)]
                    )
                ),
                wait=True,
            )

            logger.info(
                f"[{self.workspace}] Dropped workspace data from Qdrant"
            )
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            logger.error(f"[{self.workspace}] Error dropping data: {e}")
            return {"status": "error", "message": str(e)}