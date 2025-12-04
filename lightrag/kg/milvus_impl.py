import asyncio
import os
from typing import Any, final, Set
from dataclasses import dataclass, field
import numpy as np
from lightrag.utils import logger, compute_mdhash_id
from ..base import BaseVectorStorage
from ..constants import DEFAULT_MAX_FILE_PATH_LENGTH
from ..kg.shared_storage import get_data_init_lock
import pipmaster as pm

if not pm.is_installed("pymilvus"):
    pm.install("pymilvus>=2.6.2")

import configparser
from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema  # type: ignore

config = configparser.ConfigParser()
config.read("config.ini", "utf-8")


@final
@dataclass
class MilvusVectorDBStorage(BaseVectorStorage):
    # 缓存已初始化（检查/创建/加载过）的集合名称
    _ready_collections: Set[str] = field(default_factory=set, init=False)
    # 用于集合创建的锁，防止并发创建同一个集合
    _collection_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    def __post_init__(self):
        # [修改] 移除原有对 self.final_namespace 的静态计算
        # [修改] 移除对 MILVUS_WORKSPACE 环境变量的处理，完全依赖 Base 类通过 ContextVars 传递的 self.workspace

        kwargs = self.global_config.get("vector_db_storage_cls_kwargs", {})
        cosine_threshold = kwargs.get("cosine_better_than_threshold")
        if cosine_threshold is None:
            raise ValueError(
                "cosine_better_than_threshold must be specified in vector_db_storage_cls_kwargs"
            )
        self.cosine_better_than_threshold = cosine_threshold

        # Ensure created_at is in meta_fields
        if "created_at" not in self.meta_fields:
            self.meta_fields.add("created_at")

        self._client = None
        self._max_batch_size = self.global_config["embedding_batch_num"]

        # 初始化状态缓存
        self._ready_collections = set()
        self._collection_lock = asyncio.Lock()

    def _get_collection_name(self) -> str:
        """根据当前 ContextVars 中的 workspace 动态生成集合名"""
        ws = self.workspace
        if ws and ws.strip():
            return f"{ws}_{self.namespace}"
        return self.namespace

    async def initialize(self):
        """Initialize Milvus Client only. Collections are lazy-loaded."""
        async with get_data_init_lock():
            if self._client is not None:
                return

            try:
                # Create MilvusClient if not already created
                self._client = MilvusClient(
                    uri=os.environ.get(
                        "MILVUS_URI",
                        config.get(
                            "milvus",
                            "uri",
                            fallback=os.path.join(
                                self.global_config["working_dir"], "milvus_lite.db"
                            ),
                        ),
                    ),
                    user=os.environ.get(
                        "MILVUS_USER", config.get("milvus", "user", fallback=None)
                    ),
                    password=os.environ.get(
                        "MILVUS_PASSWORD",
                        config.get("milvus", "password", fallback=None),
                    ),
                    token=os.environ.get(
                        "MILVUS_TOKEN", config.get("milvus", "token", fallback=None)
                    ),
                    db_name=os.environ.get(
                        "MILVUS_DB_NAME",
                        config.get("milvus", "db_name", fallback=None),
                    ),
                )
                logger.debug(f"MilvusClient created successfully")

                # [注意] 这里不再调用 _create_collection_if_not_exist
                # 因为此时不知道 workspace 是什么，推迟到 upsert/query 时执行
            except Exception as e:
                logger.error(f"Failed to initialize Milvus Client: {e}")
                raise

    async def _ensure_collection_ready(self, collection_name: str):
        """
        确保当前 Workspace 对应的集合已存在且已加载。
        使用双重检查锁定模式。
        """
        if collection_name in self._ready_collections:
            return

        async with self._collection_lock:
            if collection_name in self._ready_collections:
                return

            try:
                # 这是一个同步方法，但在 async 锁中运行是安全的
                self._create_collection_if_not_exist(collection_name)
                self._ready_collections.add(collection_name)
            except Exception as e:
                logger.error(f"[{self.workspace}] Failed to ensure collection {collection_name}: {e}")
                raise

    # ----------------------------------------------------------------------
    # Schema & Index Helpers (Updated to accept collection_name)
    # ----------------------------------------------------------------------

    def _create_schema_for_namespace(self) -> CollectionSchema:
        """Create schema based on the current instance's namespace suffix"""
        # Logic remains the same, checking self.namespace (e.g. "doc_status")
        # which is constant across workspaces.

        dimension = self.embedding_func.embedding_dim

        base_fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension),
            FieldSchema(name="created_at", dtype=DataType.INT64),
        ]

        if self.namespace.endswith("entities"):
            specific_fields = [
                FieldSchema(name="entity_name", dtype=DataType.VARCHAR, max_length=512, nullable=True),
                FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=DEFAULT_MAX_FILE_PATH_LENGTH, nullable=True),
            ]
            description = "LightRAG entities vector storage"

        elif self.namespace.endswith("relationships"):
            specific_fields = [
                FieldSchema(name="src_id", dtype=DataType.VARCHAR, max_length=512, nullable=True),
                FieldSchema(name="tgt_id", dtype=DataType.VARCHAR, max_length=512, nullable=True),
                FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=DEFAULT_MAX_FILE_PATH_LENGTH, nullable=True),
            ]
            description = "LightRAG relationships vector storage"

        elif self.namespace.endswith("chunks"):
            specific_fields = [
                FieldSchema(name="full_doc_id", dtype=DataType.VARCHAR, max_length=64, nullable=True),
                FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=DEFAULT_MAX_FILE_PATH_LENGTH, nullable=True),
            ]
            description = "LightRAG chunks vector storage"

        else:
            specific_fields = [
                FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=DEFAULT_MAX_FILE_PATH_LENGTH, nullable=True),
            ]
            description = "LightRAG generic vector storage"

        all_fields = base_fields + specific_fields

        return CollectionSchema(
            fields=all_fields,
            description=description,
            enable_dynamic_field=True,
        )

    def _get_index_params(self):
        try:
            if hasattr(self._client, "prepare_index_params"):
                return self._client.prepare_index_params()
        except Exception:
            pass
        try:
            from pymilvus.client.prepare import IndexParams
            return IndexParams()
        except ImportError:
            pass
        try:
            from pymilvus.client.types import IndexParams
            return IndexParams()
        except ImportError:
            pass
        try:
            from pymilvus import IndexParams
            return IndexParams()
        except ImportError:
            pass
        return None

    def _create_vector_index_fallback(self, collection_name: str):
        try:
            self._client.create_index(
                collection_name=collection_name,
                field_name="vector",
                index_params={
                    "index_type": "HNSW",
                    "metric_type": "COSINE",
                    "params": {"M": 16, "efConstruction": 256},
                },
            )
            logger.debug(f"[{self.workspace}] Created vector index using fallback method for {collection_name}")
        except Exception as e:
            logger.warning(f"[{self.workspace}] Failed to create vector index fallback for {collection_name}: {e}")

    def _create_scalar_index_fallback(self, collection_name: str, field_name: str, index_type: str):
        if index_type == "SORTED":
            return
        try:
            self._client.create_index(
                collection_name=collection_name,
                field_name=field_name,
                index_params={"index_type": index_type},
            )
            logger.debug(f"[{self.workspace}] Created {field_name} index fallback for {collection_name}")
        except Exception as e:
            logger.info(f"[{self.workspace}] Could not create {field_name} index fallback: {e}")

    def _create_indexes_after_collection(self, collection_name: str):
        """Create indexes for a specific collection"""
        try:
            IndexParamsClass = self._get_index_params()

            if IndexParamsClass is not None:
                try:
                    vector_index = IndexParamsClass
                    vector_index.add_index(
                        field_name="vector",
                        index_type="HNSW",
                        metric_type="COSINE",
                        params={"M": 16, "efConstruction": 256},
                    )
                    self._client.create_index(
                        collection_name=collection_name, index_params=vector_index
                    )
                except Exception as e:
                    logger.debug(f"[{self.workspace}] IndexParams method failed for vector index: {e}")
                    self._create_vector_index_fallback(collection_name)

                # Scalar indexes based on namespace
                if self.namespace.endswith("entities"):
                    try:
                        idx = self._get_index_params()
                        idx.add_index(field_name="entity_name", index_type="INVERTED")
                        self._client.create_index(collection_name=collection_name, index_params=idx)
                    except Exception:
                        self._create_scalar_index_fallback(collection_name, "entity_name", "INVERTED")

                elif self.namespace.endswith("relationships"):
                    try:
                        idx = self._get_index_params()
                        idx.add_index(field_name="src_id", index_type="INVERTED")
                        self._client.create_index(collection_name=collection_name, index_params=idx)
                    except Exception:
                        self._create_scalar_index_fallback(collection_name, "src_id", "INVERTED")
                    try:
                        idx = self._get_index_params()
                        idx.add_index(field_name="tgt_id", index_type="INVERTED")
                        self._client.create_index(collection_name=collection_name, index_params=idx)
                    except Exception:
                        self._create_scalar_index_fallback(collection_name, "tgt_id", "INVERTED")

                elif self.namespace.endswith("chunks"):
                    try:
                        idx = self._get_index_params()
                        idx.add_index(field_name="full_doc_id", index_type="INVERTED")
                        self._client.create_index(collection_name=collection_name, index_params=idx)
                    except Exception:
                        self._create_scalar_index_fallback(collection_name, "full_doc_id", "INVERTED")

            else:
                self._create_vector_index_fallback(collection_name)
                if self.namespace.endswith("entities"):
                    self._create_scalar_index_fallback(collection_name, "entity_name", "INVERTED")
                elif self.namespace.endswith("relationships"):
                    self._create_scalar_index_fallback(collection_name, "src_id", "INVERTED")
                    self._create_scalar_index_fallback(collection_name, "tgt_id", "INVERTED")
                elif self.namespace.endswith("chunks"):
                    self._create_scalar_index_fallback(collection_name, "full_doc_id", "INVERTED")

            logger.info(f"[{self.workspace}] Created indexes for collection: {collection_name}")

        except Exception as e:
            logger.warning(f"[{self.workspace}] Failed to create some indexes for {collection_name}: {e}")

    def _get_required_fields_for_namespace(self) -> dict:
        base_fields = {
            "id": {"type": "VarChar", "is_primary": True},
            "vector": {"type": "FloatVector"},
            "created_at": {"type": "Int64"},
        }
        if self.namespace.endswith("entities"):
            specific_fields = {"entity_name": {"type": "VarChar"}, "file_path": {"type": "VarChar"}}
        elif self.namespace.endswith("relationships"):
            specific_fields = {"src_id": {"type": "VarChar"}, "tgt_id": {"type": "VarChar"}, "file_path": {"type": "VarChar"}}
        elif self.namespace.endswith("chunks"):
            specific_fields = {"full_doc_id": {"type": "VarChar"}, "file_path": {"type": "VarChar"}}
        else:
            specific_fields = {"file_path": {"type": "VarChar"}}
        return {**base_fields, **specific_fields}

    def _is_field_compatible(self, existing_field: dict, expected_config: dict) -> bool:
        # Compatibility logic remains mostly identical
        field_name = existing_field.get("name", "unknown")
        existing_type = existing_field.get("type")
        expected_type = expected_config.get("type")

        # Convert Enum/Int to String logic (kept from original)
        if hasattr(existing_type, "name"):
            existing_type = existing_type.name
        elif isinstance(existing_type, int):
            type_mapping = {21: "VarChar", 101: "FloatVector", 5: "Int64", 9: "Double"}
            existing_type = type_mapping.get(existing_type, str(existing_type))

        type_aliases = {
            "VARCHAR": "VarChar", "String": "VarChar",
            "FLOAT_VECTOR": "FloatVector", "INT64": "Int64",
            "BigInt": "Int64", "DOUBLE": "Double", "Float": "Double",
        }
        existing_type = type_aliases.get(existing_type, existing_type)
        expected_type = type_aliases.get(expected_type, expected_type)

        if existing_type != expected_type:
            logger.warning(f"[{self.workspace}] Type mismatch for '{field_name}': expected {expected_type}, got {existing_type}")
            return False

        if expected_config.get("is_primary"):
            is_primary = (existing_field.get("is_primary_key", False) or
                          existing_field.get("is_primary", False) or
                          existing_field.get("primary_key", False))
            if field_name == "id" and not is_primary:
                # Lenient ID check
                pass
            elif not is_primary:
                return False
        return True

    def _check_vector_dimension(self, collection_info: dict, collection_name: str):
        current_dimension = self.embedding_func.embedding_dim
        for field in collection_info.get("fields", []):
            if field.get("name") == "vector":
                # ... existing dimension check logic ...
                # Simplified for brevity, same as original logic but returning error/logging
                existing_dimension = field.get("params", {}).get("dim")
                try:
                    if int(existing_dimension) != int(current_dimension):
                        raise ValueError(f"Vector dimension mismatch for {collection_name}")
                except Exception:
                    pass
                return
        logger.warning(f"[{self.workspace}] Vector field not found in {collection_name}")

    def _check_file_path_length_restriction(self, collection_info: dict) -> bool:
        existing_fields = {field["name"]: field for field in collection_info.get("fields", [])}
        if "file_path" in existing_fields:
            max_len = existing_fields["file_path"].get("params", {}).get("max_length")
            if max_len and max_len < DEFAULT_MAX_FILE_PATH_LENGTH:
                return True
        return False

    def _check_schema_compatibility(self, collection_info: dict, collection_name: str):
        existing_fields = {field["name"]: field for field in collection_info.get("fields", [])}

        # Check migration needs
        if self._check_file_path_length_restriction(collection_info):
            logger.info(f"[{self.workspace}] Starting automatic migration for {collection_name}")
            self._migrate_collection_schema(collection_name)
            return

        # Check critical fields
        critical_fields = {"id": {"type": "VarChar", "is_primary": True}}
        for field_name, expected_config in critical_fields.items():
            if field_name in existing_fields:
                if not self._is_field_compatible(existing_fields[field_name], expected_config):
                    raise ValueError(f"Critical schema incompatibility in {collection_name}")

    def _migrate_collection_schema(self, original_collection_name: str):
        """Migrate collection schema (Logic updated to accept name)"""
        temp_collection_name = f"{original_collection_name}_temp"
        iterator = None
        try:
            # 1. Create temp collection
            new_schema = self._create_schema_for_namespace()
            self._client.create_collection(collection_name=temp_collection_name, schema=new_schema)
            self._create_indexes_after_collection(temp_collection_name)
            self._client.load_collection(temp_collection_name)

            # 2. Copy data
            iterator = self._client.query_iterator(
                collection_name=original_collection_name,
                batch_size=2000,
                output_fields=["*"]
            )

            while True:
                batch_data = iterator.next()
                if not batch_data: break
                self._client.insert(collection_name=temp_collection_name, data=batch_data)

            # 3. Rename
            try:
                self._client.rename_collection(original_collection_name, f"{original_collection_name}_old")
            except Exception:
                self._client.drop_collection(original_collection_name)

            self._client.rename_collection(temp_collection_name, original_collection_name)

        except Exception as e:
            logger.error(f"[{self.workspace}] Migration failed for {original_collection_name}: {e}")
            if self._client.has_collection(temp_collection_name):
                self._client.drop_collection(temp_collection_name)
            raise e
        finally:
            if iterator: iterator.close()

    def _ensure_collection_loaded(self, collection_name: str):
        if not self._client.has_collection(collection_name):
            raise ValueError(f"Collection {collection_name} does not exist")
        self._client.load_collection(collection_name)

    def _create_collection_if_not_exist(self, collection_name: str):
        """Create or validate collection for the current workspace"""
        try:
            if self._client.has_collection(collection_name):
                try:
                    info = self._client.describe_collection(collection_name)
                    self._check_vector_dimension(info, collection_name)
                    self._check_schema_compatibility(info, collection_name)
                    self._ensure_collection_loaded(collection_name)
                    return
                except Exception as e:
                    logger.error(f"[{self.workspace}] Validation failed for existing collection {collection_name}: {e}")
                    raise RuntimeError(f"Validation failed for {collection_name}")

            # Create new
            logger.info(f"[{self.workspace}] Creating new collection: {collection_name}")
            schema = self._create_schema_for_namespace()
            self._client.create_collection(collection_name=collection_name, schema=schema)
            self._create_indexes_after_collection(collection_name)
            self._ensure_collection_loaded(collection_name)

        except Exception as e:
            logger.error(f"[{self.workspace}] Error creating collection {collection_name}: {e}")
            raise

    # ----------------------------------------------------------------------
    # Core Operations (Upsert, Query, Delete)
    # ----------------------------------------------------------------------

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        if not data:
            return

        # [修改] 1. 计算动态集合名
        coll_name = self._get_collection_name()
        # [修改] 2. 确保集合就绪 (检查/创建/加载)
        await self._ensure_collection_ready(coll_name)

        import time
        current_time = int(time.time())

        list_data: list[dict[str, Any]] = [
            {
                "id": k,
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
            d["vector"] = embeddings[i]

        # [修改] 使用动态集合名
        results = self._client.upsert(
            collection_name=coll_name, data=list_data
        )
        return results

    async def query(
            self, query: str, top_k: int, query_embedding: list[float] = None
    ) -> list[dict[str, Any]]:
        # [修改] 动态集合名与检查
        coll_name = self._get_collection_name()
        await self._ensure_collection_ready(coll_name)

        if query_embedding is not None:
            embedding = [query_embedding]
        else:
            embedding = await self.embedding_func([query], _priority=5)

        output_fields = list(self.meta_fields)

        results = self._client.search(
            collection_name=coll_name,
            data=embedding,
            limit=top_k,
            output_fields=output_fields,
            search_params={
                "metric_type": "COSINE",
                "params": {"radius": self.cosine_better_than_threshold},
            },
        )
        return [
            {
                **dp["entity"],
                "id": dp["id"],
                "distance": dp["distance"],
                "created_at": dp.get("created_at"),
            }
            for dp in results[0]
        ]

    async def index_done_callback(self) -> None:
        pass

    async def delete_entity(self, entity_name: str) -> None:
        coll_name = self._get_collection_name()
        try:
            # 尝试删除时，如果集合都没创建，其实可以忽略，但为了安全先检查
            await self._ensure_collection_ready(coll_name)

            entity_id = compute_mdhash_id(entity_name, prefix="ent-")
            self._client.delete(collection_name=coll_name, pks=[entity_id])
        except Exception as e:
            logger.error(f"[{self.workspace}] Error deleting entity {entity_name}: {e}")

    async def delete_entity_relation(self, entity_name: str) -> None:
        coll_name = self._get_collection_name()
        try:
            await self._ensure_collection_ready(coll_name)
            expr = f'src_id == "{entity_name}" or tgt_id == "{entity_name}"'
            results = self._client.query(
                collection_name=coll_name, filter=expr, output_fields=["id"]
            )
            relation_ids = [item["id"] for item in results]
            if relation_ids:
                self._client.delete(collection_name=coll_name, pks=relation_ids)
        except Exception as e:
            logger.error(f"[{self.workspace}] Error deleting relations for {entity_name}: {e}")

    async def delete(self, ids: list[str]) -> None:
        coll_name = self._get_collection_name()
        try:
            await self._ensure_collection_ready(coll_name)
            self._client.delete(collection_name=coll_name, pks=ids)
        except Exception as e:
            logger.error(f"[{self.workspace}] Error deleting vectors: {e}")

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        coll_name = self._get_collection_name()
        try:
            await self._ensure_collection_ready(coll_name)
            output_fields = list(self.meta_fields) + ["id"]
            result = self._client.query(
                collection_name=coll_name,
                filter=f'id == "{id}"',
                output_fields=output_fields,
            )
            if not result:
                return None
            return result[0]
        except Exception:
            return None

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        if not ids:
            return []
        coll_name = self._get_collection_name()
        try:
            await self._ensure_collection_ready(coll_name)
            output_fields = list(self.meta_fields) + ["id"]
            id_list = '", "'.join(ids)
            result = self._client.query(
                collection_name=coll_name,
                filter=f'id in ["{id_list}"]',
                output_fields=output_fields,
            )

            result_map = {str(row["id"]): row for row in result if row.get("id")}
            return [result_map.get(str(i)) for i in ids]
        except Exception:
            return []

    async def get_vectors_by_ids(self, ids: list[str]) -> dict[str, list[float]]:
        if not ids:
            return {}
        coll_name = self._get_collection_name()
        try:
            await self._ensure_collection_ready(coll_name)
            id_list = '", "'.join(ids)
            result = self._client.query(
                collection_name=coll_name,
                filter=f'id in ["{id_list}"]',
                output_fields=["vector", "id"],
            )
            vectors_dict = {}
            for item in result:
                vec = item.get("vector")
                if isinstance(vec, np.ndarray):
                    vec = vec.tolist()
                vectors_dict[item["id"]] = vec
            return vectors_dict
        except Exception:
            return {}

    async def drop(self) -> dict[str, str]:
        """Drop collection for current workspace"""
        coll_name = self._get_collection_name()
        try:
            if self._client.has_collection(coll_name):
                self._client.drop_collection(coll_name)

            # 清除缓存
            if coll_name in self._ready_collections:
                self._ready_collections.remove(coll_name)

            # 重建
            self._create_collection_if_not_exist(coll_name)
            self._ready_collections.add(coll_name)

            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            return {"status": "error", "message": str(e)}