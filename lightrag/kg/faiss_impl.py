import os
import time
import asyncio
from typing import Any, final, Dict, Optional
import json
import numpy as np
from dataclasses import dataclass, field

from lightrag.utils import logger, compute_mdhash_id
from lightrag.base import BaseVectorStorage

from .shared_storage import (
    get_namespace_lock,
    get_update_flag,
    set_all_update_flags,
)

# You must manually install faiss-cpu or faiss-gpu before using FAISS vector db
import faiss  # type: ignore

# 1. 定义状态容器，存放每个 Workspace 独有的 Index 和元数据
@dataclass
class FaissWorkspaceState:
    index: Any  # faiss.Index
    id_to_meta: Dict[int, dict]
    faiss_index_file: str
    meta_file: str
    storage_lock: Any
    storage_updated: Any

@final
@dataclass
class FaissVectorDBStorage(BaseVectorStorage):
    """
    A Faiss-based Vector DB Storage for LightRAG.
    Uses cosine similarity by storing normalized vectors in a Faiss index with inner product search.
    """
    # 2. 状态缓存池
    _states: Dict[str, FaissWorkspaceState] = field(default_factory=dict, init=False)

    def __post_init__(self):
        # Grab config values if available
        kwargs = self.global_config.get("vector_db_storage_cls_kwargs", {})
        cosine_threshold = kwargs.get("cosine_better_than_threshold")
        if cosine_threshold is None:
            raise ValueError(
                "cosine_better_than_threshold must be specified in vector_db_storage_cls_kwargs"
            )
        self.cosine_better_than_threshold = cosine_threshold

        # Embedding dimension (e.g. 768) must match your embedding function
        self._dim = self.embedding_func.embedding_dim
        self._max_batch_size = self.global_config["embedding_batch_num"]

        # 保存根目录，不再计算具体路径
        self.working_dir = self.global_config["working_dir"]
        self._states = {}

    # 3. 核心：动态加载当前 Workspace 的状态
    async def _get_current_state(self) -> FaissWorkspaceState:
        current_ws = self.workspace

        if current_ws in self._states:
            return self._states[current_ws]

        # --- 初始化路径 ---
        if current_ws:
            workspace_dir = os.path.join(self.working_dir, current_ws)
        else:
            workspace_dir = self.working_dir

        os.makedirs(workspace_dir, exist_ok=True)
        faiss_index_file = os.path.join(
            workspace_dir, f"faiss_index_{self.namespace}.index"
        )
        meta_file = faiss_index_file + ".meta.json"

        # --- 初始化锁 ---
        storage_updated = await get_update_flag(self.namespace, workspace=current_ws)
        storage_lock = get_namespace_lock(self.namespace, workspace=current_ws)

        # --- 加载数据 (原 _load_faiss_index 逻辑) ---
        index = None
        id_to_meta = {}

        if os.path.exists(faiss_index_file):
            try:
                index = faiss.read_index(faiss_index_file)
                if os.path.exists(meta_file):
                    with open(meta_file, "r", encoding="utf-8") as f:
                        stored_dict = json.load(f)
                    # Convert string keys back to int
                    for fid_str, meta in stored_dict.items():
                        id_to_meta[int(fid_str)] = meta

                logger.info(
                    f"[{current_ws}] Faiss index loaded with {index.ntotal} vectors"
                )
            except Exception as e:
                logger.error(
                    f"[{current_ws}] Failed to load Faiss index: {e}"
                )

        # 如果加载失败或文件不存在，创建新的
        if index is None:
            logger.warning(f"[{current_ws}] Starting with an empty Faiss index.")
            index = faiss.IndexFlatIP(self._dim)
            id_to_meta = {}

        # --- 创建并缓存状态 ---
        state = FaissWorkspaceState(
            index=index,
            id_to_meta=id_to_meta,
            faiss_index_file=faiss_index_file,
            meta_file=meta_file,
            storage_lock=storage_lock,
            storage_updated=storage_updated
        )

        self._states[current_ws] = state
        return state

    async def initialize(self):
        """Initialize storage data (Warm-up default workspace)"""
        await self._get_current_state()

    # --- 辅助方法：重新加载 (用于检测到其他进程更新时) ---
    def _reload_state_data(self, state: FaissWorkspaceState):
        logger.info(f"[{self.workspace}] Reloading FAISS index due to update...")
        try:
            if os.path.exists(state.faiss_index_file):
                state.index = faiss.read_index(state.faiss_index_file)
                state.id_to_meta = {}
                if os.path.exists(state.meta_file):
                    with open(state.meta_file, "r", encoding="utf-8") as f:
                        stored_dict = json.load(f)
                    for fid_str, meta in stored_dict.items():
                        state.id_to_meta[int(fid_str)] = meta
            else:
                state.index = faiss.IndexFlatIP(self._dim)
                state.id_to_meta = {}
        except Exception as e:
            logger.error(f"[{self.workspace}] Reload error: {e}")
            state.index = faiss.IndexFlatIP(self._dim)
            state.id_to_meta = {}

    async def _get_index(self, state: FaissWorkspaceState):
        """Check if the storage should be reloaded"""
        async with state.storage_lock:
            if state.storage_updated.value:
                self._reload_state_data(state)
                state.storage_updated.value = False
            return state.index

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        if not data:
            return

        state = await self._get_current_state() # 获取状态

        logger.debug(
            f"[{self.workspace}] FAISS: Inserting {len(data)} to {self.namespace}"
        )

        current_time = int(time.time())

        # Prepare data for embedding
        list_data = []
        contents = []
        for k, v in data.items():
            meta = {mf: v[mf] for mf in self.meta_fields if mf in v}
            meta["__id__"] = k
            meta["__created_at__"] = current_time
            list_data.append(meta)
            contents.append(v["content"])

        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]

        embedding_tasks = [self.embedding_func(batch) for batch in batches]
        embeddings_list = await asyncio.gather(*embedding_tasks)

        embeddings = np.concatenate(embeddings_list, axis=0)
        if len(embeddings) != len(list_data):
            logger.error(f"Mismatch embedding size")
            return []

        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)

        # Update logic using state
        existing_ids_to_remove = []
        for meta, emb in zip(list_data, embeddings):
            # Pass state to helper
            faiss_internal_id = self._find_faiss_id_by_custom_id(state, meta["__id__"])
            if faiss_internal_id is not None:
                existing_ids_to_remove.append(faiss_internal_id)

        if existing_ids_to_remove:
            await self._remove_faiss_ids(state, existing_ids_to_remove)

        # Add new vectors
        index = await self._get_index(state)
        start_idx = index.ntotal
        index.add(embeddings)

        for i, meta in enumerate(list_data):
            fid = start_idx + i
            meta["__vector__"] = embeddings[i].tolist()
            state.id_to_meta.update({fid: meta})

        logger.debug(
            f"[{self.workspace}] Upserted {len(list_data)} vectors into Faiss index."
        )
        return [m["__id__"] for m in list_data]

    async def query(
            self, query: str, top_k: int, query_embedding: list[float] = None
    ) -> list[dict[str, Any]]:

        state = await self._get_current_state() # 获取状态

        if query_embedding is not None:
            embedding = np.array([query_embedding], dtype=np.float32)
        else:
            embedding = await self.embedding_func([query], _priority=5)
            embedding = np.array(embedding, dtype=np.float32)

        faiss.normalize_L2(embedding)

        index = await self._get_index(state)
        distances, indices = index.search(embedding, top_k)

        distances = distances[0]
        indices = indices[0]

        results = []
        for dist, idx in zip(distances, indices):
            if idx == -1:
                continue
            if dist < self.cosine_better_than_threshold:
                continue

            # 使用 state.id_to_meta
            meta = state.id_to_meta.get(idx, {})
            filtered_meta = {k: v for k, v in meta.items() if k != "__vector__"}
            results.append(
                {
                    **filtered_meta,
                    "id": meta.get("__id__"),
                    "distance": float(dist),
                    "created_at": meta.get("__created_at__"),
                }
            )

        return results

    @property
    def client_storage(self):
        # Debug helper: returns dict of all loaded workspaces
        return {ws: list(st.id_to_meta.values()) for ws, st in self._states.items()}

    async def delete(self, ids: list[str]):
        state = await self._get_current_state()
        logger.debug(
            f"[{self.workspace}] Deleting {len(ids)} vectors from {self.namespace}"
        )
        to_remove = []
        for cid in ids:
            fid = self._find_faiss_id_by_custom_id(state, cid)
            if fid is not None:
                to_remove.append(fid)

        if to_remove:
            await self._remove_faiss_ids(state, to_remove)
        logger.debug(
            f"[{self.workspace}] Successfully deleted {len(to_remove)} vectors"
        )

    async def delete_entity(self, entity_name: str) -> None:
        entity_id = compute_mdhash_id(entity_name, prefix="ent-")
        await self.delete([entity_id])

    async def delete_entity_relation(self, entity_name: str) -> None:
        state = await self._get_current_state()
        relations = []
        for fid, meta in state.id_to_meta.items():
            if meta.get("src_id") == entity_name or meta.get("tgt_id") == entity_name:
                relations.append(fid)
        if relations:
            await self._remove_faiss_ids(state, relations)

    # --- Internal Helpers (需增加 state 参数) ---

    def _find_faiss_id_by_custom_id(self, state: FaissWorkspaceState, custom_id: str):
        for fid, meta in state.id_to_meta.items():
            if meta.get("__id__") == custom_id:
                return fid
        return None

    async def _remove_faiss_ids(self, state: FaissWorkspaceState, fid_list):
        keep_fids = [fid for fid in state.id_to_meta if fid not in fid_list]

        vectors_to_keep = []
        new_id_to_meta = {}
        for new_fid, old_fid in enumerate(keep_fids):
            vec_meta = state.id_to_meta[old_fid]
            vectors_to_keep.append(vec_meta["__vector__"])
            new_id_to_meta[new_fid] = vec_meta

        async with state.storage_lock:
            state.index = faiss.IndexFlatIP(self._dim)
            if vectors_to_keep:
                arr = np.array(vectors_to_keep, dtype=np.float32)
                state.index.add(arr)
            state.id_to_meta = new_id_to_meta

    def _save_faiss_index(self, state: FaissWorkspaceState):
        faiss.write_index(state.index, state.faiss_index_file)

        serializable_dict = {}
        for fid, meta in state.id_to_meta.items():
            serializable_dict[str(fid)] = meta

        with open(state.meta_file, "w", encoding="utf-8") as f:
            json.dump(serializable_dict, f)

    async def index_done_callback(self) -> None:
        # 获取当前状态
        state = await self._get_current_state()

        async with state.storage_lock:
            if state.storage_updated.value:
                logger.warning(
                    f"[{self.workspace}] FAISS {self.namespace} updated by another process, reloading..."
                )
                self._reload_state_data(state)
                state.storage_updated.value = False
                return False

        async with state.storage_lock:
            try:
                self._save_faiss_index(state)
                await set_all_update_flags(self.namespace, workspace=self.workspace)
                state.storage_updated.value = False
            except Exception as e:
                logger.error(f"[{self.workspace}] Error saving FAISS: {e}")
                return False
        return True

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        state = await self._get_current_state()
        fid = self._find_faiss_id_by_custom_id(state, id)
        if fid is None:
            return None

        metadata = state.id_to_meta.get(fid, {})
        if not metadata:
            return None

        filtered_metadata = {k: v for k, v in metadata.items() if k != "__vector__"}
        return {
            **filtered_metadata,
            "id": metadata.get("__id__"),
            "created_at": metadata.get("__created_at__"),
        }

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        if not ids:
            return []

        state = await self._get_current_state()
        results = []
        for id in ids:
            record = None
            fid = self._find_faiss_id_by_custom_id(state, id)
            if fid is not None:
                metadata = state.id_to_meta.get(fid)
                if metadata:
                    filtered_metadata = {
                        k: v for k, v in metadata.items() if k != "__vector__"
                    }
                    record = {
                        **filtered_metadata,
                        "id": metadata.get("__id__"),
                        "created_at": metadata.get("__created_at__"),
                    }
            results.append(record)
        return results

    async def get_vectors_by_ids(self, ids: list[str]) -> dict[str, list[float]]:
        if not ids:
            return {}

        state = await self._get_current_state()
        vectors_dict = {}
        for id in ids:
            fid = self._find_faiss_id_by_custom_id(state, id)
            if fid is not None and fid in state.id_to_meta:
                metadata = state.id_to_meta[fid]
                if "__vector__" in metadata:
                    vectors_dict[id] = metadata["__vector__"]
        return vectors_dict

    async def drop(self) -> dict[str, str]:
        try:
            state = await self._get_current_state()
            async with state.storage_lock:
                state.index = faiss.IndexFlatIP(self._dim)
                state.id_to_meta = {}

                if os.path.exists(state.faiss_index_file):
                    os.remove(state.faiss_index_file)
                if os.path.exists(state.meta_file):
                    os.remove(state.meta_file)

                # Re-init empty (not strictly necessary as we cleared above, but safe)
                state.index = faiss.IndexFlatIP(self._dim)
                state.id_to_meta = {}

                await set_all_update_flags(self.namespace, workspace=self.workspace)
                state.storage_updated.value = False

            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            logger.error(f"[{self.workspace}] Drop error: {e}")
            return {"status": "error", "message": str(e)}

    async def finalize(self):
        """Finalize storage resources (Save all workspaces)"""
        for ws, state in self._states.items():
            try:
                # 简单的 finalize 保存逻辑，如果需要更严格的并发控制，可加锁
                self._save_faiss_index(state)
            except Exception as e:
                logger.error(f"[{ws}] Finalize save error: {e}")