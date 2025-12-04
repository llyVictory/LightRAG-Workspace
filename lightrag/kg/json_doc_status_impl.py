from dataclasses import dataclass, field
import os
from typing import Any, Union, final, Dict

from lightrag.base import (
    DocProcessingStatus,
    DocStatus,
    DocStatusStorage,
)
from lightrag.utils import (
    load_json,
    logger,
    write_json,
    get_pinyin_sort_key,
)
from lightrag.exceptions import StorageNotInitializedError
from .shared_storage import (
    get_namespace_data,
    get_namespace_lock,
    get_data_init_lock,
    get_update_flag,
    set_all_update_flags,
    clear_all_update_flags,
    try_initialize_namespace,
)

# 1. 定义状态容器，存放每个 Workspace 独有的数据和锁
@dataclass
class DocStatusWorkspaceState:
    data: Any
    storage_lock: Any
    storage_updated: Any
    file_name: str

@final
@dataclass
class JsonDocStatusStorage(DocStatusStorage):
    """JSON implementation of document status storage"""

    # 2. 状态缓存池
    _states: Dict[str, DocStatusWorkspaceState] = field(default_factory=dict, init=False)

    def __post_init__(self):
        # 3. 只存根目录，不再计算具体路径
        self.working_dir = self.global_config["working_dir"]
        self._states = {}

    # 4. 核心：动态加载当前 Workspace 的状态
    async def _get_current_state(self) -> DocStatusWorkspaceState:
        current_ws = self.workspace

        if current_ws in self._states:
            return self._states[current_ws]

        # --- 初始化路径 ---
        if current_ws:
            workspace_dir = os.path.join(self.working_dir, current_ws)
        else:
            workspace_dir = self.working_dir

        os.makedirs(workspace_dir, exist_ok=True)
        file_name = os.path.join(workspace_dir, f"kv_store_{self.namespace}.json")

        # --- 初始化锁和共享数据 ---
        storage_lock = get_namespace_lock(self.namespace, workspace=current_ws)
        storage_updated = await get_update_flag(self.namespace, workspace=current_ws)

        async with get_data_init_lock():
            need_init = await try_initialize_namespace(
                self.namespace, workspace=current_ws
            )
            data = await get_namespace_data(
                self.namespace, workspace=current_ws
            )

            if need_init:
                loaded_data = load_json(file_name) or {}
                async with storage_lock:
                    data.update(loaded_data)
                    logger.info(
                        f"[{current_ws}] Process {os.getpid()} doc status load {self.namespace} with {len(loaded_data)} records"
                    )

        # --- 创建并缓存状态 ---
        state = DocStatusWorkspaceState(
            data=data,
            storage_lock=storage_lock,
            storage_updated=storage_updated,
            file_name=file_name
        )
        self._states[current_ws] = state
        return state

    async def initialize(self):
        """Initialize storage data (Warm-up default workspace)"""
        await self._get_current_state()

    async def filter_keys(self, keys: set[str]) -> set[str]:
        """Return keys that should be processed (not in storage or not successfully processed)"""
        state = await self._get_current_state()
        if state.storage_lock is None:
            raise StorageNotInitializedError("JsonDocStatusStorage")

        async with state.storage_lock:
            return set(keys) - set(state.data.keys())

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        ordered_results: list[dict[str, Any] | None] = []
        state = await self._get_current_state()

        if state.storage_lock is None:
            raise StorageNotInitializedError("JsonDocStatusStorage")

        async with state.storage_lock:
            for id in ids:
                data = state.data.get(id, None)
                if data:
                    ordered_results.append(data.copy())
                else:
                    ordered_results.append(None)
        return ordered_results

    async def get_status_counts(self) -> dict[str, int]:
        """Get counts of documents in each status"""
        counts = {status.value: 0 for status in DocStatus}
        state = await self._get_current_state()

        if state.storage_lock is None:
            raise StorageNotInitializedError("JsonDocStatusStorage")

        async with state.storage_lock:
            for doc in state.data.values():
                counts[doc["status"]] += 1
        return counts

    async def get_docs_by_status(
            self, status: DocStatus
    ) -> dict[str, DocProcessingStatus]:
        """Get all documents with a specific status"""
        result = {}
        state = await self._get_current_state()

        async with state.storage_lock:
            for k, v in state.data.items():
                if v["status"] == status.value:
                    try:
                        data = v.copy()
                        data.pop("content", None)
                        if "file_path" not in data:
                            data["file_path"] = "no-file-path"
                        if "metadata" not in data:
                            data["metadata"] = {}
                        if "error_msg" not in data:
                            data["error_msg"] = None
                        result[k] = DocProcessingStatus(**data)
                    except KeyError as e:
                        logger.error(
                            f"[{self.workspace}] Missing required field for document {k}: {e}"
                        )
                        continue
        return result

    async def get_docs_by_track_id(
            self, track_id: str
    ) -> dict[str, DocProcessingStatus]:
        """Get all documents with a specific track_id"""
        result = {}
        state = await self._get_current_state()

        async with state.storage_lock:
            for k, v in state.data.items():
                if v.get("track_id") == track_id:
                    try:
                        data = v.copy()
                        data.pop("content", None)
                        if "file_path" not in data:
                            data["file_path"] = "no-file-path"
                        if "metadata" not in data:
                            data["metadata"] = {}
                        if "error_msg" not in data:
                            data["error_msg"] = None
                        result[k] = DocProcessingStatus(**data)
                    except KeyError as e:
                        logger.error(
                            f"[{self.workspace}] Missing required field for document {k}: {e}"
                        )
                        continue
        return result

    async def index_done_callback(self) -> None:
        state = await self._get_current_state()

        async with state.storage_lock:
            if state.storage_updated.value:
                data_dict = (
                    dict(state.data) if hasattr(state.data, "_getvalue") else state.data
                )
                logger.debug(
                    f"[{self.workspace}] Process {os.getpid()} doc status writing {len(data_dict)} records to {self.namespace}"
                )

                needs_reload = write_json(data_dict, state.file_name)

                if needs_reload:
                    logger.info(
                        f"[{self.workspace}] Reloading sanitized data into shared memory for {self.namespace}"
                    )
                    cleaned_data = load_json(state.file_name)
                    if cleaned_data is not None:
                        state.data.clear()
                        state.data.update(cleaned_data)

                await clear_all_update_flags(self.namespace, workspace=self.workspace)

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """
        Importance notes for in-memory storage:
        1. Changes will be persisted to disk during the next index_done_callback
        2. update flags to notify other processes that data persistence is needed
        """
        if not data:
            return

        state = await self._get_current_state()
        logger.debug(
            f"[{self.workspace}] Inserting {len(data)} records to {self.namespace}"
        )

        if state.storage_lock is None:
            raise StorageNotInitializedError("JsonDocStatusStorage")

        async with state.storage_lock:
            for doc_id, doc_data in data.items():
                if "chunks_list" not in doc_data:
                    doc_data["chunks_list"] = []
            state.data.update(data)
            await set_all_update_flags(self.namespace, workspace=self.workspace)

        await self.index_done_callback()

    async def is_empty(self) -> bool:
        """Check if the storage is empty"""
        state = await self._get_current_state()
        if state.storage_lock is None:
            raise StorageNotInitializedError("JsonDocStatusStorage")

        async with state.storage_lock:
            return len(state.data) == 0

    async def get_by_id(self, id: str) -> Union[dict[str, Any], None]:
        state = await self._get_current_state()
        async with state.storage_lock:
            return state.data.get(id)

    async def get_docs_paginated(
            self,
            status_filter: DocStatus | None = None,
            page: int = 1,
            page_size: int = 50,
            sort_field: str = "updated_at",
            sort_direction: str = "desc",
    ) -> tuple[list[tuple[str, DocProcessingStatus]], int]:
        """Get documents with pagination support"""
        if page < 1:
            page = 1
        if page_size < 10:
            page_size = 10
        elif page_size > 200:
            page_size = 200

        if sort_field not in ["created_at", "updated_at", "id", "file_path"]:
            sort_field = "updated_at"

        if sort_direction.lower() not in ["asc", "desc"]:
            sort_direction = "desc"

        all_docs = []
        state = await self._get_current_state()

        async with state.storage_lock:
            for doc_id, doc_data in state.data.items():
                if (
                        status_filter is not None
                        and doc_data.get("status") != status_filter.value
                ):
                    continue

                try:
                    data = doc_data.copy()
                    data.pop("content", None)
                    if "file_path" not in data:
                        data["file_path"] = "no-file-path"
                    if "metadata" not in data:
                        data["metadata"] = {}
                    if "error_msg" not in data:
                        data["error_msg"] = None

                    doc_status = DocProcessingStatus(**data)

                    if sort_field == "id":
                        doc_status._sort_key = doc_id
                    elif sort_field == "file_path":
                        file_path_value = getattr(doc_status, sort_field, "")
                        doc_status._sort_key = get_pinyin_sort_key(file_path_value)
                    else:
                        doc_status._sort_key = getattr(doc_status, sort_field, "")

                    all_docs.append((doc_id, doc_status))

                except KeyError as e:
                    logger.error(
                        f"[{self.workspace}] Error processing document {doc_id}: {e}"
                    )
                    continue

        reverse_sort = sort_direction.lower() == "desc"
        all_docs.sort(
            key=lambda x: getattr(x[1], "_sort_key", ""), reverse=reverse_sort
        )

        for doc_id, doc in all_docs:
            if hasattr(doc, "_sort_key"):
                delattr(doc, "_sort_key")

        total_count = len(all_docs)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_docs = all_docs[start_idx:end_idx]

        return paginated_docs, total_count

    async def get_all_status_counts(self) -> dict[str, int]:
        """Get counts of documents in each status for all documents"""
        counts = await self.get_status_counts()
        total_count = sum(counts.values())
        counts["all"] = total_count
        return counts

    async def delete(self, doc_ids: list[str]) -> None:
        """Delete specific records from storage by their IDs"""
        state = await self._get_current_state()

        async with state.storage_lock:
            any_deleted = False
            for doc_id in doc_ids:
                result = state.data.pop(doc_id, None)
                if result is not None:
                    any_deleted = True

            if any_deleted:
                await set_all_update_flags(self.namespace, workspace=self.workspace)

    async def get_doc_by_file_path(self, file_path: str) -> Union[dict[str, Any], None]:
        """Get document by file path"""
        state = await self._get_current_state()
        if state.storage_lock is None:
            raise StorageNotInitializedError("JsonDocStatusStorage")

        async with state.storage_lock:
            for doc_id, doc_data in state.data.items():
                if doc_data.get("file_path") == file_path:
                    return doc_data
        return None

    async def drop(self) -> dict[str, str]:
        """Drop all document status data from storage and clean up resources"""
        try:
            state = await self._get_current_state()

            async with state.storage_lock:
                state.data.clear()
                await set_all_update_flags(self.namespace, workspace=self.workspace)

            await self.index_done_callback()
            logger.info(
                f"[{self.workspace}] Process {os.getpid()} drop {self.namespace}"
            )
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            logger.error(f"[{self.workspace}] Error dropping {self.namespace}: {e}")
            return {"status": "error", "message": str(e)}

    async def finalize(self):
        """Finalize storage resources (Save all workspaces)"""
        # 遍历所有已加载的工作区状态并保存
        for ws, state in self._states.items():
            try:
                async with state.storage_lock:
                    if state.storage_updated.value:
                        write_json(dict(state.data), state.file_name)
            except Exception as e:
                logger.error(f"[{ws}] Finalize save error: {e}")