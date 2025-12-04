import os
import asyncio
from dataclasses import dataclass, field
from typing import Any, final, Dict, Optional

from lightrag.base import (
    BaseKVStorage,
)
from lightrag.utils import (
    load_json,
    logger,
    write_json,
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

# 1. 定义一个用于存放单个 Workspace 状态的小类
@dataclass
class WorkspaceState:
    data: Any
    storage_lock: Any
    storage_updated: Any
    file_name: str

@final
@dataclass
class JsonKVStorage(BaseKVStorage):
    # 2. 用于缓存不同 Workspace 状态的字典
    _states: Dict[str, WorkspaceState] = field(default_factory=dict, init=False)

    def __post_init__(self):
        # 3. 这里只保存根目录，不再计算具体文件路径
        self.working_dir = self.global_config["working_dir"]
        self._states = {}

        # 4. 【核心方法】动态获取当前 Workspace 的状态（懒加载）
    async def _get_current_state(self) -> WorkspaceState:
        # self.workspace 是通过 ContextVars 动态获取的当前工作区名
        current_ws = self.workspace

        # 如果这个工作区已经初始化过，直接返回缓存
        if current_ws in self._states:
            return self._states[current_ws]

        # --- 下面是原本 initialize 的逻辑，现在改为针对特定 Workspace 初始化 ---

        # 计算该 Workspace 的目录
        if current_ws:
            workspace_dir = os.path.join(self.working_dir, current_ws)
        else:
            workspace_dir = self.working_dir

        os.makedirs(workspace_dir, exist_ok=True)
        file_name = os.path.join(workspace_dir, f"kv_store_{self.namespace}.json")

        # 获取共享锁和数据对象
        storage_lock = get_namespace_lock(self.namespace, workspace=current_ws)
        storage_updated = await get_update_flag(self.namespace, workspace=current_ws)

        async with get_data_init_lock():
            need_init = await try_initialize_namespace(self.namespace, workspace=current_ws)
            data = await get_namespace_data(self.namespace, workspace=current_ws)

            if need_init:
                loaded_data = load_json(file_name) or {}
                async with storage_lock:
                    # 迁移旧数据结构的逻辑
                    if self.namespace.endswith("_cache"):
                        # 注意：这里需要传入 context 里的 file_name
                        loaded_data = await self._migrate_legacy_cache_structure(
                            loaded_data, file_name, current_ws
                        )

                    data.update(loaded_data)
                    data_count = len(loaded_data)
                    logger.info(
                        f"[{current_ws}] Process {os.getpid()} KV load {self.namespace} with {data_count} records"
                    )

        # 创建状态对象并缓存
        state = WorkspaceState(
            data=data,
            storage_lock=storage_lock,
            storage_updated=storage_updated,
            file_name=file_name
        )
        self._states[current_ws] = state
        return state

    async def initialize(self):
        """
        服务启动时调用。
        由于是动态加载，这里只需要预热一下默认工作区，或者什么都不做。
        """
        # 预热当前上下文（通常是 default）
        await self._get_current_state()

    async def index_done_callback(self) -> None:
        # 获取当前状态
        state = await self._get_current_state()

        async with state.storage_lock:
            if state.storage_updated.value:
                data_dict = (
                    dict(state.data) if hasattr(state.data, "_getvalue") else state.data
                )
                data_count = len(data_dict)

                logger.debug(
                    f"[{self.workspace}] Process {os.getpid()} KV writing {data_count} records to {self.namespace}"
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

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        state = await self._get_current_state() # <--- 关键修改：先拿状态

        async with state.storage_lock: # <--- 用状态里的锁
            result = state.data.get(id) # <--- 用状态里的数据
            if result:
                result = dict(result)
                result.setdefault("create_time", 0)
                result.setdefault("update_time", 0)
                result["_id"] = id
            return result

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        state = await self._get_current_state()

        async with state.storage_lock:
            results = []
            for id in ids:
                data = state.data.get(id, None)
                if data:
                    result = {k: v for k, v in data.items()}
                    result.setdefault("create_time", 0)
                    result.setdefault("update_time", 0)
                    result["_id"] = id
                    results.append(result)
                else:
                    results.append(None)
            return results

    async def filter_keys(self, keys: set[str]) -> set[str]:
        state = await self._get_current_state()
        async with state.storage_lock:
            return set(keys) - set(state.data.keys())

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        if not data:
            return

        import time
        current_time = int(time.time())

        state = await self._get_current_state() # 获取状态

        logger.debug(
            f"[{self.workspace}] Inserting {len(data)} records to {self.namespace}"
        )

        # 这里的检查要改一下，检查 state 是否有效
        if state is None:
            raise StorageNotInitializedError("JsonKVStorage")

        async with state.storage_lock:
            for k, v in data.items():
                if self.namespace.endswith("text_chunks"):
                    if "llm_cache_list" not in v:
                        v["llm_cache_list"] = []

                if k in state.data:
                    v["update_time"] = current_time
                else:
                    v["create_time"] = current_time
                    v["update_time"] = current_time
                v["_id"] = k

            state.data.update(data)
            await set_all_update_flags(self.namespace, workspace=self.workspace)

    async def delete(self, ids: list[str]) -> None:
        state = await self._get_current_state()
        async with state.storage_lock:
            any_deleted = False
            for doc_id in ids:
                result = state.data.pop(doc_id, None)
                if result is not None:
                    any_deleted = True

            if any_deleted:
                await set_all_update_flags(self.namespace, workspace=self.workspace)

    async def is_empty(self) -> bool:
        state = await self._get_current_state()
        async with state.storage_lock:
            return len(state.data) == 0

    async def drop(self) -> dict[str, str]:
        try:
            state = await self._get_current_state()
            async with state.storage_lock:
                state.data.clear()
                await set_all_update_flags(self.namespace, workspace=self.workspace)

            await self.index_done_callback()
            logger.info(
                f"[{self.workspace}] Process {os.getpid()} drop {self.namespace}"
            )

            # 可选：Drop 之后可能需要从缓存中移除这个 state，看具体业务需求
            # self._states.pop(self.workspace, None)

            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            logger.error(f"[{self.workspace}] Error dropping {self.namespace}: {e}")
            return {"status": "error", "message": str(e)}

    # 注意：这个方法我加了 file_name 和 workspace 参数，因为它们不能再从 self 里取了
    async def _migrate_legacy_cache_structure(self, data: dict, file_name: str, workspace: str) -> dict:
        from lightrag.utils import generate_cache_key

        if not data:
            return data

        first_key = next(iter(data.keys()))
        if ":" in first_key and len(first_key.split(":")) == 3:
            return data

        migrated_data = {}
        migration_count = 0

        for key, value in data.items():
            if isinstance(value, dict) and all(
                    isinstance(v, dict) and "return" in v for v in value.values()
            ):
                mode = key
                for cache_hash, cache_entry in value.items():
                    cache_type = cache_entry.get("cache_type", "extract")
                    flattened_key = generate_cache_key(mode, cache_type, cache_hash)
                    migrated_data[flattened_key] = cache_entry
                    migration_count += 1
            else:
                migrated_data[key] = value

        if migration_count > 0:
            logger.info(
                f"[{workspace}] Migrated {migration_count} legacy cache entries to flattened structure"
            )
            # 使用传入的 file_name
            needs_reload = write_json(migrated_data, file_name)

            if needs_reload:
                logger.info(
                    f"[{workspace}] Reloading sanitized migration data for {self.namespace}"
                )
                cleaned_data = load_json(file_name)
                if cleaned_data is not None:
                    return cleaned_data

        return migrated_data

    async def finalize(self):
        """
        服务关闭时调用。
        需要遍历所有活跃的 workspace 进行保存。
        """
        # 遍历缓存中所有的 state 进行保存
        for workspace, state in self._states.items():
            if self.namespace.endswith("_cache"):
                # 这里稍微 tricky，因为 index_done_callback 依赖 context
                # 但 finalize 时 context 可能不对。
                # 最好是手动把 index_done_callback 的逻辑抽取出来，或者在这里模拟 context
                # 简单起见，我们直接复用 index_done_callback 的核心保存逻辑:
                async with state.storage_lock:
                    if state.storage_updated.value:
                        write_json(dict(state.data), state.file_name)