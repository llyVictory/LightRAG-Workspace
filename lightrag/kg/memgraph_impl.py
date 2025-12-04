import os
import asyncio
import random
from dataclasses import dataclass
from typing import final
import configparser

from ..utils import logger
from ..base import BaseGraphStorage
from ..types import KnowledgeGraph, KnowledgeGraphNode, KnowledgeGraphEdge
from ..kg.shared_storage import get_data_init_lock
import pipmaster as pm

if not pm.is_installed("neo4j"):
    pm.install("neo4j")
from neo4j import (
    AsyncGraphDatabase,
    AsyncManagedTransaction,
)
from neo4j.exceptions import TransientError, ResultFailedError

from dotenv import load_dotenv

# use the .env that is inside the current folder
load_dotenv(dotenv_path=".env", override=False)

MAX_GRAPH_NODES = int(os.getenv("MAX_GRAPH_NODES", 1000))

config = configparser.ConfigParser()
config.read("config.ini", "utf-8")


@final
@dataclass
class MemgraphStorage(BaseGraphStorage):
    def __init__(self, namespace, global_config, embedding_func):
        # [修改] 移除 workspace 参数，不进行静态绑定
        super().__init__(
            namespace=namespace,
            # workspace=workspace, # 基类现在通过 property 获取，不需要传
            global_config=global_config,
            embedding_func=embedding_func,
        )
        self._driver = None
        self._DATABASE = None

    def _get_workspace_label(self) -> str:
        """
        Return workspace label dynamically from context.
        Uses the workspace property from BaseGraphStorage which reads ContextVar.
        If empty, fallback to 'base' to ensure valid label.
        """
        ws = self.workspace
        if not ws or not ws.strip():
            return "base"
        return ws

    async def initialize(self):
        """
        Initialize the Memgraph driver.
        Note: The driver is shared across workspaces, but index creation is workspace-specific.
        """
        async with get_data_init_lock():
            # Only initialize driver once
            if self._driver is None:
                URI = os.environ.get(
                    "MEMGRAPH_URI",
                    config.get("memgraph", "uri", fallback="bolt://localhost:7687"),
                )
                USERNAME = os.environ.get(
                    "MEMGRAPH_USERNAME", config.get("memgraph", "username", fallback="")
                )
                PASSWORD = os.environ.get(
                    "MEMGRAPH_PASSWORD", config.get("memgraph", "password", fallback="")
                )
                DATABASE = os.environ.get(
                    "MEMGRAPH_DATABASE",
                    config.get("memgraph", "database", fallback="memgraph"),
                )

                self._driver = AsyncGraphDatabase.driver(
                    URI,
                    auth=(USERNAME, PASSWORD),
                )
                self._DATABASE = DATABASE
                logger.info(f"Connected to Memgraph at {URI}")

            # [关键] 每次 initialize (通常是服务启动时) 尝试为当前 workspace 创建索引
            # 注意：如果是请求级别动态切换 workspace，这里可能覆盖不到新 workspace 的索引创建
            # 建议在 upsert 等操作前检查，或者接受 lazy creation
            try:
                async with self._driver.session(database=self._DATABASE) as session:
                    workspace_label = self._get_workspace_label()
                    try:
                        # Create index for base nodes on entity_id if it doesn't exist
                        await session.run(
                            f"""CREATE INDEX ON :{workspace_label}(entity_id)"""
                        )
                        logger.info(
                            f"[{self.workspace}] Created index on :{workspace_label}(entity_id) in Memgraph."
                        )
                    except Exception as e:
                        # Index may already exist, which is not an error
                        logger.warning(
                            f"[{self.workspace}] Index creation on :{workspace_label}(entity_id) may have failed or already exists: {e}"
                        )
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] Failed to initialize index for workspace: {e}"
                )

    async def finalize(self):
        if self._driver is not None:
            await self._driver.close()
            self._driver = None

    async def __aexit__(self, exc_type, exc, tb):
        await self.finalize()

    async def index_done_callback(self):
        # Memgraph handles persistence automatically
        pass

    # --- 以下方法逻辑基本不变，但 workspace_label 会动态变化 ---

    async def has_node(self, node_id: str) -> bool:
        if self._driver is None:
            raise RuntimeError(
                "Memgraph driver is not initialized. Call 'await initialize()' first."
            )
        async with self._driver.session(
                database=self._DATABASE, default_access_mode="READ"
        ) as session:
            result = None
            try:
                workspace_label = self._get_workspace_label()
                query = f"MATCH (n:`{workspace_label}` {{entity_id: $entity_id}}) RETURN count(n) > 0 AS node_exists"
                result = await session.run(query, entity_id=node_id)
                single_result = await result.single()
                await result.consume()
                return (
                    single_result["node_exists"] if single_result is not None else False
                )
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] Error checking node existence for {node_id}: {str(e)}"
                )
                if result is not None:
                    await result.consume()
                raise

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        if self._driver is None:
            raise RuntimeError(
                "Memgraph driver is not initialized. Call 'await initialize()' first."
            )
        async with self._driver.session(
                database=self._DATABASE, default_access_mode="READ"
        ) as session:
            result = None
            try:
                workspace_label = self._get_workspace_label()
                query = (
                    f"MATCH (a:`{workspace_label}` {{entity_id: $source_entity_id}})-[r]-(b:`{workspace_label}` {{entity_id: $target_entity_id}}) "
                    "RETURN COUNT(r) > 0 AS edgeExists"
                )
                result = await session.run(
                    query,
                    source_entity_id=source_node_id,
                    target_entity_id=target_node_id,
                )
                single_result = await result.single()
                await result.consume()
                return (
                    single_result["edgeExists"] if single_result is not None else False
                )
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] Error checking edge existence: {str(e)}"
                )
                if result is not None:
                    await result.consume()
                raise

    async def get_node(self, node_id: str) -> dict[str, str] | None:
        if self._driver is None:
            raise RuntimeError(
                "Memgraph driver is not initialized. Call 'await initialize()' first."
            )
        async with self._driver.session(
                database=self._DATABASE, default_access_mode="READ"
        ) as session:
            try:
                workspace_label = self._get_workspace_label()
                query = (
                    f"MATCH (n:`{workspace_label}` {{entity_id: $entity_id}}) RETURN n"
                )
                result = await session.run(query, entity_id=node_id)
                try:
                    records = await result.fetch(2)
                    if len(records) > 1:
                        logger.warning(
                            f"[{self.workspace}] Multiple nodes found with label '{node_id}'. Using first node."
                        )
                    if records:
                        node = records[0]["n"]
                        node_dict = dict(node)
                        # Remove workspace label from labels list
                        if "labels" in node_dict:
                            node_dict["labels"] = [
                                label
                                for label in node_dict["labels"]
                                if label != workspace_label
                            ]
                        return node_dict
                    return None
                finally:
                    await result.consume()
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] Error getting node {node_id}: {str(e)}"
                )
                raise

    async def node_degree(self, node_id: str) -> int:
        if self._driver is None:
            raise RuntimeError(
                "Memgraph driver is not initialized. Call 'await initialize()' first."
            )
        async with self._driver.session(
                database=self._DATABASE, default_access_mode="READ"
        ) as session:
            try:
                workspace_label = self._get_workspace_label()
                query = f"""
                    MATCH (n:`{workspace_label}` {{entity_id: $entity_id}})
                    OPTIONAL MATCH (n)-[r]-()
                    RETURN COUNT(r) AS degree
                """
                result = await session.run(query, entity_id=node_id)
                try:
                    record = await result.single()
                    if not record:
                        return 0
                    return record["degree"]
                finally:
                    await result.consume()
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] Error getting node degree for {node_id}: {str(e)}"
                )
                raise

    async def get_all_labels(self) -> list[str]:
        if self._driver is None:
            raise RuntimeError(
                "Memgraph driver is not initialized. Call 'await initialize()' first."
            )
        async with self._driver.session(
                database=self._DATABASE, default_access_mode="READ"
        ) as session:
            result = None
            try:
                workspace_label = self._get_workspace_label()
                query = f"""
                MATCH (n:`{workspace_label}`)
                WHERE n.entity_id IS NOT NULL
                RETURN DISTINCT n.entity_id AS label
                ORDER BY label
                """
                result = await session.run(query)
                labels = []
                async for record in result:
                    labels.append(record["label"])
                await result.consume()
                return labels
            except Exception as e:
                logger.error(f"[{self.workspace}] Error getting all labels: {str(e)}")
                if result is not None:
                    await result.consume()
                raise

    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        if self._driver is None:
            raise RuntimeError(
                "Memgraph driver is not initialized. Call 'await initialize()' first."
            )
        try:
            async with self._driver.session(
                    database=self._DATABASE, default_access_mode="READ"
            ) as session:
                results = None
                try:
                    workspace_label = self._get_workspace_label()
                    query = f"""MATCH (n:`{workspace_label}` {{entity_id: $entity_id}})
                            OPTIONAL MATCH (n)-[r]-(connected:`{workspace_label}`)
                            WHERE connected.entity_id IS NOT NULL
                            RETURN n, r, connected"""
                    results = await session.run(query, entity_id=source_node_id)

                    edges = []
                    async for record in results:
                        source_node = record["n"]
                        connected_node = record["connected"]

                        if not source_node or not connected_node:
                            continue

                        source_label = source_node.get("entity_id")
                        target_label = connected_node.get("entity_id")

                        if source_label and target_label:
                            edges.append((source_label, target_label))

                    await results.consume()
                    return edges
                except Exception as e:
                    logger.error(
                        f"[{self.workspace}] Error getting edges for node {source_node_id}: {str(e)}"
                    )
                    if results is not None:
                        await results.consume()
                    raise
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error in get_node_edges for {source_node_id}: {str(e)}"
            )
            raise

    async def get_edge(
            self, source_node_id: str, target_node_id: str
    ) -> dict[str, str] | None:
        if self._driver is None:
            raise RuntimeError(
                "Memgraph driver is not initialized. Call 'await initialize()' first."
            )
        async with self._driver.session(
                database=self._DATABASE, default_access_mode="READ"
        ) as session:
            result = None
            try:
                workspace_label = self._get_workspace_label()
                query = f"""
                MATCH (start:`{workspace_label}` {{entity_id: $source_entity_id}})-[r]-(end:`{workspace_label}` {{entity_id: $target_entity_id}})
                RETURN properties(r) as edge_properties
                """
                result = await session.run(
                    query,
                    source_entity_id=source_node_id,
                    target_entity_id=target_node_id,
                )
                records = await result.fetch(2)
                await result.consume()
                if records:
                    edge_result = dict(records[0]["edge_properties"])
                    # Fill default values
                    for key, default_value in {
                        "weight": 1.0,
                        "source_id": None,
                        "description": None,
                        "keywords": None,
                    }.items():
                        if key not in edge_result:
                            edge_result[key] = default_value
                    return edge_result
                return None
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] Error getting edge: {str(e)}"
                )
                if result is not None:
                    await result.consume()
                raise

    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        if self._driver is None:
            raise RuntimeError(
                "Memgraph driver is not initialized. Call 'await initialize()' first."
            )
        properties = node_data
        entity_type = properties["entity_type"]
        if "entity_id" not in properties:
            raise ValueError(
                "Memgraph: node properties must contain an 'entity_id' field"
            )

        max_retries = 100
        initial_wait_time = 0.2
        backoff_factor = 1.1
        jitter_factor = 0.1

        for attempt in range(max_retries):
            try:
                async with self._driver.session(database=self._DATABASE) as session:
                    workspace_label = self._get_workspace_label()

                    async def execute_upsert(tx: AsyncManagedTransaction):
                        # 确保创建索引，防止新 workspace 没有索引
                        try:
                            await tx.run(f"CREATE INDEX ON :{workspace_label}(entity_id)")
                        except Exception:
                            pass # Ignore if exists

                        query = f"""
                        MERGE (n:`{workspace_label}` {{entity_id: $entity_id}})
                        SET n += $properties
                        SET n:`{entity_type}`
                        """
                        result = await tx.run(
                            query, entity_id=node_id, properties=properties
                        )
                        await result.consume()

                    await session.execute_write(execute_upsert)
                    break

            except (TransientError, ResultFailedError) as e:
                # Retry logic for transient errors
                if attempt < max_retries - 1:
                    jitter = random.uniform(0, jitter_factor) * initial_wait_time
                    wait_time = (initial_wait_time * (backoff_factor**attempt) + jitter)
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"[{self.workspace}] Upsert node failed after retries: {e}")
                    raise
            except Exception as e:
                logger.error(f"[{self.workspace}] Upsert node error: {e}")
                raise

    async def upsert_edge(
            self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ) -> None:
        if self._driver is None:
            raise RuntimeError(
                "Memgraph driver is not initialized. Call 'await initialize()' first."
            )

        edge_properties = edge_data
        max_retries = 100
        initial_wait_time = 0.2
        backoff_factor = 1.1
        jitter_factor = 0.1

        for attempt in range(max_retries):
            try:
                async with self._driver.session(database=self._DATABASE) as session:
                    async def execute_upsert(tx: AsyncManagedTransaction):
                        workspace_label = self._get_workspace_label()
                        query = f"""
                        MATCH (source:`{workspace_label}` {{entity_id: $source_entity_id}})
                        WITH source
                        MATCH (target:`{workspace_label}` {{entity_id: $target_entity_id}})
                        MERGE (source)-[r:DIRECTED]-(target)
                        SET r += $properties
                        RETURN r
                        """
                        result = await tx.run(
                            query,
                            source_entity_id=source_node_id,
                            target_entity_id=target_node_id,
                            properties=edge_properties,
                        )
                        await result.consume()

                    await session.execute_write(execute_upsert)
                    break

            except (TransientError, ResultFailedError) as e:
                if attempt < max_retries - 1:
                    jitter = random.uniform(0, jitter_factor) * initial_wait_time
                    wait_time = (initial_wait_time * (backoff_factor**attempt) + jitter)
                    await asyncio.sleep(wait_time)
                else:
                    raise
            except Exception as e:
                logger.error(f"[{self.workspace}] Upsert edge error: {e}")
                raise

    async def delete_node(self, node_id: str) -> None:
        if self._driver is None:
            raise RuntimeError(
                "Memgraph driver is not initialized. Call 'await initialize()' first."
            )

        async def _do_delete(tx: AsyncManagedTransaction):
            workspace_label = self._get_workspace_label()
            query = f"""
            MATCH (n:`{workspace_label}` {{entity_id: $entity_id}})
            DETACH DELETE n
            """
            result = await tx.run(query, entity_id=node_id)
            await result.consume()

        try:
            async with self._driver.session(database=self._DATABASE) as session:
                await session.execute_write(_do_delete)
        except Exception as e:
            logger.error(f"[{self.workspace}] Node deletion error: {e}")
            raise

    async def remove_nodes(self, nodes: list[str]):
        if self._driver is None:
            raise RuntimeError("Memgraph driver is not initialized.")
        for node in nodes:
            await self.delete_node(node)

    async def remove_edges(self, edges: list[tuple[str, str]]):
        if self._driver is None:
            raise RuntimeError("Memgraph driver is not initialized.")

        for source, target in edges:
            async def _do_delete_edge(tx: AsyncManagedTransaction):
                workspace_label = self._get_workspace_label()
                query = f"""
                MATCH (source:`{workspace_label}` {{entity_id: $source_entity_id}})-[r]-(target:`{workspace_label}` {{entity_id: $target_entity_id}})
                DELETE r
                """
                result = await tx.run(
                    query, source_entity_id=source, target_entity_id=target
                )
                await result.consume()

            try:
                async with self._driver.session(database=self._DATABASE) as session:
                    await session.execute_write(_do_delete_edge)
            except Exception as e:
                logger.error(f"[{self.workspace}] Edge deletion error: {e}")
                raise

    async def drop(self) -> dict[str, str]:
        if self._driver is None:
            raise RuntimeError("Memgraph driver is not initialized.")
        try:
            async with self._driver.session(database=self._DATABASE) as session:
                workspace_label = self._get_workspace_label()
                query = f"MATCH (n:`{workspace_label}`) DETACH DELETE n"
                result = await session.run(query)
                await result.consume()
                logger.info(
                    f"[{self.workspace}] Dropped workspace {workspace_label}"
                )
                return {"status": "success", "message": "workspace data dropped"}
        except Exception as e:
            logger.error(f"[{self.workspace}] Drop error: {e}")
            return {"status": "error", "message": str(e)}

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        src_degree = await self.node_degree(src_id)
        trg_degree = await self.node_degree(tgt_id)
        return int(src_degree or 0) + int(trg_degree or 0)

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

        workspace_label = self._get_workspace_label()
        result = KnowledgeGraph()
        seen_nodes = set()
        seen_edges = set()

        async with self._driver.session(
                database=self._DATABASE, default_access_mode="READ"
        ) as session:
            try:
                if node_label == "*":
                    # Check total nodes
                    count_query = (
                        f"MATCH (n:`{workspace_label}`) RETURN count(n) as total"
                    )
                    count_result = await session.run(count_query)
                    count_record = await count_result.single()
                    await count_result.consume()

                    if count_record and count_record["total"] > max_nodes:
                        result.is_truncated = True

                    # Get most connected nodes
                    main_query = f"""
                    MATCH (n:`{workspace_label}`)
                    OPTIONAL MATCH (n)-[r]-()
                    WITH n, COALESCE(count(r), 0) AS degree
                    ORDER BY degree DESC
                    LIMIT $max_nodes
                    WITH collect({{node: n}}) AS filtered_nodes
                    UNWIND filtered_nodes AS node_info
                    WITH collect(node_info.node) AS kept_nodes, filtered_nodes
                    OPTIONAL MATCH (a)-[r]-(b)
                    WHERE a IN kept_nodes AND b IN kept_nodes
                    RETURN filtered_nodes AS node_info,
                        collect(DISTINCT r) AS relationships
                    """
                    result_set = await session.run(main_query, {"max_nodes": max_nodes})
                    record = await result_set.single()
                    await result_set.consume()

                else:
                    # Subgraph query
                    subgraph_query = f"""
                    MATCH (start:`{workspace_label}`)
                    WHERE start.entity_id = $entity_id

                    MATCH path = (start)-[*BFS 0..{max_depth}]-(end:`{workspace_label}`)
                    WHERE ALL(n IN nodes(path) WHERE '{workspace_label}' IN labels(n))
                    WITH collect(DISTINCT end) + start AS all_nodes_unlimited
                    WITH
                    CASE
                        WHEN size(all_nodes_unlimited) <= $max_nodes THEN all_nodes_unlimited
                        ELSE all_nodes_unlimited[0..$max_nodes]
                    END AS limited_nodes,
                    size(all_nodes_unlimited) > $max_nodes AS is_truncated

                    UNWIND limited_nodes AS n
                    MATCH (n)-[r]-(m)
                    WHERE m IN limited_nodes
                    WITH collect(DISTINCT n) AS limited_nodes, collect(DISTINCT r) AS relationships, is_truncated

                    RETURN
                    [node IN limited_nodes | {{node: node}}] AS node_info,
                    relationships,
                    is_truncated
                    """
                    result_set = await session.run(
                        subgraph_query,
                        {"entity_id": node_label, "max_nodes": max_nodes},
                    )
                    record = await result_set.single()
                    await result_set.consume()

                    if record and record.get("is_truncated"):
                        result.is_truncated = True

                if record:
                    for node_info in record["node_info"]:
                        node = node_info["node"]
                        node_id = node.id
                        if node_id not in seen_nodes:
                            result.nodes.append(
                                KnowledgeGraphNode(
                                    id=f"{node_id}",
                                    labels=[node.get("entity_id")],
                                    properties=dict(node),
                                )
                            )
                            seen_nodes.add(node_id)

                    for rel in record["relationships"]:
                        edge_id = rel.id
                        if edge_id not in seen_edges:
                            start = rel.start_node
                            end = rel.end_node
                            result.edges.append(
                                KnowledgeGraphEdge(
                                    id=f"{edge_id}",
                                    type=rel.type,
                                    source=f"{start.id}",
                                    target=f"{end.id}",
                                    properties=dict(rel),
                                )
                            )
                            seen_edges.add(edge_id)

            except Exception as e:
                logger.warning(
                    f"[{self.workspace}] Memgraph subgraph query error: {str(e)}"
                )

        return result

    async def get_all_nodes(self) -> list[dict]:
        if self._driver is None:
            raise RuntimeError("Memgraph driver is not initialized.")
        workspace_label = self._get_workspace_label()
        async with self._driver.session(
                database=self._DATABASE, default_access_mode="READ"
        ) as session:
            query = f"""
            MATCH (n:`{workspace_label}`)
            RETURN n
            """
            result = await session.run(query)
            nodes = []
            async for record in result:
                node = record["n"]
                node_dict = dict(node)
                node_dict["id"] = node_dict.get("entity_id")
                nodes.append(node_dict)
            await result.consume()
            return nodes

    async def get_all_edges(self) -> list[dict]:
        if self._driver is None:
            raise RuntimeError("Memgraph driver is not initialized.")
        workspace_label = self._get_workspace_label()
        async with self._driver.session(
                database=self._DATABASE, default_access_mode="READ"
        ) as session:
            query = f"""
            MATCH (a:`{workspace_label}`)-[r]-(b:`{workspace_label}`)
            RETURN DISTINCT a.entity_id AS source, b.entity_id AS target, properties(r) AS properties
            """
            result = await session.run(query)
            edges = []
            async for record in result:
                edge_properties = record["properties"]
                edge_properties["source"] = record["source"]
                edge_properties["target"] = record["target"]
                edges.append(edge_properties)
            await result.consume()
            return edges

    async def get_popular_labels(self, limit: int = 300) -> list[str]:
        if self._driver is None:
            raise RuntimeError("Memgraph driver is not initialized.")

        try:
            workspace_label = self._get_workspace_label()
            async with self._driver.session(
                    database=self._DATABASE, default_access_mode="READ"
            ) as session:
                query = f"""
                MATCH (n:`{workspace_label}`)
                WHERE n.entity_id IS NOT NULL
                OPTIONAL MATCH (n)-[r]-()
                WITH n.entity_id AS label, count(r) AS degree
                ORDER BY degree DESC, label ASC
                LIMIT {limit}
                RETURN label
                """
                result = await session.run(query)
                labels = []
                async for record in result:
                    labels.append(record["label"])
                await result.consume()
                return labels
        except Exception as e:
            logger.error(f"[{self.workspace}] Error getting popular labels: {str(e)}")
            return []

    async def search_labels(self, query: str, limit: int = 50) -> list[str]:
        if self._driver is None:
            raise RuntimeError("Memgraph driver is not initialized.")

        query_lower = query.lower().strip()
        if not query_lower:
            return []

        try:
            workspace_label = self._get_workspace_label()
            async with self._driver.session(
                    database=self._DATABASE, default_access_mode="READ"
            ) as session:
                cypher_query = f"""
                MATCH (n:`{workspace_label}`)
                WHERE n.entity_id IS NOT NULL
                WITH n.entity_id AS label, toLower(n.entity_id) AS label_lower
                WHERE label_lower CONTAINS $query_lower
                WITH label, label_lower,
                     CASE
                         WHEN label_lower = $query_lower THEN 1000
                         WHEN label_lower STARTS WITH $query_lower THEN 500
                         ELSE 100 - size(label)
                     END AS score
                ORDER BY score DESC, label ASC
                LIMIT {limit}
                RETURN label
                """
                result = await session.run(cypher_query, query_lower=query_lower)
                labels = []
                async for record in result:
                    labels.append(record["label"])
                await result.consume()
                return labels
        except Exception as e:
            logger.error(f"[{self.workspace}] Error searching labels: {str(e)}")
            return []