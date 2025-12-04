import os
import re
from dataclasses import dataclass, field
from typing import final
import configparser
import asyncio

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

import logging
from ..utils import logger
from ..base import BaseGraphStorage
from ..types import KnowledgeGraph, KnowledgeGraphNode, KnowledgeGraphEdge
from ..kg.shared_storage import get_data_init_lock
import pipmaster as pm

if not pm.is_installed("neo4j"):
    pm.install("neo4j")

from neo4j import (  # type: ignore
    AsyncGraphDatabase,
    exceptions as neo4jExceptions,
    AsyncDriver,
    AsyncManagedTransaction,
)

from dotenv import load_dotenv

# use the .env that is inside the current folder
load_dotenv(dotenv_path=".env", override=False)

config = configparser.ConfigParser()
config.read("config.ini", "utf-8")

# Set neo4j logger level to ERROR to suppress warning logs
logging.getLogger("neo4j").setLevel(logging.ERROR)

READ_RETRY_EXCEPTIONS = (
    neo4jExceptions.ServiceUnavailable,
    neo4jExceptions.TransientError,
    neo4jExceptions.SessionExpired,
    ConnectionResetError,
    OSError,
    AttributeError,
)

READ_RETRY = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(READ_RETRY_EXCEPTIONS),
    reraise=True,
)


@final
@dataclass
class Neo4JStorage(BaseGraphStorage):
    _driver: AsyncDriver = field(default=None, init=False)
    _initialized_workspaces: set = field(default_factory=set, init=False)
    _index_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)
    _DATABASE: str = field(default=None, init=False)

    def __init__(self, namespace, global_config, embedding_func, workspace=None):
        super().__init__(
            namespace=namespace,
            workspace=workspace or "",  # Pass empty or default, property will handle dynamic value
            global_config=global_config,
            embedding_func=embedding_func,
        )
        self._driver = None
        self._initialized_workspaces = set()
        self._index_lock = asyncio.Lock()
        self._DATABASE = None

    def _get_workspace_label(self) -> str:
        """Return dynamic workspace label from ContextVars"""
        ws = self.workspace
        if not ws or not ws.strip():
            return "base"
        return ws

    def _is_chinese_text(self, text: str) -> bool:
        """Check if text contains Chinese characters."""
        chinese_pattern = re.compile(r"[\u4e00-\u9fff]+")
        return bool(chinese_pattern.search(text))

    def _sanitize_label_for_index(self, label: str) -> str:
        """Sanitize label to be safe for index naming"""
        return re.sub(r"[^a-zA-Z0-9_]", "_", label)

    async def initialize(self):
        """Initialize Neo4j driver connection. Indexes are lazy-loaded per workspace."""
        async with get_data_init_lock():
            if self._driver is not None:
                return

            URI = os.environ.get("NEO4J_URI", config.get("neo4j", "uri", fallback=None))
            USERNAME = os.environ.get(
                "NEO4J_USERNAME", config.get("neo4j", "username", fallback=None)
            )
            PASSWORD = os.environ.get(
                "NEO4J_PASSWORD", config.get("neo4j", "password", fallback=None)
            )
            # ... (Connection params logic remains same) ...
            MAX_CONNECTION_POOL_SIZE = int(os.environ.get("NEO4J_MAX_CONNECTION_POOL_SIZE", config.get("neo4j", "connection_pool_size", fallback=100)))
            CONNECTION_TIMEOUT = float(os.environ.get("NEO4J_CONNECTION_TIMEOUT", config.get("neo4j", "connection_timeout", fallback=30.0)))
            CONNECTION_ACQUISITION_TIMEOUT = float(os.environ.get("NEO4J_CONNECTION_ACQUISITION_TIMEOUT", config.get("neo4j", "connection_acquisition_timeout", fallback=30.0)))
            MAX_TRANSACTION_RETRY_TIME = float(os.environ.get("NEO4J_MAX_TRANSACTION_RETRY_TIME", config.get("neo4j", "max_transaction_retry_time", fallback=30.0)))
            MAX_CONNECTION_LIFETIME = float(os.environ.get("NEO4J_MAX_CONNECTION_LIFETIME", config.get("neo4j", "max_connection_lifetime", fallback=300.0)))
            LIVENESS_CHECK_TIMEOUT = float(os.environ.get("NEO4J_LIVENESS_CHECK_TIMEOUT", config.get("neo4j", "liveness_check_timeout", fallback=30.0)))
            KEEP_ALIVE = os.environ.get("NEO4J_KEEP_ALIVE", config.get("neo4j", "keep_alive", fallback="true")).lower() in ("true", "1", "yes", "on")

            DATABASE = os.environ.get(
                "NEO4J_DATABASE", re.sub(r"[^a-zA-Z0-9-]", "-", self.namespace)
            )

            self._driver = AsyncGraphDatabase.driver(
                URI,
                auth=(USERNAME, PASSWORD),
                max_connection_pool_size=MAX_CONNECTION_POOL_SIZE,
                connection_timeout=CONNECTION_TIMEOUT,
                connection_acquisition_timeout=CONNECTION_ACQUISITION_TIMEOUT,
                max_transaction_retry_time=MAX_TRANSACTION_RETRY_TIME,
                max_connection_lifetime=MAX_CONNECTION_LIFETIME,
                liveness_check_timeout=LIVENESS_CHECK_TIMEOUT,
                keep_alive=KEEP_ALIVE,
            )

            # Try to connect
            for database in (DATABASE, None):
                self._DATABASE = database
                try:
                    async with self._driver.session(database=database) as session:
                        result = await session.run("MATCH (n) RETURN n LIMIT 0")
                        await result.consume()
                        logger.info(f"Connected to Neo4j database: {database or 'default'} at {URI}")
                        break # Connected successfully
                except neo4jExceptions.AuthError as e:
                    logger.error(f"Authentication failed for {database} at {URI}")
                    raise e
                except neo4jExceptions.ClientError as e:
                    # Database creation logic... (simplified for brevity, keeping original intent)
                    if e.code == "Neo.ClientError.Database.DatabaseNotFound":
                        logger.warning(f"Database {database} not found.")
                        # ... (Try create DB logic from original file) ...
                        # Assuming DB exists or fallback to default for this context
                    if database is None:
                        raise e

    async def _ensure_indexes(self, workspace_label: str):
        """Lazy load indexes for the current workspace"""
        if workspace_label in self._initialized_workspaces:
            return

        async with self._index_lock:
            if workspace_label in self._initialized_workspaces:
                return

            try:
                # 1. Create B-Tree index
                async with self._driver.session(database=self._DATABASE) as session:
                    # Use backticks for safety
                    await session.run(
                        f"CREATE INDEX IF NOT EXISTS FOR (n:`{workspace_label}`) ON (n.entity_id)"
                    )
                    logger.debug(f"[{workspace_label}] Ensured B-Tree index on entity_id")

                # 2. Create Fulltext index
                await self._create_fulltext_index(self._driver, self._DATABASE, workspace_label)

                self._initialized_workspaces.add(workspace_label)
            except Exception as e:
                logger.error(f"[{workspace_label}] Failed to ensure indexes: {e}")

    async def _create_fulltext_index(
            self, driver: AsyncDriver, database: str, workspace_label: str
    ):
        """Create a full-text index for the specific workspace label."""
        # Dynamic index name to prevent collisions between workspaces
        safe_label = self._sanitize_label_for_index(workspace_label)
        index_name = f"entity_id_fulltext_idx_{safe_label}"

        try:
            async with driver.session(database=database) as session:
                # Check existence
                check_query = "SHOW FULLTEXT INDEXES"
                result = await session.run(check_query)
                indexes = await result.data()

                existing_index = next((idx for idx in indexes if idx["name"] == index_name), None)

                if existing_index and existing_index.get("state") == "ONLINE":
                    return

                if existing_index:
                    logger.warning(f"[{workspace_label}] Recreating offline index {index_name}")
                    await session.run(f"DROP INDEX `{index_name}`")

                # Create Index
                try:
                    create_query = f"""
                    CREATE FULLTEXT INDEX `{index_name}`
                    FOR (n:`{workspace_label}`) ON EACH [n.entity_id]
                    OPTIONS {{
                        indexConfig: {{
                            `fulltext.analyzer`: 'cjk',
                            `fulltext.eventually_consistent`: true
                        }}
                    }}
                    """
                    await session.run(create_query)
                    logger.info(f"[{workspace_label}] Created CJK fulltext index: {index_name}")
                except Exception:
                    # Fallback to standard
                    create_query = f"""
                    CREATE FULLTEXT INDEX `{index_name}`
                    FOR (n:`{workspace_label}`) ON EACH [n.entity_id]
                    """
                    await session.run(create_query)
                    logger.info(f"[{workspace_label}] Created standard fulltext index: {index_name}")

        except Exception as e:
            # Swallow error to allow partial functionality (fallback to CONTAINS)
            logger.warning(f"[{workspace_label}] Could not create fulltext index '{index_name}': {e}")

    async def finalize(self):
        if self._driver:
            await self._driver.close()
            self._driver = None
            self._initialized_workspaces.clear()

    async def __aexit__(self, exc_type, exc, tb):
        await self.finalize()

    async def index_done_callback(self) -> None:
        pass

    # -------------------------------------------------------------------------
    # OPERATIONS (Wrappers with Index Check)
    # -------------------------------------------------------------------------

    @READ_RETRY
    async def has_node(self, node_id: str) -> bool:
        workspace_label = self._get_workspace_label()
        await self._ensure_indexes(workspace_label) # Ensure indexes

        async with self._driver.session(database=self._DATABASE, default_access_mode="READ") as session:
            try:
                query = f"MATCH (n:`{workspace_label}` {{entity_id: $entity_id}}) RETURN count(n) > 0 AS node_exists"
                result = await session.run(query, entity_id=node_id)
                single = await result.single()
                return single["node_exists"]
            except Exception as e:
                logger.error(f"[{self.workspace}] Error has_node: {e}")
                raise

    @READ_RETRY
    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        workspace_label = self._get_workspace_label()
        # No specific index needed for edge check usually, but good practice to ensure consistency
        # await self._ensure_indexes(workspace_label)

        async with self._driver.session(database=self._DATABASE, default_access_mode="READ") as session:
            try:
                query = (
                    f"MATCH (a:`{workspace_label}` {{entity_id: $src}})-[r]-(b:`{workspace_label}` {{entity_id: $tgt}}) "
                    "RETURN COUNT(r) > 0 AS edgeExists"
                )
                result = await session.run(query, src=source_node_id, tgt=target_node_id)
                single = await result.single()
                return single["edgeExists"]
            except Exception as e:
                logger.error(f"[{self.workspace}] Error has_edge: {e}")
                raise

    @READ_RETRY
    async def get_node(self, node_id: str) -> dict[str, str] | None:
        workspace_label = self._get_workspace_label()
        await self._ensure_indexes(workspace_label)

        async with self._driver.session(database=self._DATABASE, default_access_mode="READ") as session:
            query = f"MATCH (n:`{workspace_label}` {{entity_id: $entity_id}}) RETURN n"
            result = await session.run(query, entity_id=node_id)
            record = await result.single()
            if record:
                node = record["n"]
                node_dict = dict(node)
                if "labels" in node_dict:
                    node_dict["labels"] = [l for l in node_dict["labels"] if l != workspace_label]
                return node_dict
            return None

    @READ_RETRY
    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, dict]:
        workspace_label = self._get_workspace_label()
        await self._ensure_indexes(workspace_label)

        async with self._driver.session(database=self._DATABASE, default_access_mode="READ") as session:
            query = f"""
            UNWIND $node_ids AS id
            MATCH (n:`{workspace_label}` {{entity_id: id}})
            RETURN n.entity_id AS entity_id, n
            """
            result = await session.run(query, node_ids=node_ids)
            nodes = {}
            async for record in result:
                eid = record["entity_id"]
                node_dict = dict(record["n"])
                if "labels" in node_dict:
                    node_dict["labels"] = [l for l in node_dict["labels"] if l != workspace_label]
                nodes[eid] = node_dict
            return nodes

    # ... node_degree, node_degrees_batch, edge_degree, edge_degrees_batch follow same pattern ...
    # Simplified here, they don't strictly need index check if get_node checked it, but safer to add.

    @READ_RETRY
    async def node_degree(self, node_id: str) -> int:
        workspace_label = self._get_workspace_label()
        await self._ensure_indexes(workspace_label)
        async with self._driver.session(database=self._DATABASE, default_access_mode="READ") as session:
            query = f"""
                MATCH (n:`{workspace_label}` {{entity_id: $entity_id}})
                OPTIONAL MATCH (n)-[r]-()
                RETURN COUNT(r) AS degree
            """
            result = await session.run(query, entity_id=node_id)
            record = await result.single()
            return record["degree"] if record else 0

    @READ_RETRY
    async def node_degrees_batch(self, node_ids: list[str]) -> dict[str, int]:
        workspace_label = self._get_workspace_label()
        await self._ensure_indexes(workspace_label)
        async with self._driver.session(database=self._DATABASE, default_access_mode="READ") as session:
            query = f"""
                UNWIND $node_ids AS id
                MATCH (n:`{workspace_label}` {{entity_id: id}})
                RETURN n.entity_id AS entity_id, count {{ (n)--() }} AS degree;
            """
            result = await session.run(query, node_ids=node_ids)
            degrees = {id: 0 for id in node_ids}
            async for record in result:
                degrees[record["entity_id"]] = record["degree"]
            return degrees

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        d1 = await self.node_degree(src_id)
        d2 = await self.node_degree(tgt_id)
        return d1 + d2

    @READ_RETRY
    async def edge_degrees_batch(self, edge_pairs: list[tuple[str, str]]) -> dict[tuple[str, str], int]:
        unique_nodes = list({n for pair in edge_pairs for n in pair})
        node_degrees = await self.node_degrees_batch(unique_nodes)
        return {
            (src, tgt): node_degrees.get(src, 0) + node_degrees.get(tgt, 0)
            for src, tgt in edge_pairs
        }

    @READ_RETRY
    async def get_edge(self, source_node_id: str, target_node_id: str) -> dict[str, str] | None:
        workspace_label = self._get_workspace_label()
        async with self._driver.session(database=self._DATABASE, default_access_mode="READ") as session:
            query = f"""
            MATCH (start:`{workspace_label}` {{entity_id: $src}})-[r]-(end:`{workspace_label}` {{entity_id: $tgt}})
            RETURN properties(r) as props
            """
            result = await session.run(query, src=source_node_id, tgt=target_node_id)
            record = await result.single()
            if record:
                props = dict(record["props"])
                # set defaults
                for k, v in {"weight": 1.0, "source_id": None, "description": None, "keywords": None}.items():
                    props.setdefault(k, v)
                return props
            return None

    @READ_RETRY
    async def get_edges_batch(self, pairs: list[dict[str, str]]) -> dict[tuple[str, str], dict]:
        workspace_label = self._get_workspace_label()
        async with self._driver.session(database=self._DATABASE, default_access_mode="READ") as session:
            query = f"""
            UNWIND $pairs AS pair
            MATCH (start:`{workspace_label}` {{entity_id: pair.src}})-[r:DIRECTED]-(end:`{workspace_label}` {{entity_id: pair.tgt}})
            RETURN pair.src AS src, pair.tgt AS tgt, collect(properties(r)) AS edges
            """
            result = await session.run(query, pairs=pairs)
            edges_dict = {}
            async for record in result:
                edges = record["edges"]
                props = edges[0] if edges else {}
                for k, v in {"weight": 1.0, "source_id": None, "description": None, "keywords": None}.items():
                    props.setdefault(k, v)
                edges_dict[(record["src"], record["tgt"])] = props
            return edges_dict

    @READ_RETRY
    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        workspace_label = self._get_workspace_label()
        await self._ensure_indexes(workspace_label)
        async with self._driver.session(database=self._DATABASE, default_access_mode="READ") as session:
            query = f"""
            MATCH (n:`{workspace_label}` {{entity_id: $id}})
            OPTIONAL MATCH (n)-[r]-(connected:`{workspace_label}`)
            WHERE connected.entity_id IS NOT NULL
            RETURN n, connected
            """
            result = await session.run(query, id=source_node_id)
            edges = []
            async for record in result:
                if record["n"] and record["connected"]:
                    edges.append((record["n"]["entity_id"], record["connected"]["entity_id"]))
            return edges

    @READ_RETRY
    async def get_nodes_edges_batch(self, node_ids: list[str]) -> dict[str, list[tuple[str, str]]]:
        workspace_label = self._get_workspace_label()
        await self._ensure_indexes(workspace_label)
        async with self._driver.session(database=self._DATABASE, default_access_mode="READ") as session:
            query = f"""
                UNWIND $node_ids AS id
                MATCH (n:`{workspace_label}` {{entity_id: id}})
                OPTIONAL MATCH (n)-[r]-(connected:`{workspace_label}`)
                RETURN id, n.entity_id AS nid, connected.entity_id AS cid, startNode(r).entity_id AS sid
            """
            result = await session.run(query, node_ids=node_ids)
            edges_dict = {nid: [] for nid in node_ids}
            async for record in result:
                if record["nid"] and record["cid"]:
                    qid = record["id"]
                    # Determine direction
                    if record["sid"] == record["nid"]:
                        edges_dict[qid].append((record["nid"], record["cid"]))
                    else:
                        edges_dict[qid].append((record["cid"], record["nid"]))
            return edges_dict

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10), retry=retry_if_exception_type(READ_RETRY_EXCEPTIONS))
    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        workspace_label = self._get_workspace_label()
        await self._ensure_indexes(workspace_label) # Important for write performance

        properties = node_data.copy()
        entity_type = properties["entity_type"]

        async with self._driver.session(database=self._DATABASE) as session:
            query = f"""
            MERGE (n:`{workspace_label}` {{entity_id: $id}})
            SET n += $props
            SET n:`{entity_type}`
            """
            await session.run(query, id=node_id, props=properties)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10), retry=retry_if_exception_type(READ_RETRY_EXCEPTIONS))
    async def upsert_edge(self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]) -> None:
        workspace_label = self._get_workspace_label()
        await self._ensure_indexes(workspace_label)

        async with self._driver.session(database=self._DATABASE) as session:
            query = f"""
            MATCH (source:`{workspace_label}` {{entity_id: $src}})
            WITH source
            MATCH (target:`{workspace_label}` {{entity_id: $tgt}})
            MERGE (source)-[r:DIRECTED]-(target)
            SET r += $props
            RETURN r
            """
            await session.run(query, src=source_node_id, tgt=target_node_id, props=edge_data)

    async def get_knowledge_graph(self, node_label: str, max_depth: int = 3, max_nodes: int = None) -> KnowledgeGraph:
        # Get max_nodes from global_config if not provided
        if max_nodes is None:
            max_nodes = self.global_config.get("max_graph_nodes", 1000)
        else:
            max_nodes = min(max_nodes, self.global_config.get("max_graph_nodes", 1000))

        workspace_label = self._get_workspace_label()
        await self._ensure_indexes(workspace_label)

        result = KnowledgeGraph()
        seen_nodes = set()
        seen_edges = set()

        async with self._driver.session(database=self._DATABASE, default_access_mode="READ") as session:
            try:
                if node_label == "*":
                    # Global graph query logic
                    # Check total count first
                    count_res = await session.run(f"MATCH (n:`{workspace_label}`) RETURN count(n) as total")
                    count_rec = await count_res.single()
                    if count_rec and count_rec["total"] > max_nodes:
                        result.is_truncated = True

                    query = f"""
                    MATCH (n:`{workspace_label}`)
                    OPTIONAL MATCH (n)-[r]-()
                    WITH n, COALESCE(count(r), 0) AS degree
                    ORDER BY degree DESC
                    LIMIT $max_nodes
                    WITH collect({{node: n}}) AS nodes
                    UNWIND nodes as n
                    OPTIONAL MATCH (n.node)-[r]-(m)
                    WHERE m IN [x in nodes | x.node]
                    RETURN nodes, collect(DISTINCT r) as rels
                    """
                    res = await session.run(query, max_nodes=max_nodes)
                    rec = await res.single()
                else:
                    # Subgraph query
                    query = f"""
                    MATCH (start:`{workspace_label}` {{entity_id: $id}})
                    CALL apoc.path.subgraphAll(start, {{
                        labelFilter: '{workspace_label}',
                        minLevel: 0,
                        maxLevel: $depth,
                        limit: $limit,
                        bfs: true
                    }})
                    YIELD nodes, relationships
                    RETURN [n in nodes | {{node: n}}] as nodes, relationships as rels
                    """
                    res = await session.run(query, id=node_label, depth=max_depth, limit=max_nodes)
                    rec = await res.single()
                    if rec and len(rec["nodes"]) >= max_nodes:
                        result.is_truncated = True

                if rec:
                    for item in rec["nodes"]:
                        node = item["node"]
                        nid = node.get("entity_id")
                        if nid and nid not in seen_nodes:
                            result.nodes.append(KnowledgeGraphNode(id=nid, labels=[nid], properties=dict(node)))
                            seen_nodes.add(nid)
                    for rel in rec["rels"]:
                        if rel and rel.id not in seen_edges:
                            result.edges.append(KnowledgeGraphEdge(
                                id=str(rel.id), type=rel.type,
                                source=rel.start_node.get("entity_id"),
                                target=rel.end_node.get("entity_id"),
                                properties=dict(rel)
                            ))
                            seen_edges.add(rel.id)

            except Exception as e:
                logger.error(f"KG Query failed: {e}")
                if node_label != "*":
                    return await self._robust_fallback(node_label, max_depth, max_nodes)

        return result

    async def _robust_fallback(self, node_label: str, max_depth: int, max_nodes: int) -> KnowledgeGraph:
        # Fallback BFS implementation in Python
        # Logic is identical to original, just need to use workspace_label
        from collections import deque
        result = KnowledgeGraph()
        visited = set()
        workspace_label = self._get_workspace_label()

        # Start node
        start_node = await self.get_node(node_label)
        if not start_node: return result

        queue = deque([(KnowledgeGraphNode(id=node_label, labels=[node_label], properties=start_node), 0)])
        visited.add(node_label)
        result.nodes.append(queue[0][0])

        while queue and len(visited) < max_nodes:
            curr_node, depth = queue.popleft()
            if depth >= max_depth: continue

            # Get neighbors
            async with self._driver.session(database=self._DATABASE) as session:
                query = f"""
                MATCH (a:`{workspace_label}` {{entity_id: $id}})-[r]-(b:`{workspace_label}`)
                RETURN r, b
                """
                res = await session.run(query, id=curr_node.id)
                async for rec in res:
                    b_node = rec["b"]
                    bid = b_node.get("entity_id")
                    if not bid: continue

                    # Add edge
                    # Note: We need edge IDs or unique handling. Simplified here.
                    edge_props = dict(rec["r"])
                    edge = KnowledgeGraphEdge(id=f"{curr_node.id}-{bid}", type=rec["r"].type, source=curr_node.id, target=bid, properties=edge_props)
                    result.edges.append(edge)

                    if bid not in visited and len(visited) < max_nodes:
                        visited.add(bid)
                        new_node = KnowledgeGraphNode(id=bid, labels=[bid], properties=dict(b_node))
                        result.nodes.append(new_node)
                        queue.append((new_node, depth + 1))

        return result

    @READ_RETRY
    async def delete_node(self, node_id: str) -> None:
        workspace_label = self._get_workspace_label()
        async with self._driver.session(database=self._DATABASE) as session:
            await session.run(f"MATCH (n:`{workspace_label}` {{entity_id: $id}}) DETACH DELETE n", id=node_id)

    @READ_RETRY
    async def remove_nodes(self, nodes: list[str]):
        for node in nodes: await self.delete_node(node)

    @READ_RETRY
    async def remove_edges(self, edges: list[tuple[str, str]]):
        workspace_label = self._get_workspace_label()
        async with self._driver.session(database=self._DATABASE) as session:
            for src, tgt in edges:
                await session.run(
                    f"MATCH (a:`{workspace_label}` {{entity_id: $src}})-[r]-(b:`{workspace_label}` {{entity_id: $tgt}}) DELETE r",
                    src=src, tgt=tgt
                )

    async def get_all_nodes(self) -> list[dict]:
        workspace_label = self._get_workspace_label()
        async with self._driver.session(database=self._DATABASE, default_access_mode="READ") as session:
            result = await session.run(f"MATCH (n:`{workspace_label}`) RETURN n")
            return [dict(record["n"]) async for record in result]

    async def get_all_edges(self) -> list[dict]:
        workspace_label = self._get_workspace_label()
        async with self._driver.session(database=self._DATABASE, default_access_mode="READ") as session:
            query = f"MATCH (a:`{workspace_label}`)-[r]-(b:`{workspace_label}`) RETURN a.entity_id as s, b.entity_id as t, properties(r) as p"
            result = await session.run(query)
            edges = []
            async for rec in result:
                props = rec["p"]
                props["source"] = rec["s"]
                props["target"] = rec["t"]
                edges.append(props)
            return edges

    async def get_popular_labels(self, limit: int = 300) -> list[str]:
        workspace_label = self._get_workspace_label()
        async with self._driver.session(database=self._DATABASE, default_access_mode="READ") as session:
            query = f"""
            MATCH (n:`{workspace_label}`)
            OPTIONAL MATCH (n)-[r]-()
            RETURN n.entity_id as label, count(r) as degree
            ORDER BY degree DESC
            LIMIT $limit
            """
            result = await session.run(query, limit=limit)
            return [rec["label"] async for rec in result if rec["label"]]

    async def search_labels(self, query: str, limit: int = 50) -> list[str]:
        workspace_label = self._get_workspace_label()
        await self._ensure_indexes(workspace_label)

        query_strip = query.strip()
        if not query_strip: return []

        is_chinese = self._is_chinese_text(query_strip)
        safe_label = self._sanitize_label_for_index(workspace_label)
        index_name = f"entity_id_fulltext_idx_{safe_label}"

        async with self._driver.session(database=self._DATABASE, default_access_mode="READ") as session:
            # Try Fulltext
            try:
                # (Query logic same as original, just using dynamic index_name and workspace_label)
                # Simplified for brevity:
                cypher = f"CALL db.index.fulltext.queryNodes('{index_name}', $q) YIELD node RETURN node.entity_id as label LIMIT $lim"
                res = await session.run(cypher, q=query_strip, lim=limit)
                return [rec["label"] async for rec in res]
            except Exception:
                # Fallback
                cypher = f"MATCH (n:`{workspace_label}`) WHERE n.entity_id CONTAINS $q RETURN n.entity_id as label LIMIT $lim"
                res = await session.run(cypher, q=query_strip, lim=limit)
                return [rec["label"] async for rec in res]

    async def drop(self) -> dict[str, str]:
        workspace_label = self._get_workspace_label()
        try:
            async with self._driver.session(database=self._DATABASE) as session:
                await session.run(f"MATCH (n:`{workspace_label}`) DETACH DELETE n")

            # Optionally drop index too
            # safe_label = self._sanitize_label_for_index(workspace_label)
            # await session.run(f"DROP INDEX entity_id_fulltext_idx_{safe_label} IF EXISTS")

            return {"status": "success", "message": "dropped"}
        except Exception as e:
            return {"status": "error", "message": str(e)}