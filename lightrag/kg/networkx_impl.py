import os
from dataclasses import dataclass, field
from typing import final, Dict, Any

from lightrag.types import KnowledgeGraph, KnowledgeGraphNode, KnowledgeGraphEdge
from lightrag.utils import logger
from lightrag.base import BaseGraphStorage
import networkx as nx
from .shared_storage import (
    get_namespace_lock,
    get_update_flag,
    set_all_update_flags,
)

from dotenv import load_dotenv

# use the .env that is inside the current folder
load_dotenv(dotenv_path=".env", override=False)

# 1. 定义状态容器
@dataclass
class NetworkXWorkspaceState:
    graph: nx.Graph
    graphml_xml_file: str
    storage_lock: Any
    storage_updated: Any

@final
@dataclass
class NetworkXStorage(BaseGraphStorage):
    # 2. 状态缓存池
    _states: Dict[str, NetworkXWorkspaceState] = field(default_factory=dict, init=False)

    @staticmethod
    def load_nx_graph(file_name) -> nx.Graph:
        if os.path.exists(file_name):
            return nx.read_graphml(file_name)
        return None

    @staticmethod
    def write_nx_graph(graph: nx.Graph, file_name, workspace="_"):
        logger.info(
            f"[{workspace}] Writing graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        )
        nx.write_graphml(graph, file_name)

    def __post_init__(self):
        # 只保存配置，不进行具体初始化
        self.working_dir = self.global_config["working_dir"]
        self._states = {}

    # 3. 核心：动态加载当前 Workspace 的状态
    async def _get_current_state(self) -> NetworkXWorkspaceState:
        current_ws = self.workspace

        if current_ws in self._states:
            return self._states[current_ws]

        # --- 计算路径 ---
        if current_ws:
            workspace_dir = os.path.join(self.working_dir, current_ws)
        else:
            workspace_dir = self.working_dir

        os.makedirs(workspace_dir, exist_ok=True)
        graphml_xml_file = os.path.join(
            workspace_dir, f"graph_{self.namespace}.graphml"
        )

        # --- 获取锁 ---
        storage_updated = await get_update_flag(
            self.namespace, workspace=current_ws
        )
        storage_lock = get_namespace_lock(
            self.namespace, workspace=current_ws
        )

        # --- 加载图 ---
        preloaded_graph = NetworkXStorage.load_nx_graph(graphml_xml_file)
        if preloaded_graph is not None:
            logger.info(
                f"[{current_ws}] Loaded graph from {graphml_xml_file} with {preloaded_graph.number_of_nodes()} nodes, {preloaded_graph.number_of_edges()} edges"
            )
        else:
            logger.info(
                f"[{current_ws}] Created new empty graph file: {graphml_xml_file}"
            )
        graph = preloaded_graph or nx.Graph()

        # --- 缓存状态 ---
        state = NetworkXWorkspaceState(
            graph=graph,
            graphml_xml_file=graphml_xml_file,
            storage_lock=storage_lock,
            storage_updated=storage_updated
        )
        self._states[current_ws] = state
        return state

    async def initialize(self):
        """Initialize storage data (Warm-up default workspace)"""
        await self._get_current_state()

    async def _get_graph(self):
        """Get the graph object for the current workspace, reloading if needed."""
        state = await self._get_current_state()

        # Acquire lock to prevent concurrent read and write
        async with state.storage_lock:
            # Check if data needs to be reloaded
            if state.storage_updated.value:
                logger.info(
                    f"[{self.workspace}] Process {os.getpid()} reloading graph {state.graphml_xml_file} due to modifications by another process"
                )
                # Reload data
                state.graph = (
                        NetworkXStorage.load_nx_graph(state.graphml_xml_file) or nx.Graph()
                )
                # Reset update flag
                state.storage_updated.value = False

            return state.graph

    async def has_node(self, node_id: str) -> bool:
        graph = await self._get_graph()
        return graph.has_node(node_id)

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        graph = await self._get_graph()
        return graph.has_edge(source_node_id, target_node_id)

    async def get_node(self, node_id: str) -> dict[str, str] | None:
        graph = await self._get_graph()
        return graph.nodes.get(node_id)

    async def node_degree(self, node_id: str) -> int:
        graph = await self._get_graph()
        return graph.degree(node_id)

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        graph = await self._get_graph()
        src_degree = graph.degree(src_id) if graph.has_node(src_id) else 0
        tgt_degree = graph.degree(tgt_id) if graph.has_node(tgt_id) else 0
        return src_degree + tgt_degree

    async def get_edge(
            self, source_node_id: str, target_node_id: str
    ) -> dict[str, str] | None:
        graph = await self._get_graph()
        return graph.edges.get((source_node_id, target_node_id))

    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        graph = await self._get_graph()
        if graph.has_node(source_node_id):
            return list(graph.edges(source_node_id))
        return None

    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        graph = await self._get_graph()
        graph.add_node(node_id, **node_data)

    async def upsert_edge(
            self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ) -> None:
        graph = await self._get_graph()
        graph.add_edge(source_node_id, target_node_id, **edge_data)

    async def delete_node(self, node_id: str) -> None:
        graph = await self._get_graph()
        if graph.has_node(node_id):
            graph.remove_node(node_id)
            logger.debug(f"[{self.workspace}] Node {node_id} deleted from the graph")
        else:
            logger.warning(
                f"[{self.workspace}] Node {node_id} not found in the graph for deletion"
            )

    async def remove_nodes(self, nodes: list[str]):
        graph = await self._get_graph()
        for node in nodes:
            if graph.has_node(node):
                graph.remove_node(node)

    async def remove_edges(self, edges: list[tuple[str, str]]):
        graph = await self._get_graph()
        for source, target in edges:
            if graph.has_edge(source, target):
                graph.remove_edge(source, target)

    async def get_all_labels(self) -> list[str]:
        graph = await self._get_graph()
        labels = set()
        for node in graph.nodes():
            labels.add(str(node))
        return sorted(list(labels))

    async def get_popular_labels(self, limit: int = 300) -> list[str]:
        graph = await self._get_graph()
        degrees = dict(graph.degree())
        sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
        popular_labels = [str(node) for node, _ in sorted_nodes[:limit]]
        logger.debug(
            f"[{self.workspace}] Retrieved {len(popular_labels)} popular labels (limit: {limit})"
        )
        return popular_labels

    async def search_labels(self, query: str, limit: int = 50) -> list[str]:
        graph = await self._get_graph()
        query_lower = query.lower().strip()

        if not query_lower:
            return []

        matches = []
        for node in graph.nodes():
            node_str = str(node)
            node_lower = node_str.lower()

            if query_lower not in node_lower:
                continue

            if node_lower == query_lower:
                score = 1000
            elif node_lower.startswith(query_lower):
                score = 500
            else:
                score = 100 - len(node_str)
                if f" {query_lower}" in node_lower or f"_{query_lower}" in node_lower:
                    score += 50

            matches.append((node_str, score))

        matches.sort(key=lambda x: (-x[1], x[0]))
        search_results = [match[0] for match in matches[:limit]]

        logger.debug(
            f"[{self.workspace}] Search query '{query}' returned {len(search_results)} results (limit: {limit})"
        )
        return search_results

    async def get_knowledge_graph(
            self,
            node_label: str,
            max_depth: int = 3,
            max_nodes: int = None,
    ) -> KnowledgeGraph:
        # (代码逻辑保持不变，因为 _get_graph 已经动态化)
        # Get max_nodes from global_config if not provided
        if max_nodes is None:
            max_nodes = self.global_config.get("max_graph_nodes", 1000)
        else:
            max_nodes = min(max_nodes, self.global_config.get("max_graph_nodes", 1000))

        graph = await self._get_graph()
        result = KnowledgeGraph()

        if node_label == "*":
            degrees = dict(graph.degree())
            sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)

            if len(sorted_nodes) > max_nodes:
                result.is_truncated = True
                logger.info(
                    f"[{self.workspace}] Graph truncated: {len(sorted_nodes)} nodes found, limited to {max_nodes}"
                )

            limited_nodes = [node for node, _ in sorted_nodes[:max_nodes]]
            subgraph = graph.subgraph(limited_nodes)
        else:
            if node_label not in graph:
                logger.warning(
                    f"[{self.workspace}] Node {node_label} not found in the graph"
                )
                return KnowledgeGraph()

            bfs_nodes = []
            visited = set()
            queue = [(node_label, 0, graph.degree(node_label))]
            has_unexplored_neighbors = False

            while queue and len(bfs_nodes) < max_nodes:
                current_depth = queue[0][1]
                current_level_nodes = []
                while queue and queue[0][1] == current_depth:
                    current_level_nodes.append(queue.pop(0))

                current_level_nodes.sort(key=lambda x: x[2], reverse=True)

                for current_node, depth, degree in current_level_nodes:
                    if current_node not in visited:
                        visited.add(current_node)
                        bfs_nodes.append(current_node)

                        if depth < max_depth:
                            neighbors = list(graph.neighbors(current_node))
                            unvisited_neighbors = [
                                n for n in neighbors if n not in visited
                            ]
                            for neighbor in unvisited_neighbors:
                                neighbor_degree = graph.degree(neighbor)
                                queue.append((neighbor, depth + 1, neighbor_degree))
                        else:
                            neighbors = list(graph.neighbors(current_node))
                            unvisited_neighbors = [
                                n for n in neighbors if n not in visited
                            ]
                            if unvisited_neighbors:
                                has_unexplored_neighbors = True

                    if len(bfs_nodes) >= max_nodes:
                        break

            if (queue and len(bfs_nodes) >= max_nodes) or has_unexplored_neighbors:
                if len(bfs_nodes) >= max_nodes:
                    result.is_truncated = True
                    logger.info(
                        f"[{self.workspace}] Graph truncated: max_nodes limit {max_nodes} reached"
                    )
                else:
                    logger.info(
                        f"[{self.workspace}] Graph truncated: found {len(bfs_nodes)} nodes within max_depth {max_depth}"
                    )

            subgraph = graph.subgraph(bfs_nodes)

        seen_nodes = set()
        seen_edges = set()
        for node in subgraph.nodes():
            if str(node) in seen_nodes:
                continue

            node_data = dict(subgraph.nodes[node])
            labels = []
            if "entity_type" in node_data:
                if isinstance(node_data["entity_type"], list):
                    labels.extend(node_data["entity_type"])
                else:
                    labels.append(node_data["entity_type"])

            node_properties = {k: v for k, v in node_data.items()}

            result.nodes.append(
                KnowledgeGraphNode(
                    id=str(node), labels=[str(node)], properties=node_properties
                )
            )
            seen_nodes.add(str(node))

        for edge in subgraph.edges():
            source, target = edge
            if str(source) > str(target):
                source, target = target, source
            edge_id = f"{source}-{target}"
            if edge_id in seen_edges:
                continue

            edge_data = dict(subgraph.edges[edge])

            result.edges.append(
                KnowledgeGraphEdge(
                    id=edge_id,
                    type="DIRECTED",
                    source=str(source),
                    target=str(target),
                    properties=edge_data,
                )
            )
            seen_edges.add(edge_id)

        logger.info(
            f"[{self.workspace}] Subgraph query successful | Node count: {len(result.nodes)} | Edge count: {len(result.edges)}"
        )
        return result

    async def get_all_nodes(self) -> list[Dict[str, Any]]:
        """Get all nodes in the graph."""
        graph = await self._get_graph()
        nodes = []
        for node, data in graph.nodes(data=True):
            node_data = data.copy()
            node_data["id"] = str(node)
            nodes.append(node_data)
        return nodes

    async def get_all_edges(self) -> list[Dict[str, Any]]:
        """Get all edges in the graph."""
        graph = await self._get_graph()
        edges = []
        for source, target, data in graph.edges(data=True):
            edge_data = data.copy()
            edge_data["source"] = str(source)
            edge_data["target"] = str(target)
            edges.append(edge_data)
        return edges

    async def index_done_callback(self) -> bool:
        """Save data to disk for current workspace"""
        state = await self._get_current_state()

        async with state.storage_lock:
            if state.storage_updated.value:
                logger.info(
                    f"[{self.workspace}] Graph was updated by another process, reloading..."
                )
                state.graph = (
                        NetworkXStorage.load_nx_graph(state.graphml_xml_file) or nx.Graph()
                )
                state.storage_updated.value = False
                return False

        async with state.storage_lock:
            try:
                NetworkXStorage.write_nx_graph(
                    state.graph, state.graphml_xml_file, self.workspace
                )
                await set_all_update_flags(self.namespace, workspace=self.workspace)
                state.storage_updated.value = False
                return True
            except Exception as e:
                logger.error(f"[{self.workspace}] Error saving graph: {e}")
                return False

        return True

    async def drop(self) -> dict[str, str]:
        """Drop all graph data for current workspace"""
        try:
            state = await self._get_current_state()

            async with state.storage_lock:
                if os.path.exists(state.graphml_xml_file):
                    os.remove(state.graphml_xml_file)
                state.graph = nx.Graph()

                await set_all_update_flags(self.namespace, workspace=self.workspace)
                state.storage_updated.value = False
                logger.info(
                    f"[{self.workspace}] Process {os.getpid()} drop graph file:{state.graphml_xml_file}"
                )
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error dropping graph file: {e}"
            )
            return {"status": "error", "message": str(e)}

    async def finalize(self):
        """Finalize storage resources (Save all workspaces)"""
        for ws, state in self._states.items():
            try:
                # 简单的 finalize 保存逻辑，如果需要更严格的并发控制，可加锁
                NetworkXStorage.write_nx_graph(state.graph, state.graphml_xml_file, ws)
            except Exception as e:
                logger.error(f"[{ws}] Finalize save error: {e}")