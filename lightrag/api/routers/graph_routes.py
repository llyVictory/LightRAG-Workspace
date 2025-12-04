"""
This module contains all graph-related routes for the LightRAG API.
"""

from typing import Optional, Dict, Any
import traceback
from fastapi import APIRouter, Depends, Query, HTTPException
from pydantic import BaseModel, Field

from lightrag.utils import logger
from ..utils_api import get_combined_auth_dependency
from lightrag.context_utils import set_current_workspace, reset_current_workspace

router = APIRouter(tags=["graph"])


class EntityUpdateRequest(BaseModel):
    entity_name: str
    updated_data: Dict[str, Any]
    allow_rename: bool = False
    allow_merge: bool = False
    workspace: str = Field(default="default", description="Workspace for data isolation")


class RelationUpdateRequest(BaseModel):
    source_id: str
    target_id: str
    updated_data: Dict[str, Any]
    workspace: str = Field(default="default", description="Workspace for data isolation")


class EntityMergeRequest(BaseModel):
    entities_to_change: list[str] = Field(
        ...,
        description="List of entity names to be merged and deleted. These are typically duplicate or misspelled entities.",
        min_length=1,
        examples=[["Elon Msk", "Ellon Musk"]],
    )
    entity_to_change_into: str = Field(
        ...,
        description="Target entity name that will receive all relationships from the source entities. This entity will be preserved.",
        min_length=1,
        examples=["Elon Musk"],
    )
    workspace: str = Field(default="default", description="Workspace for data isolation")


class EntityCreateRequest(BaseModel):
    entity_name: str = Field(
        ...,
        description="Unique name for the new entity",
        min_length=1,
        examples=["Tesla"],
    )
    entity_data: Dict[str, Any] = Field(
        ...,
        description="Dictionary containing entity properties. Common fields include 'description' and 'entity_type'.",
        examples=[
            {
                "description": "Electric vehicle manufacturer",
                "entity_type": "ORGANIZATION",
            }
        ],
    )
    workspace: str = Field(default="default", description="Workspace for data isolation")


class RelationCreateRequest(BaseModel):
    source_entity: str = Field(
        ...,
        description="Name of the source entity. This entity must already exist in the knowledge graph.",
        min_length=1,
        examples=["Elon Musk"],
    )
    target_entity: str = Field(
        ...,
        description="Name of the target entity. This entity must already exist in the knowledge graph.",
        min_length=1,
        examples=["Tesla"],
    )
    relation_data: Dict[str, Any] = Field(
        ...,
        description="Dictionary containing relationship properties. Common fields include 'description', 'keywords', and 'weight'.",
        examples=[
            {
                "description": "Elon Musk is the CEO of Tesla",
                "keywords": "CEO, founder",
                "weight": 1.0,
            }
        ],
    )
    workspace: str = Field(default="default", description="Workspace for data isolation")


def create_graph_routes(rag, api_key: Optional[str] = None):
    combined_auth = get_combined_auth_dependency(api_key)

    @router.get("/graph/label/list", dependencies=[Depends(combined_auth)])
    async def get_graph_labels(
            workspace: str = Query("default", description="Workspace for data isolation")
    ):
        """
        Get all graph labels

        Returns:
            List[str]: List of graph labels
        """
        token = set_current_workspace(workspace)
        try:
            return await rag.get_graph_labels()
        except Exception as e:
            logger.error(f"Error getting graph labels: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500, detail=f"Error getting graph labels: {str(e)}"
            )
        finally:
            reset_current_workspace(token)

    @router.get("/graph/label/popular", dependencies=[Depends(combined_auth)])
    async def get_popular_labels(
            limit: int = Query(
                300, description="Maximum number of popular labels to return", ge=1, le=1000
            ),
            workspace: str = Query("default", description="Workspace for data isolation")
    ):
        """
        Get popular labels by node degree (most connected entities)

        Args:
            limit (int): Maximum number of labels to return (default: 300, max: 1000)

        Returns:
            List[str]: List of popular labels sorted by degree (highest first)
        """
        token = set_current_workspace(workspace)
        try:
            return await rag.chunk_entity_relation_graph.get_popular_labels(limit)
        except Exception as e:
            logger.error(f"Error getting popular labels: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500, detail=f"Error getting popular labels: {str(e)}"
            )
        finally:
            reset_current_workspace(token)

    @router.get("/graph/label/search", dependencies=[Depends(combined_auth)])
    async def search_labels(
            q: str = Query(..., description="Search query string"),
            limit: int = Query(
                50, description="Maximum number of search results to return", ge=1, le=100
            ),
            workspace: str = Query("default", description="Workspace for data isolation")
    ):
        """
        Search labels with fuzzy matching

        Args:
            q (str): Search query string
            limit (int): Maximum number of results to return (default: 50, max: 100)

        Returns:
            List[str]: List of matching labels sorted by relevance
        """
        token = set_current_workspace(workspace)
        try:
            return await rag.chunk_entity_relation_graph.search_labels(q, limit)
        except Exception as e:
            logger.error(f"Error searching labels with query '{q}': {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500, detail=f"Error searching labels: {str(e)}"
            )
        finally:
            reset_current_workspace(token)

    @router.get("/graphs", dependencies=[Depends(combined_auth)])
    async def get_knowledge_graph(
            label: str = Query(..., description="Label to get knowledge graph for"),
            max_depth: int = Query(3, description="Maximum depth of graph", ge=1),
            max_nodes: int = Query(1000, description="Maximum nodes to return", ge=1),
            workspace: str = Query("default", description="Workspace for data isolation")
    ):
        """
        Retrieve a connected subgraph of nodes where the label includes the specified label.
        """
        token = set_current_workspace(workspace)
        try:
            # Log the label parameter to check for leading spaces
            logger.debug(
                f"get_knowledge_graph called with label: '{label}' (length: {len(label)}, repr: {repr(label)})"
            )

            return await rag.get_knowledge_graph(
                node_label=label,
                max_depth=max_depth,
                max_nodes=max_nodes,
            )
        except Exception as e:
            logger.error(f"Error getting knowledge graph for label '{label}': {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500, detail=f"Error getting knowledge graph: {str(e)}"
            )
        finally:
            reset_current_workspace(token)

    @router.get("/graph/entity/exists", dependencies=[Depends(combined_auth)])
    async def check_entity_exists(
            name: str = Query(..., description="Entity name to check"),
            workspace: str = Query("default", description="Workspace for data isolation")
    ):
        """
        Check if an entity with the given name exists in the knowledge graph
        """
        token = set_current_workspace(workspace)
        try:
            exists = await rag.chunk_entity_relation_graph.has_node(name)
            return {"exists": exists}
        except Exception as e:
            logger.error(f"Error checking entity existence for '{name}': {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500, detail=f"Error checking entity existence: {str(e)}"
            )
        finally:
            reset_current_workspace(token)

    @router.post("/graph/entity/edit", dependencies=[Depends(combined_auth)])
    async def update_entity(request: EntityUpdateRequest):
        """
        Update an entity's properties in the knowledge graph
        """
        token = set_current_workspace(request.workspace)
        try:
            result = await rag.aedit_entity(
                entity_name=request.entity_name,
                updated_data=request.updated_data,
                allow_rename=request.allow_rename,
                allow_merge=request.allow_merge,
            )

            # Extract operation_summary from result, with fallback for backward compatibility
            operation_summary = result.get(
                "operation_summary",
                {
                    "merged": False,
                    "merge_status": "not_attempted",
                    "merge_error": None,
                    "operation_status": "success",
                    "target_entity": None,
                    "final_entity": request.updated_data.get(
                        "entity_name", request.entity_name
                    ),
                    "renamed": request.updated_data.get(
                        "entity_name", request.entity_name
                    )
                               != request.entity_name,
                },
            )

            # Separate entity data from operation_summary for clean response
            entity_data = dict(result)
            entity_data.pop("operation_summary", None)

            # Generate appropriate response message based on merge status
            response_message = (
                f"Entity merged successfully into '{operation_summary['final_entity']}'"
                if operation_summary.get("merged")
                else "Entity updated successfully"
            )
            return {
                "status": "success",
                "message": response_message,
                "data": entity_data,
                "operation_summary": operation_summary,
            }
        except ValueError as ve:
            logger.error(
                f"Validation error updating entity '{request.entity_name}': {str(ve)}"
            )
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(f"Error updating entity '{request.entity_name}': {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500, detail=f"Error updating entity: {str(e)}"
            )
        finally:
            reset_current_workspace(token)

    @router.post("/graph/relation/edit", dependencies=[Depends(combined_auth)])
    async def update_relation(request: RelationUpdateRequest):
        """Update a relation's properties in the knowledge graph"""
        token = set_current_workspace(request.workspace)
        try:
            result = await rag.aedit_relation(
                source_entity=request.source_id,
                target_entity=request.target_id,
                updated_data=request.updated_data,
            )
            return {
                "status": "success",
                "message": "Relation updated successfully",
                "data": result,
            }
        except ValueError as ve:
            logger.error(
                f"Validation error updating relation between '{request.source_id}' and '{request.target_id}': {str(ve)}"
            )
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(
                f"Error updating relation between '{request.source_id}' and '{request.target_id}': {str(e)}"
            )
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500, detail=f"Error updating relation: {str(e)}"
            )
        finally:
            reset_current_workspace(token)

    @router.post("/graph/entity/create", dependencies=[Depends(combined_auth)])
    async def create_entity(request: EntityCreateRequest):
        """Create a new entity in the knowledge graph"""
        token = set_current_workspace(request.workspace)
        try:
            result = await rag.acreate_entity(
                entity_name=request.entity_name,
                entity_data=request.entity_data,
            )

            return {
                "status": "success",
                "message": f"Entity '{request.entity_name}' created successfully",
                "data": result,
            }
        except ValueError as ve:
            logger.error(
                f"Validation error creating entity '{request.entity_name}': {str(ve)}"
            )
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(f"Error creating entity '{request.entity_name}': {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500, detail=f"Error creating entity: {str(e)}"
            )
        finally:
            reset_current_workspace(token)

    @router.post("/graph/relation/create", dependencies=[Depends(combined_auth)])
    async def create_relation(request: RelationCreateRequest):
        """Create a new relationship between two entities in the knowledge graph"""
        token = set_current_workspace(request.workspace)
        try:
            result = await rag.acreate_relation(
                source_entity=request.source_entity,
                target_entity=request.target_entity,
                relation_data=request.relation_data,
            )

            return {
                "status": "success",
                "message": f"Relation created successfully between '{request.source_entity}' and '{request.target_entity}'",
                "data": result,
            }
        except ValueError as ve:
            logger.error(
                f"Validation error creating relation between '{request.source_entity}' and '{request.target_entity}': {str(ve)}"
            )
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(
                f"Error creating relation between '{request.source_entity}' and '{request.target_entity}': {str(e)}"
            )
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500, detail=f"Error creating relation: {str(e)}"
            )
        finally:
            reset_current_workspace(token)

    @router.post("/graph/entities/merge", dependencies=[Depends(combined_auth)])
    async def merge_entities(request: EntityMergeRequest):
        """Merge multiple entities into a single entity, preserving all relationships"""
        token = set_current_workspace(request.workspace)
        try:
            result = await rag.amerge_entities(
                source_entities=request.entities_to_change,
                target_entity=request.entity_to_change_into,
            )
            return {
                "status": "success",
                "message": f"Successfully merged {len(request.entities_to_change)} entities into '{request.entity_to_change_into}'",
                "data": result,
            }
        except ValueError as ve:
            logger.error(
                f"Validation error merging entities {request.entities_to_change} into '{request.entity_to_change_into}': {str(ve)}"
            )
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(
                f"Error merging entities {request.entities_to_change} into '{request.entity_to_change_into}': {str(e)}"
            )
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500, detail=f"Error merging entities: {str(e)}"
            )
        finally:
            reset_current_workspace(token)

    return router