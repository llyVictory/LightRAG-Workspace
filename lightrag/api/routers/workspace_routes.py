"""
This module contains all workspace-related routes for the LightRAG API.
"""

import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import traceback

from fastapi import APIRouter, Depends, HTTPException, Body, Query
from pydantic import BaseModel, Field, field_validator

from lightrag.utils import logger
from ..utils_api import get_combined_auth_dependency
from .document_routes import  DocumentManager,ClearDocumentsResponse, ClearCacheRequest, ClearCacheResponse
from ...exceptions import UnprocessableEntityError

router = APIRouter(
    prefix="/workspace",
    tags=["workspace"],
)


# --- Request/Response Models ---

class WorkspaceInfo(BaseModel):
    name: str = Field(description="Name of the workspace")
    created_at: Optional[str] = Field(description="Creation time (ISO format)")
    is_default: bool = Field(description="Whether this is the default workspace")

class CreateWorkspaceRequest(BaseModel):
    name: str = Field(..., min_length=1, description="Name of the new workspace")

    @field_validator('name')
    def validate_name(cls, v):
        if not v.strip() or any(c in v for c in '/\\:*?"<>|'):
            raise ValueError("Invalid workspace name. Contains special characters.")
        return v.strip()

class RenameWorkspaceRequest(BaseModel):
    old_name: str = Field(..., min_length=1, description="Current name of the workspace")
    new_name: str = Field(..., min_length=1, description="New name for the workspace")

    @field_validator('new_name')
    def validate_new_name(cls, v):
        if not v.strip() or any(c in v for c in '/\\:*?"<>|'):
            raise ValueError("Invalid new name. Contains special characters.")
        return v.strip()

class DeleteWorkspaceRequest(BaseModel):
    name: str = Field(..., min_length=1, description="Name of the workspace to delete")

# --- Helper Functions ---

def get_workspace_dir(working_dir: str, workspace_name: str) -> Path:
    """Securely resolve workspace directory path"""
    base_path = Path(working_dir).resolve()
    # Remove potential path traversal characters
    safe_name = os.path.basename(workspace_name)
    if safe_name != workspace_name:
        raise HTTPException(status_code=400, detail="Invalid workspace name format")

    target_path = (base_path / safe_name).resolve()

    # Security check: ensure target path is inside working_dir
    if not str(target_path).startswith(str(base_path)):
        raise HTTPException(status_code=403, detail="Access denied: Invalid path")

    return target_path

def format_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).isoformat()

# --- Routes Definition ---

def create_workspace_routes(rag, doc_manager: DocumentManager, api_key: Optional[str] = None):
    combined_auth = get_combined_auth_dependency(api_key)

    # Use rag.working_dir as the base storage path
    WORKING_DIR = rag.working_dir
    INPUTSING_DIR = rag.inputsing_dir



    # --- Document Functions ---
    import asyncio
    from lightrag.context_utils import set_current_workspace, reset_current_workspace


    async def clear_documents(
            workspace: str = Query("default") # [修改] 增加 workspace 参数
    ):
        """
        Clear all documents from the RAG system.

        This endpoint deletes all documents, entities, relationships, and files from the system.
        It uses the storage drop methods to properly clean up all data and removes all files
        from the input directory.

        Returns:
            ClearDocumentsResponse: A response object containing the status and message.
                - status="success":           All documents and files were successfully cleared.
                - status="partial_success":   Document clear job exit with some errors.
                - status="busy":              Operation could not be completed because the pipeline is busy.
                - status="fail":              All storage drop operations failed, with message
                - message: Detailed information about the operation results, including counts
                  of deleted files and any errors encountered.

        Raises:
            HTTPException: Raised when a serious error occurs during the clearing process,
                          with status code 500 and error details in the detail field.
        """
        from lightrag.kg.shared_storage import (
            get_namespace_data,
            get_namespace_lock,
        )
        token = set_current_workspace(workspace)

        try:
            # Get pipeline status and lock
            pipeline_status = await get_namespace_data(
                "pipeline_status", workspace=rag.workspace
            )
            pipeline_status_lock = get_namespace_lock(
                "pipeline_status", workspace=rag.workspace
            )

            # Check and set status with lock
            async with pipeline_status_lock:
                if pipeline_status.get("busy", False):
                    return ClearDocumentsResponse(
                        status="busy",
                        message="Cannot clear documents while pipeline is busy",
                    )
                # Set busy to true
                pipeline_status.update(
                    {
                        "busy": True,
                        "job_name": "Clearing Documents",
                        "job_start": datetime.now().isoformat(),
                        "docs": 0,
                        "batchs": 0,
                        "cur_batch": 0,
                        "request_pending": False,  # Clear any previous request
                        "latest_message": "Starting document clearing process",
                    }
                )
                # Cleaning history_messages without breaking it as a shared list object
                del pipeline_status["history_messages"][:]
                pipeline_status["history_messages"].append(
                    "Starting document clearing process"
                )

            try:
                # Use drop method to clear all data
                drop_tasks = []
                storages = [
                    rag.text_chunks,
                    rag.full_docs,
                    rag.full_entities,
                    rag.full_relations,
                    rag.entity_chunks,
                    rag.relation_chunks,
                    rag.entities_vdb,
                    rag.relationships_vdb,
                    rag.chunks_vdb,
                    rag.chunk_entity_relation_graph,
                    rag.doc_status,
                ]

                # Log storage drop start
                if "history_messages" in pipeline_status:
                    pipeline_status["history_messages"].append(
                        "Starting to drop storage components"
                    )

                for storage in storages:
                    if storage is not None:
                        drop_tasks.append(storage.drop())

                # Wait for all drop tasks to complete
                drop_results = await asyncio.gather(*drop_tasks, return_exceptions=True)

                # Check for errors and log results
                errors = []
                storage_success_count = 0
                storage_error_count = 0

                for i, result in enumerate(drop_results):
                    storage_name = storages[i].__class__.__name__
                    if isinstance(result, Exception):
                        error_msg = f"Error dropping {storage_name}: {str(result)}"
                        errors.append(error_msg)
                        logger.error(error_msg)
                        storage_error_count += 1
                    else:
                        namespace = storages[i].namespace
                        workspace = storages[i].workspace
                        logger.info(
                            f"Successfully dropped {storage_name}: {workspace}/{namespace}"
                        )
                        storage_success_count += 1

                # Log storage drop results
                if "history_messages" in pipeline_status:
                    if storage_error_count > 0:
                        pipeline_status["history_messages"].append(
                            f"Dropped {storage_success_count} storage components with {storage_error_count} errors"
                        )
                    else:
                        pipeline_status["history_messages"].append(
                            f"Successfully dropped all {storage_success_count} storage components"
                        )

                # If all storage operations failed, return error status and don't proceed with file deletion
                if storage_success_count == 0 and storage_error_count > 0:
                    error_message = "All storage drop operations failed. Aborting document clearing process."
                    logger.error(error_message)
                    if "history_messages" in pipeline_status:
                        pipeline_status["history_messages"].append(error_message)
                    return ClearDocumentsResponse(status="fail", message=error_message)

                # Log file deletion start
                if "history_messages" in pipeline_status:
                    pipeline_status["history_messages"].append(
                        "Starting to delete files in input directory"
                    )

                # Delete only files in the current directory, preserve files in subdirectories
                deleted_files_count = 0
                file_errors_count = 0

                for file_path in doc_manager.input_dir.glob("*"):
                    if file_path.is_file():
                        try:
                            file_path.unlink()
                            deleted_files_count += 1
                        except Exception as e:
                            logger.error(f"Error deleting file {file_path}: {str(e)}")
                            file_errors_count += 1

                # Log file deletion results
                if "history_messages" in pipeline_status:
                    if file_errors_count > 0:
                        pipeline_status["history_messages"].append(
                            f"Deleted {deleted_files_count} files with {file_errors_count} errors"
                        )
                        errors.append(f"Failed to delete {file_errors_count} files")
                    else:
                        pipeline_status["history_messages"].append(
                            f"Successfully deleted {deleted_files_count} files"
                        )

                # Prepare final result message
                final_message = ""
                if errors:
                    final_message = f"Cleared documents with some errors. Deleted {deleted_files_count} files."
                    status = "partial_success"
                else:
                    final_message = f"All documents cleared successfully. Deleted {deleted_files_count} files."
                    status = "success"

                # Log final result
                if "history_messages" in pipeline_status:
                    pipeline_status["history_messages"].append(final_message)

                # Return response based on results
                return ClearDocumentsResponse(status=status, message=final_message)
            except Exception as e:
                error_msg = f"Error clearing documents: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                if "history_messages" in pipeline_status:
                    pipeline_status["history_messages"].append(error_msg)
                raise HTTPException(status_code=500, detail=str(e))
            finally:
                # Reset busy status after completion
                async with pipeline_status_lock:
                    pipeline_status["busy"] = False
                    completion_msg = "Document clearing process completed"
                    pipeline_status["latest_message"] = completion_msg
                    if "history_messages" in pipeline_status:
                        pipeline_status["history_messages"].append(completion_msg)
        finally:
            reset_current_workspace(token)

    async def clear_cache(request: ClearCacheRequest):
        """
        Clear all cache data from the LLM response cache storage.

        This endpoint clears all cached LLM responses regardless of mode.
        The request body is accepted for API compatibility but is ignored.

        Args:
            request (ClearCacheRequest): The request body (ignored for compatibility).

        Returns:
            ClearCacheResponse: A response object containing the status and message.

        Raises:
            HTTPException: If an error occurs during cache clearing (500).
        """
        token = set_current_workspace(request.workspace)
        try:
            # Call the aclear_cache method (no modes parameter)
            await rag.aclear_cache()

            # Prepare success message
            message = "Successfully cleared all cache"

            return ClearCacheResponse(status="success", message=message)
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            reset_current_workspace(token)

    # --- Workspace Routes ---
    @router.get("/list", response_model=List[WorkspaceInfo], dependencies=[Depends(combined_auth)])
    async def list_workspaces():
        """
        List all available workspaces (directories in rag_storage).
        """
        try:
            workspaces = []
            base_path = Path(WORKING_DIR)

            if not base_path.exists():
                return []

            # List directories
            for entry in os.scandir(base_path):
                if entry.is_dir() and not entry.name.startswith('.'):
                    stat = entry.stat()
                    workspaces.append(WorkspaceInfo(
                        name=entry.name,
                        created_at=format_timestamp(stat.st_ctime),
                        is_default=(entry.name == "default")
                    ))

            # Sort by name
            return sorted(workspaces, key=lambda x: x.name)
        except Exception as e:
            logger.error(f"Error listing workspaces: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/create", dependencies=[Depends(combined_auth)])
    async def create_workspace(request: CreateWorkspaceRequest):
        """
        Create a new workspace directory in both rag_storage and inputs.
        """
        try:
            target_work_path = get_workspace_dir(WORKING_DIR, request.name)
            target_input_path = get_workspace_dir(INPUTSING_DIR, request.name)

            # Check if it exists in rag_storage (source of truth)
            if target_work_path.exists():
                raise HTTPException(status_code=409, detail=f"Workspace '{request.name}' already exists")

            # Create directories
            target_work_path.mkdir(parents=True, exist_ok=True)
            target_input_path.mkdir(parents=True, exist_ok=True)

            logger.info(f"Created new workspace: {request.name}")
            return {"status": "success", "message": f"Workspace '{request.name}' created successfully"}

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error creating workspace: {str(e)}")
            # Cleanup on failure if partial creation happened?
            # Simplified: just report error.
            raise HTTPException(status_code=500, detail=str(e))

    @router.delete("/delete", dependencies=[Depends(combined_auth)])
    async def delete_workspace(workspace: str = Query(..., description="Name of the workspace to delete")):
        """
        Delete a workspace and all its data from both storage and inputs.
        Warning: This action is irreversible.
        """
        try:
            # 如果workspace 为 default，拒绝删除
            if workspace == "default":
                raise UnprocessableEntityError(
                    "can not delete default workspace! ",
                )
            # 清除文档
            await clear_documents(workspace=workspace)
            # 清除缓存
            await clear_cache(ClearCacheRequest(workspace=workspace))

            # 清除文件夹
            target_work_path = get_workspace_dir(WORKING_DIR, workspace)
            target_input_path = get_workspace_dir(INPUTSING_DIR, workspace)

            if not target_work_path.exists():
                raise HTTPException(status_code=404, detail=f"Workspace '{workspace}' not found")

            # Remove directory trees
            shutil.rmtree(target_work_path)

            # Also remove from inputs if it exists
            if target_input_path.exists():
                shutil.rmtree(target_input_path)

            logger.info(f"Deleted workspace: {workspace}")
            return {"status": "success", "message": f"Workspace '{workspace}' deleted successfully"}

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error deleting workspace: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    # @router.put("/rename", dependencies=[Depends(combined_auth)])
    # async def rename_workspace(request: RenameWorkspaceRequest):
    #     """
    #     Rename an existing workspace in both storage and inputs.
    #     """
    #     try:
    #         old_name = request.old_name
    #         new_name = request.new_name
    #         old_work_path = get_workspace_dir(WORKING_DIR, old_name)
    #         new_work_path = get_workspace_dir(WORKING_DIR, new_name)
    #
    #         old_input_path = get_workspace_dir(INPUTSING_DIR, old_name)
    #         new_input_path = get_workspace_dir(INPUTSING_DIR, new_name)
    #
    #         if not old_work_path.exists():
    #             raise HTTPException(status_code=404, detail=f"Source workspace '{request.old_name}' not found")
    #
    #         if new_work_path.exists():
    #             raise HTTPException(status_code=409, detail=f"Target workspace name '{request.new_name}' already exists")
    #
    #         # Rename storage directory
    #         os.rename(old_work_path, new_work_path)
    #
    #         # Rename input directory if it exists
    #         if old_input_path.exists():
    #             if new_input_path.exists():
    #                 logger.warning(f"Target input directory {new_input_path} already exists, merging/overwriting")
    #                 # Strategy: Move content or just fail?
    #                 # Simplest: Fail if new input exists but old work didn't (handled by check above)
    #                 # But here input path might independently exist.
    #                 # Let's assume safe rename.
    #                 pass
    #             os.rename(old_input_path, new_input_path)
    #
    #         logger.info(f"Renamed workspace from {request.old_name} to {request.new_name}")
    #         return {
    #             "status": "success",
    #             "message": f"Workspace renamed from '{request.old_name}' to '{request.new_name}'",
    #             "old_name": request.old_name,
    #             "new_name": request.new_name
    #         }
    #
    #     except HTTPException:
    #         raise
    #     except Exception as e:
    #         logger.error(f"Error renaming workspace: {str(e)}")
    #         raise HTTPException(status_code=500, detail=str(e))

    return router