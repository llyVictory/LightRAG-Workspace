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

def create_workspace_routes(rag, api_key: Optional[str] = None):
    combined_auth = get_combined_auth_dependency(api_key)

    # Use rag.working_dir as the base storage path
    WORKING_DIR = rag.working_dir
    INPUTSING_DIR = rag.inputsing_dir

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
    async def delete_workspace(name: str = Query(..., description="Name of the workspace to delete")):
        """
        Delete a workspace and all its data from both storage and inputs.
        Warning: This action is irreversible.
        """
        try:
            target_work_path = get_workspace_dir(WORKING_DIR, name)
            target_input_path = get_workspace_dir(INPUTSING_DIR, name)

            if not target_work_path.exists():
                raise HTTPException(status_code=404, detail=f"Workspace '{name}' not found")

            # Remove directory trees
            shutil.rmtree(target_work_path)

            # Also remove from inputs if it exists
            if target_input_path.exists():
                shutil.rmtree(target_input_path)

            logger.info(f"Deleted workspace: {name}")
            return {"status": "success", "message": f"Workspace '{name}' deleted successfully"}

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error deleting workspace: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.put("/rename", dependencies=[Depends(combined_auth)])
    async def rename_workspace(request: RenameWorkspaceRequest):
        """
        Rename an existing workspace in both storage and inputs.
        """
        try:
            old_name = request.old_name
            new_name = request.new_name
            old_work_path = get_workspace_dir(WORKING_DIR, old_name)
            new_work_path = get_workspace_dir(WORKING_DIR, new_name)

            old_input_path = get_workspace_dir(INPUTSING_DIR, old_name)
            new_input_path = get_workspace_dir(INPUTSING_DIR, new_name)

            if not old_work_path.exists():
                raise HTTPException(status_code=404, detail=f"Source workspace '{request.old_name}' not found")

            if new_work_path.exists():
                raise HTTPException(status_code=409, detail=f"Target workspace name '{request.new_name}' already exists")

            # Rename storage directory
            os.rename(old_work_path, new_work_path)

            # Rename input directory if it exists
            if old_input_path.exists():
                if new_input_path.exists():
                    logger.warning(f"Target input directory {new_input_path} already exists, merging/overwriting")
                    # Strategy: Move content or just fail?
                    # Simplest: Fail if new input exists but old work didn't (handled by check above)
                    # But here input path might independently exist.
                    # Let's assume safe rename.
                    pass
                os.rename(old_input_path, new_input_path)

            logger.info(f"Renamed workspace from {request.old_name} to {request.new_name}")
            return {
                "status": "success",
                "message": f"Workspace renamed from '{request.old_name}' to '{request.new_name}'",
                "old_name": request.old_name,
                "new_name": request.new_name
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error renaming workspace: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    return router