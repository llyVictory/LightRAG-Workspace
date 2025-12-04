"""
This module contains all query-related routes for the LightRAG API.
"""

import json
from typing import Any, Dict, List, Literal, Optional
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from lightrag.base import QueryParam
from lightrag.api.utils_api import get_combined_auth_dependency
from lightrag.utils import logger
from pydantic import BaseModel, Field, field_validator
from lightrag.context_utils import set_current_workspace, reset_current_workspace

router = APIRouter(tags=["query"])


class QueryRequest(BaseModel):
    query: str = Field(
        min_length=3,
        description="The query text",
    )

    mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"] = Field(
        default="mix",
        description="Query mode",
    )

    only_need_context: Optional[bool] = Field(
        default=None,
        description="If True, only returns the retrieved context without generating a response.",
    )

    only_need_prompt: Optional[bool] = Field(
        default=None,
        description="If True, only returns the generated prompt without producing a response.",
    )

    response_type: Optional[str] = Field(
        min_length=1,
        default=None,
        description="Defines the response format. Examples: 'Multiple Paragraphs', 'Single Paragraph', 'Bullet Points'.",
    )

    top_k: Optional[int] = Field(
        ge=1,
        default=None,
        description="Number of top items to retrieve. Represents entities in 'local' mode and relationships in 'global' mode.",
    )

    chunk_top_k: Optional[int] = Field(
        ge=1,
        default=None,
        description="Number of text chunks to retrieve initially from vector search and keep after reranking.",
    )

    max_entity_tokens: Optional[int] = Field(
        default=None,
        description="Maximum number of tokens allocated for entity context in unified token control system.",
        ge=1,
    )

    max_relation_tokens: Optional[int] = Field(
        default=None,
        description="Maximum number of tokens allocated for relationship context in unified token control system.",
        ge=1,
    )

    max_total_tokens: Optional[int] = Field(
        default=None,
        description="Maximum total tokens budget for the entire query context (entities + relations + chunks + system prompt).",
        ge=1,
    )

    hl_keywords: list[str] = Field(
        default_factory=list,
        description="List of high-level keywords to prioritize in retrieval. Leave empty to use the LLM to generate the keywords.",
    )

    ll_keywords: list[str] = Field(
        default_factory=list,
        description="List of low-level keywords to refine retrieval focus. Leave empty to use the LLM to generate the keywords.",
    )

    conversation_history: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Stores past conversation history to maintain context. Format: [{'role': 'user/assistant', 'content': 'message'}].",
    )

    user_prompt: Optional[str] = Field(
        default=None,
        description="User-provided prompt for the query. If provided, this will be used instead of the default value from prompt template.",
    )

    enable_rerank: Optional[bool] = Field(
        default=None,
        description="Enable reranking for retrieved text chunks. If True but no rerank model is configured, a warning will be issued. Default is True.",
    )

    include_references: Optional[bool] = Field(
        default=True,
        description="If True, includes reference list in responses. Affects /query and /query/stream endpoints. /query/data always includes references.",
    )

    include_chunk_content: Optional[bool] = Field(
        default=False,
        description="If True, includes actual chunk text content in references. Only applies when include_references=True. Useful for evaluation and debugging.",
    )

    stream: Optional[bool] = Field(
        default=True,
        description="If True, enables streaming output for real-time responses. Only affects /query/stream endpoint.",
    )

    workspace: str = Field(
        default="default",
        description="Workspace for data isolation. Defaults to 'default'."
    )

    @field_validator("query", mode="after")
    @classmethod
    def query_strip_after(cls, query: str) -> str:
        return query.strip()

    @field_validator("conversation_history", mode="after")
    @classmethod
    def conversation_history_role_check(
            cls, conversation_history: List[Dict[str, Any]] | None
    ) -> List[Dict[str, Any]] | None:
        if conversation_history is None:
            return None
        for msg in conversation_history:
            if "role" not in msg:
                raise ValueError("Each message must have a 'role' key.")
            if not isinstance(msg["role"], str) or not msg["role"].strip():
                raise ValueError("Each message 'role' must be a non-empty string.")
        return conversation_history

    def to_query_params(self, is_stream: bool) -> "QueryParam":
        """Converts a QueryRequest instance into a QueryParam instance."""
        # Use Pydantic's `.model_dump(exclude_none=True)` to remove None values automatically
        # Exclude API-level parameters that don't belong in QueryParam
        request_data = self.model_dump(
            exclude_none=True, exclude={"query", "include_chunk_content", "workspace"}
        )

        # Ensure `mode` and `stream` are set explicitly
        param = QueryParam(**request_data)
        param.stream = is_stream
        return param


class ReferenceItem(BaseModel):
    """A single reference item in query responses."""

    reference_id: str = Field(description="Unique reference identifier")
    file_path: str = Field(description="Path to the source file")
    content: Optional[List[str]] = Field(
        default=None,
        description="List of chunk contents from this file (only present when include_chunk_content=True)",
    )


class QueryResponse(BaseModel):
    response: str = Field(
        description="The generated response",
    )
    references: Optional[List[ReferenceItem]] = Field(
        default=None,
        description="Reference list (Disabled when include_references=False, /query/data always includes references.)",
    )


class QueryDataResponse(BaseModel):
    status: str = Field(description="Query execution status")
    message: str = Field(description="Status message")
    data: Dict[str, Any] = Field(
        description="Query result data containing entities, relationships, chunks, and references"
    )
    metadata: Dict[str, Any] = Field(
        description="Query metadata including mode, keywords, and processing information"
    )


class StreamChunkResponse(BaseModel):
    """Response model for streaming chunks in NDJSON format"""

    references: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Reference list (only in first chunk when include_references=True)",
    )
    response: Optional[str] = Field(
        default=None, description="Response content chunk or complete response"
    )
    error: Optional[str] = Field(
        default=None, description="Error message if processing fails"
    )


def create_query_routes(rag, api_key: Optional[str] = None, top_k: int = 60):
    combined_auth = get_combined_auth_dependency(api_key)

    @router.post(
        "/query",
        response_model=QueryResponse,
        dependencies=[Depends(combined_auth)],
        # ... (Response examples omitted for brevity)
    )
    async def query_text(request: QueryRequest):
        token = set_current_workspace(request.workspace)
        try:
            param = request.to_query_params(False)
            param.stream = False

            result = await rag.aquery_llm(request.query, param=param)

            llm_response = result.get("llm_response", {})
            data = result.get("data", {})
            references = data.get("references", [])

            response_content = llm_response.get("content", "")
            if not response_content:
                response_content = "No relevant context found for the query."

            if request.include_references and request.include_chunk_content:
                chunks = data.get("chunks", [])
                ref_id_to_content = {}
                for chunk in chunks:
                    ref_id = chunk.get("reference_id", "")
                    content = chunk.get("content", "")
                    if ref_id and content:
                        ref_id_to_content.setdefault(ref_id, []).append(content)

                enriched_references = []
                for ref in references:
                    ref_copy = ref.copy()
                    ref_id = ref.get("reference_id", "")
                    if ref_id in ref_id_to_content:
                        ref_copy["content"] = ref_id_to_content[ref_id]
                    enriched_references.append(ref_copy)
                references = enriched_references

            if request.include_references:
                return QueryResponse(response=response_content, references=references)
            else:
                return QueryResponse(response=response_content, references=None)
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            reset_current_workspace(token)

    @router.post(
        "/query/stream",
        dependencies=[Depends(combined_auth)],
        # ... (Response examples omitted for brevity)
    )
    async def query_text_stream(request: QueryRequest):
        token = set_current_workspace(request.workspace)
        # We cannot use finally here because we are returning a generator
        # that will execute later. We need to handle context within the generator
        # or ensure the context is propagated.

        # However, rag.aquery_llm is awaited *before* StreamingResponse starts.
        # So rag.aquery_llm will run with the correct context.
        # The generator itself just yields results from memory/iterator.

        try:
            stream_mode = request.stream if request.stream is not None else True
            param = request.to_query_params(stream_mode)

            # This call executes storage operations and LLM call initiation
            result = await rag.aquery_llm(request.query, param=param)

            # Now we can reset context because storage access is mostly done,
            # except if response_iterator accesses storage lazily.
            # LightRAG's response_iterator is usually an LLM stream wrapper, not storage.
            # But to be safe, we should probably wrap the generator.

            async def stream_generator():
                # Re-set context inside generator in case of lazy evaluation
                # although usually not needed if aquery_llm finished logic.
                gen_token = set_current_workspace(request.workspace)
                try:
                    references = result.get("data", {}).get("references", [])
                    llm_response = result.get("llm_response", {})

                    if request.include_references and request.include_chunk_content:
                        data = result.get("data", {})
                        chunks = data.get("chunks", [])
                        ref_id_to_content = {}
                        for chunk in chunks:
                            ref_id = chunk.get("reference_id", "")
                            content = chunk.get("content", "")
                            if ref_id and content:
                                ref_id_to_content.setdefault(ref_id, []).append(content)

                        enriched_references = []
                        for ref in references:
                            ref_copy = ref.copy()
                            ref_id = ref.get("reference_id", "")
                            if ref_id in ref_id_to_content:
                                ref_copy["content"] = ref_id_to_content[ref_id]
                            enriched_references.append(ref_copy)
                        references = enriched_references

                    if llm_response.get("is_streaming"):
                        if request.include_references:
                            yield f"{json.dumps({'references': references})}\n"

                        response_stream = llm_response.get("response_iterator")
                        if response_stream:
                            try:
                                async for chunk in response_stream:
                                    if chunk:
                                        yield f"{json.dumps({'response': chunk})}\n"
                            except Exception as e:
                                logger.error(f"Streaming error: {str(e)}")
                                yield f"{json.dumps({'error': str(e)})}\n"
                    else:
                        response_content = llm_response.get("content", "")
                        if not response_content:
                            response_content = "No relevant context found for the query."

                        complete_response = {"response": response_content}
                        if request.include_references:
                            complete_response["references"] = references

                        yield f"{json.dumps(complete_response)}\n"
                finally:
                    reset_current_workspace(gen_token)

            return StreamingResponse(
                stream_generator(),
                media_type="application/x-ndjson",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Content-Type": "application/x-ndjson",
                    "X-Accel-Buffering": "no",
                },
            )
        except Exception as e:
            logger.error(f"Error processing streaming query: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            # Safe to reset here because aquery_llm has returned
            reset_current_workspace(token)

    @router.post(
        "/query/data",
        response_model=QueryDataResponse,
        dependencies=[Depends(combined_auth)],
        # ... (Response examples omitted for brevity)
    )
    async def query_data(request: QueryRequest):
        token = set_current_workspace(request.workspace)
        try:
            param = request.to_query_params(False)
            response = await rag.aquery_data(request.query, param=param)

            if isinstance(response, dict):
                return QueryDataResponse(**response)
            else:
                return QueryDataResponse(
                    status="failure",
                    message="Invalid response type",
                    data={},
                    metadata={} # Add missing field to match model
                )
        except Exception as e:
            logger.error(f"Error processing data query: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            reset_current_workspace(token)

    return router