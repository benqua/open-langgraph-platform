"""Assistant Endpoints for Agent Protocol

This API follows a layered architecture pattern, separating business logic into a
service layer (assistant_service.py). This pattern was first applied to the
assistant API and will be used to refactor all other APIs (runs, threads, etc.)
in the future.

Architecture:
- API Layer (this file): Thin FastAPI route handlers, request/response processing.
- Service Layer (assistant_service.py): Business logic, validation, orchestration.

Key Components:
- create_assistant: Create an assistant (with duplicate check).
- list_assistants: List a user's assistants.
- search_assistants: Search with filtering and pagination.
- get_assistant: Get a specific assistant.
- update_assistant: Update an assistant (creates version history).
- delete_assistant: Delete an assistant.
- set_assistant_latest: Roll back to a specific version.
- list_assistant_versions: List version history.
- get_assistant_schemas: Extract graph schemas (5 types).
- get_assistant_graph: Get graph structure (for visualization).
- get_assistant_subgraphs: Get subgraphs.

Usage Example:
    from fastapi import FastAPI
    from .api.assistants import router

    app = FastAPI()
    app.include_router(router)

    # POST /assistants - Create an assistant
    # GET /assistants - List assistants
    # GET /assistants/{assistant_id} - Get a specific assistant
    # PATCH /assistants/{assistant_id} - Update an assistant
    # DELETE /assistants/{assistant_id} - Delete an assistant
"""

from typing import Any

from fastapi import APIRouter, Body, Depends

from ..core.auth_deps import get_current_user
from ..models import (
    Assistant,
    AssistantCreate,
    AssistantList,
    AssistantSearchRequest,
    AssistantUpdate,
    User,
)
from ..services.assistant_service import AssistantService, get_assistant_service

router = APIRouter()


@router.post("/assistants", response_model=Assistant)
async def create_assistant(
    request: AssistantCreate,
    user: User = Depends(get_current_user),
    service: AssistantService = Depends(get_assistant_service),
) -> Assistant:
    """Create a new assistant.

    Creates an assistant based on a graph ID defined in open_langgraph.json.
    Performs a duplicate check and acts according to the if_exists policy.

    Workflow:
    1. Validate request data (graph_id, config, context).
    2. Check if the graph exists and is loadable.
    3. Check for duplicate assistants (user_id + graph_id + config combination).
    4. Create the assistant record.
    5. Create the version 1 history record.

    Args:
        request (AssistantCreate): The assistant creation request data.
            - graph_id: The graph ID defined in open_langgraph.json (required).
            - name: The assistant's name (optional, defaults to "Assistant for {graph_id}").
            - config: LangGraph configuration (optional, defaults to {}).
            - context: Runtime context (optional, replaces configurable in LangGraph 0.6.0+).
            - metadata: User-defined metadata (optional).
            - if_exists: Duplicate policy ("error" or "do_nothing").
        user (User): The authenticated user (dependency injection).
        service (AssistantService): The assistant service (dependency injection).

    Returns:
        Assistant: The created assistant (including assistant_id, version=1).

    Raises:
        HTTPException(400): If the graph does not exist or fails to load.
        HTTPException(400): If both config and context are specified.
        HTTPException(409): If an identical assistant already exists (if_exists="error").
    """
    return await service.create_assistant(request, user.identity)


@router.get("/assistants", response_model=AssistantList)
async def list_assistants(
    user: User = Depends(get_current_user),
    service: AssistantService = Depends(get_assistant_service),
) -> AssistantList:
    """List all of a user's assistants.

    Returns all assistants owned by the authenticated user.
    Automatically filtered by user_id for multi-tenancy isolation.

    Args:
        user (User): The authenticated user (dependency injection).
        service (AssistantService): The assistant service (dependency injection).

    Returns:
        AssistantList: A list of assistants and the total count.
            - assistants: An array of assistants.
            - total: The total number of assistants.
    """
    assistants = await service.list_assistants(user.identity)
    return AssistantList(assistants=assistants, total=len(assistants))


@router.post("/assistants/search", response_model=list[Assistant])
async def search_assistants(
    request: AssistantSearchRequest,
    user: User = Depends(get_current_user),
    service: AssistantService = Depends(get_assistant_service),
) -> list[Assistant]:
    """Search for assistants using filters.

    Filters a user's assistants by name, description, graph_id, metadata, etc.,
    and applies pagination to the results.

    Filter Conditions:
    - name: Partial match search on the name (case-insensitive).
    - description: Partial match search on the description (case-insensitive).
    - graph_id: Exact match on the graph ID.
    - metadata: Filters metadata using the JSONB containment operator (@>).

    Args:
        request (AssistantSearchRequest): Search filters and pagination parameters.
            - name: Name filter (partial match).
            - description: Description filter (partial match).
            - graph_id: Graph ID filter (exact match).
            - metadata: Metadata filter (JSONB @> operator).
            - offset: Starting position (default: 0).
            - limit: Maximum number of items (default: 20).
        user (User): The authenticated user (dependency injection).
        service (AssistantService): The assistant service (dependency injection).

    Returns:
        list[Assistant]: A filtered and paginated list of assistants.
    """
    return await service.search_assistants(request, user.identity)


@router.post("/assistants/count", response_model=int)
async def count_assistants(
    request: AssistantSearchRequest,
    user: User = Depends(get_current_user),
    service: AssistantService = Depends(get_assistant_service),
) -> int:
    """Get the total count of assistants matching the filter criteria.

    Uses the same filters as search_assistants() to return the total count.
    Used to calculate the total number of pages in a pagination UI.

    Args:
        request (AssistantSearchRequest): Search filters (excluding offset, limit).
            - name: Name filter (partial match).
            - description: Description filter (partial match).
            - graph_id: Graph ID filter (exact match).
            - metadata: Metadata filter (JSONB @> operator).
        user (User): The authenticated user (dependency injection).
        service (AssistantService): The assistant service (dependency injection).

    Returns:
        int: The total number of assistants matching the filter criteria.
    """
    return await service.count_assistants(request, user.identity)


@router.get("/assistants/{assistant_id}", response_model=Assistant)
async def get_assistant(
    assistant_id: str,
    user: User = Depends(get_current_user),
    service: AssistantService = Depends(get_assistant_service),
) -> Assistant:
    """Get a specific assistant by ID.

    Retrieves an assistant owned by the user or provided by the system.
    System assistants are the default assistants for graphs defined in open_langgraph.json.

    Args:
        assistant_id (str): The unique identifier for the assistant.
        user (User): The authenticated user (dependency injection).
        service (AssistantService): The assistant service (dependency injection).

    Returns:
        Assistant: The retrieved assistant.

    Raises:
        HTTPException(404): If the assistant is not found.
    """
    return await service.get_assistant(assistant_id, user.identity)


@router.patch("/assistants/{assistant_id}", response_model=Assistant)
async def update_assistant(
    assistant_id: str,
    request: AssistantUpdate,
    user: User = Depends(get_current_user),
    service: AssistantService = Depends(get_assistant_service),
) -> Assistant:
    """Update an assistant and create a version history.

    Updates an assistant and archives the previous version in the assistant_versions table.
    The version number is automatically incremented, and users can later roll back
    to a specific version.

    Workflow:
    1. Synchronize config and context.
    2. Query the existing assistant.
    3. Query the max version number and increment it.
    4. Create a new version history record.
    5. Update the main assistant record.

    Args:
        assistant_id (str): The unique identifier for the assistant.
        request (AssistantUpdate): The fields to update.
            - name: Assistant name (optional).
            - description: Description (optional).
            - graph_id: Change the graph ID (optional).
            - config: LangGraph configuration (optional).
            - context: Runtime context (optional).
            - metadata: Metadata (optional).
        user (User): The authenticated user (dependency injection).
        service (AssistantService): The assistant service (dependency injection).

    Returns:
        Assistant: The updated assistant (with the new version number).

    Raises:
        HTTPException(400): If both config and context are specified.
        HTTPException(404): If the assistant is not found.
    """
    return await service.update_assistant(assistant_id, request, user.identity)


@router.delete("/assistants/{assistant_id}")
async def delete_assistant(
    assistant_id: str,
    user: User = Depends(get_current_user),
    service: AssistantService = Depends(get_assistant_service),
) -> dict[str, str]:
    """Delete an assistant.

    Permanently deletes an assistant.
    Due to CASCADE settings, associated version history, runs, and events
    are also deleted.

    Args:
        assistant_id (str): The unique identifier for the assistant.
        user (User): The authenticated user (dependency injection).
        service (AssistantService): The assistant service (dependency injection).

    Returns:
        dict: Deletion status {"status": "deleted"}.

    Raises:
        HTTPException(404): If the assistant is not found.
    """
    return await service.delete_assistant(assistant_id, user.identity)


@router.post("/assistants/{assistant_id}/latest", response_model=Assistant)
async def set_assistant_latest(
    assistant_id: str,
    version: int = Body(..., embed=True, description="The version number to set as latest"),
    user: User = Depends(get_current_user),
    service: AssistantService = Depends(get_assistant_service),
) -> Assistant:
    """Set a specific version as the latest version (rollback).

    Sets a past version stored in the assistant_versions table as the latest
    version of the assistant. This allows users to roll back to a previous
    configuration or graph.

    Workflow:
    1. Check if the assistant exists.
    2. Check if the requested version exists.
    3. Update the main assistant record with the content of that version.

    Args:
        assistant_id (str): The unique identifier for the assistant.
        version (int): The version number to restore.
        user (User): The authenticated user (dependency injection).
        service (AssistantService): The assistant service (dependency injection).

    Returns:
        Assistant: The assistant with the restored version.

    Raises:
        HTTPException(404): If the assistant or version is not found.
    """
    return await service.set_assistant_latest(assistant_id, version, user.identity)


@router.post("/assistants/{assistant_id}/versions", response_model=list[Assistant])
async def list_assistant_versions(
    assistant_id: str,
    user: User = Depends(get_current_user),
    service: AssistantService = Depends(get_assistant_service),
) -> list[Assistant]:
    """List all version history of an assistant.

    Returns all versions stored in the assistant_versions table, sorted by most recent.
    Each version preserves the past configuration, graph, and metadata.

    Args:
        assistant_id (str): The unique identifier for the assistant.
        user (User): The authenticated user (dependency injection).
        service (AssistantService): The assistant service (dependency injection).

    Returns:
        list[Assistant]: A list of versions (sorted by most recent).

    Raises:
        HTTPException(404): If the assistant or versions are not found.
    """
    return await service.list_assistant_versions(assistant_id, user.identity)


@router.get("/assistants/{assistant_id}/schemas")
async def get_assistant_schemas(
    assistant_id: str,
    user: User = Depends(get_current_user),
    service: AssistantService = Depends(get_assistant_service),
) -> dict[str, Any]:
    """Get the graph schemas of an assistant (5 types).

    Extracts and returns all schemas of the LangGraph graph used by the assistant.
    Clients can use this information to understand the input format, output format,
    and state structure.

    Returned Schemas:
    1. input_schema: JSON schema for the graph input.
    2. output_schema: JSON schema for the graph output.
    3. state_schema: JSON schema for the graph state (channels).
    4. config_schema: JSON schema for the configurable settings.
    5. context_schema: JSON schema for the runtime context.

    Args:
        assistant_id (str): The unique identifier for the assistant.
        user (User): The authenticated user (dependency injection).
        service (AssistantService): The assistant service (dependency injection).

    Returns:
        dict: A dictionary containing the graph_id and the 5 schemas.
            - graph_id: The unique identifier for the graph.
            - input_schema: The input schema.
            - output_schema: The output schema.
            - state_schema: The state schema.
            - config_schema: The configuration schema.
            - context_schema: The context schema.

    Raises:
        HTTPException(404): If the assistant is not found.
        HTTPException(400): If schema extraction fails.
    """
    return await service.get_assistant_schemas(assistant_id, user.identity)


@router.get("/assistants/{assistant_id}/graph")
async def get_assistant_graph(
    assistant_id: str,
    xray: bool | int | None = None,
    user: User = Depends(get_current_user),
    service: AssistantService = Depends(get_assistant_service),
) -> dict[str, Any]:
    """Get the graph structure (for visualization).

    Returns the structure of the assistant's LangGraph graph in JSON format.
    This allows for visualizing the entire graph structure, including nodes,
    edges, and conditional branches.

    xray parameter:
    - False (default): Returns only the top-level graph structure.
    - True: Fully expands all subgraphs.
    - int (positive): Expands to a specific depth.

    Args:
        assistant_id (str): The unique identifier for the assistant.
        xray (bool | int | None): Subgraph expansion option (default: False).
        user (User): The authenticated user (dependency injection).
        service (AssistantService): The assistant service (dependency injection).

    Returns:
        dict: Graph structure JSON (including nodes, edges).

    Raises:
        HTTPException(404): If the assistant is not found.
        HTTPException(422): If the xray value is invalid or the graph does not support visualization.
        HTTPException(400): If graph retrieval fails.
    """
    # If xray is None, set default to False (return only top-level graph)
    xray_value = xray if xray is not None else False
    return await service.get_assistant_graph(assistant_id, xray_value, user.identity)


@router.get("/assistants/{assistant_id}/subgraphs")
async def get_assistant_subgraphs(
    assistant_id: str,
    recurse: bool = False,
    namespace: str | None = None,
    user: User = Depends(get_current_user),
    service: AssistantService = Depends(get_assistant_service),
) -> dict[str, Any]:
    """Get the subgraphs of an assistant.

    Extracts the schemas of subgraphs contained within a LangGraph graph.
    Subgraphs are nested graphs used to modularize complex workflows.

    Args:
        assistant_id (str): The unique identifier for the assistant.
        recurse (bool): Whether to recursively retrieve nested subgraphs (default: False).
        namespace (str | None): Retrieve only subgraphs in a specific namespace (or all if None).
        user (User): The authenticated user (dependency injection).
        service (AssistantService): The assistant service (dependency injection).

    Returns:
        dict: A dictionary of subgraph schemas in the form {namespace: schemas}.
            Each schema includes input_schema, output_schema, state_schema,
            config_schema, and context_schema.

    Raises:
        HTTPException(404): If the assistant is not found.
        HTTPException(422): If the graph does not support subgraphs.
        HTTPException(400): If subgraph retrieval fails.
    """
    return await service.get_assistant_subgraphs(assistant_id, namespace, recurse, user.identity)
