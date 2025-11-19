"""LangGraph Integration Service and Graph Manager

This module is responsible for loading LangGraph graphs, managing configurations,
and creating execution settings for Open LangGraph. It dynamically loads graph
definitions from open_langgraph.json and automatically creates a default
assistant for each graph.

Key Components:
- LangGraphService: Handles graph loading, caching, and configuration management.
- inject_user_context(): Injects user context into the LangGraph config.
- create_thread_config(): Creates thread-specific execution configurations.
- create_run_config(): Creates run-specific configurations, including observability callbacks.

Usage Example:
    from services.langgraph_service import get_langgraph_service

    service = get_langgraph_service()
    await service.initialize()
    graph = await service.get_graph("weather_agent")
"""

import importlib.util
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, TypedDict, cast
from uuid import uuid5

from langgraph.graph.state import CompiledStateGraph

from ..constants import ASSISTANT_NAMESPACE_UUID
from ..observability.langfuse_integration import get_tracing_callbacks

CompiledGraph = CompiledStateGraph[Any, Any, Any, Any]


class GraphDefinition(TypedDict):
    file_path: str
    export_name: str


class LangGraphService:
    """Service for loading and managing LangGraph graphs.

    This class reads the open_langgraph.json configuration file to dynamically
    load LangGraph graphs and automatically creates a default assistant for each graph.

    Key Features:
    - Graph Registry Management: Loads graph definitions from open_langgraph.json.
    - Graph Caching: Caches loaded graphs in memory to improve performance.
    - Automatic Compilation: Compiles graphs with a Postgres checkpointer.
    - Default Assistant Creation: Creates an assistant with a deterministic UUID for each graph.

    Architectural Patterns:
    - Singleton: A single instance is used throughout the application.
    - Lazy Loading: Graphs are loaded and compiled only when needed.
    - Caching: Compiled graphs are stored in memory for reuse.
    """

    def __init__(self, config_path: str = "open_langgraph.json") -> None:
        # Path to the configuration file (can be overridden by OPEN_LANGGRAPH_CONFIG env var or open_langgraph.json)
        self.config_path = Path(config_path)
        self.config: dict[str, Any] | None = None
        # Graph registry: graph_id -> {file_path, export_name}
        self._graph_registry: dict[str, GraphDefinition] = {}
        # Compiled graph cache: graph_id -> CompiledGraph
        self._graph_cache: dict[str, CompiledGraph] = {}

    async def initialize(self) -> None:
        """Load the configuration file and set up the graph registry.

        This method finds and loads the open_langgraph.json configuration file,
        then initializes the graph registry. It automatically creates a default
        assistant for each graph, allowing clients to run graphs using only the graph_id.

        Configuration File Resolution Priority:
        1) OPEN_LANGGRAPH_CONFIG environment variable (absolute or relative path)
        2) self.config_path specified in the constructor (if it exists)
        3) open_langgraph.json in the current working directory
        4) langgraph.json in the current working directory (fallback)

        Workflow:
        1. Resolve the configuration file path (based on the priority above).
        2. Load and parse the JSON file.
        3. Initialize the graph registry (_load_graph_registry).
        4. Create a default assistant for each graph (_ensure_default_assistants).

        Raises:
            ValueError: If the configuration file cannot be found.
        """
        # 1) Environment variable override has priority
        env_path = os.getenv("OPEN_LANGGRAPH_CONFIG")
        resolved_path: Path
        if env_path:
            resolved_path = Path(env_path)
        # 2) Use the path provided in the constructor if it exists
        elif self.config_path and Path(self.config_path).exists():
            resolved_path = Path(self.config_path)
        # 3) Use open_langgraph.json in the current directory if it exists
        elif Path("open_langgraph.json").exists():
            resolved_path = Path("open_langgraph.json")
        # 4) Fallback to langgraph.json
        else:
            resolved_path = Path("langgraph.json")

        if not resolved_path.exists():
            raise ValueError(
                "Configuration file not found. Expected one of: "
                "OPEN_LANGGRAPH_CONFIG path, ./open_langgraph.json, or ./langgraph.json"
            )

        # Store the selected path for later reference
        self.config_path = resolved_path

        with self.config_path.open() as f:
            loaded_config = json.load(f)

        if not isinstance(loaded_config, dict):
            raise ValueError(f"Invalid configuration format in {self.config_path}; expected JSON object")

        self.config = cast("dict[str, Any]", loaded_config)

        # Load the graph registry from the configuration file
        self._load_graph_registry()

        # Create a default assistant for each graph with a deterministic UUID
        # to allow clients to pass graph_id directly.
        await self._ensure_default_assistants()

    def _load_graph_registry(self) -> None:
        """Parse graph definitions from open_langgraph.json and register them.

        This method reads the "graphs" section of the configuration file
        and parses the file path and export name for each graph.

        Path Format:
            "./graphs/weather_agent.py:graph"
            - Before the colon (:): Path to the Python file
            - After the colon (:): Name of the variable to export from the module

        Action:
            Stores a dictionary {file_path, export_name} in _graph_registry
            for each graph_id.

        Raises:
            ValueError: If the path format is invalid (missing a colon).
        """
        if self.config is None:
            self._graph_registry = {}
            return

        graphs_config = self.config.get("graphs", {})

        for graph_id, graph_path in graphs_config.items():
            # Parse path format: "./graphs/weather_agent.py:graph"
            if ":" not in graph_path:
                raise ValueError(f"Invalid graph path format: {graph_path}")

            file_path, export_name = graph_path.split(":", 1)
            self._graph_registry[graph_id] = {
                "file_path": file_path,
                "export_name": export_name,
            }

    async def _ensure_default_assistants(self) -> None:
        """Create a default assistant with a deterministic UUID for each graph.

        This method creates one default assistant per graph, allowing clients
        to run a graph using only its graph_id.

        UUID Generation:
            Uses uuid5(ASSISTANT_NAMESPACE_UUID, graph_id) to ensure that
            the same graph_id always produces the same assistant_id.
            This maintains consistent IDs across server restarts.

        Idempotency:
            Skips existing assistants, making it safe to call multiple times.

        Generated Assistant:
        - assistant_id: uuid5(namespace, graph_id)
        - name: graph_id
        - description: "Default assistant for graph '{graph_id}'"
        - graph_id: The corresponding graph ID
        - config: {} (empty config)
        - user_id: "system"
        """
        from sqlalchemy import select

        from ..core.orm import Assistant as AssistantORM
        from ..core.orm import get_session

        # Derive assistant_id from graph_id using a fixed namespace
        NS = ASSISTANT_NAMESPACE_UUID
        session_gen = get_session()
        session = await anext(session_gen)
        try:
            for graph_id in self._graph_registry:
                # Generate deterministic UUID
                assistant_id = str(uuid5(NS, graph_id))
                existing = await session.scalar(
                    select(AssistantORM).where(AssistantORM.assistant_id == assistant_id)
                )
                if existing:
                    # Skip if already exists (ensures idempotency)
                    continue
                # Create a new default assistant
                session.add(
                    AssistantORM(
                        assistant_id=assistant_id,
                        name=graph_id,
                        description=f"Default assistant for graph '{graph_id}'",
                        graph_id=graph_id,
                        config={},
                        user_id="system",
                    )
                )
            await session.commit()
        finally:
            await session.close()

    async def get_graph(self, graph_id: str, force_reload: bool = False) -> CompiledGraph:
        """Get a compiled graph by its ID (with caching and LangGraph integration).

        This method loads the requested graph and compiles it with a Postgres
        checkpointer to ensure state persistence.

        Workflow:
        1. Check if the graph exists in the registry.
        2. Check the cache: Return the cached graph unless force_reload is True.
        3. Load the graph from the file (_load_graph_from_file).
        4. Handle graph compilation:
           a. Uncompiled StateGraph: Compile with Postgres checkpointer.
           b. Already compiled graph: Try to inject the checkpointer with copy().
           c. If injection fails: Use the original graph (with a warning).
        5. Store the compiled graph in the cache.
        6. Return the compiled graph.

        Args:
            graph_id (str): The ID of the graph to load (defined in open_langgraph.json).
            force_reload (bool): If True, ignore the cache and reload (default: False).

        Returns:
            StateGraph[Any]: The graph compiled with a Postgres checkpointer.

        Raises:
            ValueError: If the graph is not found in the registry.

        Note:
            - Postgres Checkpointer: Saves state snapshots (checkpoints).
            - Postgres Store: Long-term memory and key-value storage.
            - Caching: Improves performance for repeated loads of the same graph.
        """
        if graph_id not in self._graph_registry:
            raise ValueError(f"Graph not found: {graph_id}")

        # If a cached graph exists and not forcing a reload, return it
        if not force_reload and graph_id in self._graph_cache:
            return self._graph_cache[graph_id]

        graph_info = self._graph_registry[graph_id]

        # Load the graph from the file
        base_graph = await self._load_graph_from_file(graph_id, graph_info)

        # Compile all graphs with a Postgres checkpointer to ensure persistence
        from ..core.database import db_manager

        checkpointer_cm = await db_manager.get_checkpointer()
        store_cm = await db_manager.get_store()

        compiled_graph: CompiledGraph
        if isinstance(base_graph, CompiledStateGraph):
            try:
                compiled_graph = cast(
                    "CompiledGraph",
                    base_graph.copy(update={"checkpointer": checkpointer_cm, "store": store_cm}),
                )
            except Exception:
                print(
                    f"⚠️  Pre-compiled graph '{graph_id}' does not support checkpointer injection; running without persistence"
                )
                compiled_graph = cast("CompiledGraph", base_graph)
        elif hasattr(base_graph, "compile"):
            print(f"🔧 Compiling graph '{graph_id}' with Postgres persistence")
            compiled_graph = cast(
                "CompiledGraph",
                base_graph.compile(checkpointer=checkpointer_cm, store=store_cm),
            )
        else:
            raise TypeError(f"Graph '{graph_id}' must export a StateGraph or CompiledStateGraph")

        # Store the compiled graph in the cache
        self._graph_cache[graph_id] = compiled_graph

        return compiled_graph

    async def _load_graph_from_file(self, graph_id: str, graph_info: GraphDefinition) -> Any:
        """Dynamically load a graph module from the filesystem

        This method dynamically imports a graph module from a Python file
        and returns the graph object with the specified export name.

        Workflow:
        1. Check if the file path exists
        2. Create a module spec using importlib
        3. Dynamically load and execute the module
        4. Extract the graph object specified by export_name
        5. Return the graph object (regardless of whether it's compiled)

        Args:
            graph_id (str): Graph ID (for logging/debugging)
            graph_info (dict[str, str]): Graph information
                - file_path: Path to the Python file
                - export_name: Name of the variable to export from the module

        Returns:
            StateGraph | CompiledGraph: The loaded graph object
                (compilation status depends on the module)

        Raises:
            ValueError: If the file does not exist, module loading fails, or the export cannot be found

        Note:
            The graph may be in a compiled or uncompiled state.
            Checkpointer injection is handled by the caller (get_graph).
        """
        file_path = Path(graph_info["file_path"])
        if not file_path.exists():
            raise ValueError(f"Graph file not found: {file_path}")

        # Dynamically import the graph module
        spec = importlib.util.spec_from_file_location(f"graphs.{graph_id}", str(file_path.resolve()))
        if spec is None or spec.loader is None:
            raise ValueError(f"Failed to load graph module: {file_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Get the exported graph
        export_name = graph_info["export_name"]
        if not hasattr(module, export_name):
            raise ValueError(f"Graph export not found: {export_name} in {file_path}")

        graph = getattr(module, export_name)

        # The graph may already be compiled in the module.
        # Checkpointer/store injection is handled at runtime.
        return graph

    def list_graphs(self) -> dict[str, str]:
        """Return a list of all registered graphs

        Returns:
            dict[str, str]: A mapping of graph_id to file_path
                e.g., {"weather_agent": "./graphs/weather_agent.py"}
        """
        return {graph_id: info["file_path"] for graph_id, info in self._graph_registry.items()}

    def invalidate_cache(self, graph_id: str | None = None) -> None:
        """Invalidate the graph cache (for hot reloading)

        This method deletes a cached graph, forcing it to be reloaded from the filesystem
        on the next get_graph() call.

        Use cases:
        - Hot reloading after changing graph code during development
        - Applying a new version of a graph after deployment

        Args:
            graph_id (str | None): The ID of the graph to invalidate.
                If None, clears the entire graph cache.
        """
        if graph_id:
            self._graph_cache.pop(graph_id, None)
        else:
            self._graph_cache.clear()

    def get_config(self) -> dict[str, Any] | None:
        """Return the loaded configuration file content

        Returns:
            dict[str, Any] | None: The full content of open_langgraph.json
        """
        return self.config

    def get_dependencies(self) -> list[str]:
        """Return the dependencies section of the configuration file

        Returns:
            list: A list of dependency packages (the "dependencies" field in open_langgraph.json)
        """
        if self.config is None:
            return []
        deps = self.config.get("dependencies", [])
        if isinstance(deps, list):
            return [str(dep) for dep in deps]
        return []


# Global service instance (singleton pattern)
_langgraph_service: LangGraphService | None = None


def get_langgraph_service() -> LangGraphService:
    """Return the global LangGraph service instance (singleton).

    This function returns the same LangGraphService instance throughout the application,
    sharing the graph cache and configuration.

    Returns:
        LangGraphService: The singleton service instance.
    """
    global _langgraph_service
    if _langgraph_service is None:
        _langgraph_service = LangGraphService()
    return _langgraph_service


def inject_user_context(user: Any, base_config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Inject user context into LangGraph config (for multi-tenancy isolation).

    This function injects user information into LangGraph's configurable section,
    allowing graph nodes to access user data.

    Injected Information:
    - user_id: Unique user identifier (for multi-tenancy isolation).
    - user_display_name: User's display name.
    - langgraph_auth_user: Full authentication payload (for graph nodes).

    Use Cases:
    - Accessing user info in graph nodes via Runtime[Context].
    - Filtering data and checking permissions by user.
    - Including user ID in logging and tracing.

    Args:
        user: Authenticated user object (with identity, display_name, to_dict()).
        base_config (dict | None): Existing config (default: {}).

    Returns:
        dict: LangGraph config with user context injected.

    Note:
        - Does not overwrite existing configurable values (uses setdefault).
        - Skips user info injection if user is None.
        - Injects minimal identity if to_dict() fails.
    """
    config: dict[str, Any] = (base_config or {}).copy()
    configurable = config.get("configurable")
    if not isinstance(configurable, dict):
        configurable = {}
    config["configurable"] = configurable

    # Inject user-related data (only if user exists)
    if user:
        # Default user identifier for multi-tenancy isolation
        identity = getattr(user, "identity", None)
        if identity is not None:
            config["configurable"].setdefault("user_id", identity)
        display_name = getattr(user, "display_name", None)
        config["configurable"].setdefault("user_display_name", display_name or identity)

        # Full authentication payload for use in graph nodes
        if "langgraph_auth_user" not in config["configurable"]:
            try:
                payload = user.to_dict()  # type: ignore[attr-defined]
                if isinstance(payload, dict):
                    config["configurable"]["langgraph_auth_user"] = payload
                else:
                    raise TypeError("User payload is not a dictionary")
            except Exception:
                # Fallback: use minimal dictionary if to_dict() is not available
                if identity is not None:
                    config["configurable"]["langgraph_auth_user"] = {"identity": identity}

    return config


def create_thread_config(
    thread_id: str,
    user: Any,
    additional_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a LangGraph config for a specific thread (with user context).

    This function creates a per-thread execution config and automatically injects
    user information. LangGraph uses this config to load the correct thread
    state from the checkpointer.

    Workflow:
    1. Create a base config including thread_id.
    2. Merge additional_config into the base config.
    3. Inject user information with inject_user_context().
    4. Return the completed config.

    Args:
        thread_id (str): Unique thread identifier.
        user: Authenticated user object.
        additional_config (dict | None): Additional config (default: None).

    Returns:
        dict: LangGraph config including thread_id and user context.

    Usage Example:
        config = create_thread_config("thread_123", user)
        state = await graph.aget_state(config)
    """
    base_config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}

    if isinstance(additional_config, dict):
        base_config.update(additional_config)

    return inject_user_context(user, base_config)


def create_run_config(
    run_id: str,
    thread_id: str,
    user: Any,
    additional_config: dict[str, Any] | None = None,
    checkpoint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a LangGraph config for a specific run (with observability callbacks).

    This function creates a per-run config and automatically adds:
    - thread_id, run_id: Execution context identifiers.
    - User context: For multi-tenancy isolation and permission management.
    - Observability callbacks: For integration with tracing systems like Langfuse.
    - Checkpoint parameters: For resuming from a specific state.

    Design Principle:
        This function is **additive** and does not remove or rename any settings
        provided by the client. It just ensures the configurable dictionary
        exists and merges server-side keys so that graph nodes can rely on them.

    Args:
        run_id (str): Unique run identifier.
        thread_id (str): Unique thread identifier.
        user: Authenticated user object.
        additional_config (dict | None): Additional config provided by the client.
        checkpoint (dict | None): Checkpoint parameters (for resuming from a specific state).

    Returns:
        dict: The complete LangGraph run config.
            - configurable: thread_id, run_id, user context, checkpoint params.
            - callbacks: Observability callbacks (e.g., for Langfuse).
            - metadata: Metadata for tracing systems.

    Note:
        - Does not overwrite values already set by the client (uses setdefault).
        - Automatically adds callbacks and metadata if Langfuse is enabled.
        - Checkpoint parameters are merged into configurable.
    """

    cfg: dict[str, Any] = deepcopy(additional_config) if additional_config else {}

    # Ensure the configurable section exists
    cfg.setdefault("configurable", {})

    # Merge server-provided fields (without overwriting if client already set them)
    cfg["configurable"].setdefault("thread_id", thread_id)
    cfg["configurable"].setdefault("run_id", run_id)

    # Add observability callbacks from various potential sources
    tracing_callbacks = get_tracing_callbacks()
    if tracing_callbacks:
        existing_callbacks = cfg.get("callbacks", [])
        if not isinstance(existing_callbacks, list):
            # Could log a warning here for more robustness
            existing_callbacks = []

        # Combine existing callbacks with new tracing callbacks non-destructively
        cfg["callbacks"] = existing_callbacks + tracing_callbacks

        # Add metadata for Langfuse
        cfg.setdefault("metadata", {})
        cfg["metadata"]["langfuse_session_id"] = thread_id
        if user:
            cfg["metadata"]["langfuse_user_id"] = user.identity
            cfg["metadata"]["langfuse_tags"] = [
                "open_langgraph_run",
                f"run:{run_id}",
                f"thread:{thread_id}",
                f"user:{user.identity}",
            ]
        else:
            cfg["metadata"]["langfuse_tags"] = [
                "open_langgraph_run",
                f"run:{run_id}",
                f"thread:{thread_id}",
            ]

    # Apply checkpoint parameters if provided
    if checkpoint:
        cfg["configurable"].update({k: v for k, v in checkpoint.items() if v is not None})

    # Finally, inject user context via the existing helper
    return inject_user_context(user, cfg)
