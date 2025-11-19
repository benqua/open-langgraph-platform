# Subgraph Agent

## Overview

The **Subgraph Agent** is an example that demonstrates the subgraph composition pattern, where an existing LangGraph graph is reused as a node. This pattern allows you to build complex agent systems with a modular structure, integrating existing graphs into new workflows.

### Advantages of the Subgraph Pattern

- **Reusability**: Reuse an existing graph (`react_agent`) by inserting it as a node.
- **Modularity**: Separate complex logic into independent subgraphs.
- **Maintainability**: Develop and test each subgraph independently.
- **Composability**: Combine multiple subgraphs to build complex workflows.

### Graph Structure

```
__start__ → no_stream → subgraph_agent → __end__
```

The main graph has a linear structure that goes through a preprocessing node (`no_stream`), executes the subgraph (`react_agent`), and returns the final result.

## File Structure

### 1. `__init__.py`

The module entry point that exports the `graph` object.

**Key Contents:**
- Overview of the subgraph composition pattern.
- `subgraph_agent` node - Executes the `react_agent` graph as a subgraph.
- `no_stream` node - LLM call using a streaming deactivation tag.

### 2. `graph.py`

Defines the main graph that includes the subgraph.

**Key Components:**

#### Node Functions

**`no_stream(state, runtime)`**
- A preprocessing node that calls the LLM with the `langsmith:nostream` tag.
- Generates an initial response before passing it to the subgraph.
- Records the entire response at once in LangSmith tracing, without streaming.

**Operational Flow:**
1. Load model settings and system prompt from the Runtime Context.
2. Initialize the chat model with the `langsmith:nostream` tag.
3. Format the current UTC time into the system prompt.
4. Call the LLM by combining the system message and conversation history.
5. Add the LLM response to the message list and return it.

**`subgraph_agent` Node**
- Adds `react_agent.graph` directly as a node.
- LangGraph allows compiled graphs to be used as nodes.
- The subgraph receives the state from the main graph, executes, and returns the updated state.

#### Graph Builder

```python
builder = StateGraph(State, input_schema=InputState, context_schema=Context)
```

It uses the same `State`, `InputState`, and `Context` as `react_agent` to ensure seamless data flow between the subgraph and the main graph.

## Subgraph Integration Method

### 1. Using the Same State Structure

The main graph and the subgraph share the same `State`, `InputState`, and `Context`:

```python
from react_agent.context import Context
from react_agent.state import InputState, State
from react_agent import graph as react_graph
```

This allows data to flow directly without transformation when passing the state.

### 2. Adding the Subgraph Node

Add the compiled graph directly as a node:

```python
builder.add_node("subgraph_agent", react_graph)
```

LangGraph treats the subgraph like a regular node, passing the main graph's state as input and receiving the updated state as output.

### 3. Connecting Edges

```python
builder.add_edge("__start__", "no_stream")      # Start → Preprocessing
builder.add_edge("no_stream", "subgraph_agent") # Preprocessing → Subgraph
builder.add_edge("subgraph_agent", "__end__")   # Subgraph → End
```

## Data Flow

### State Sharing Pattern

```
1. Input Message → InputState
2. Execute no_stream node
   - System Prompt + Message History → LLM Call
   - Add AIMessage → Update State
3. Updated State → subgraph_agent (react_agent graph)
   - Execute react_agent's ReAct cycle
   - Repeat tool calls and inference
   - Generate final AIMessage
4. Final State → __end__
```

### Runtime Context Passing

The main graph and the subgraph both share the same `Runtime[Context]`:

- **Model Settings**: `runtime.context.model`
- **System Prompt**: `runtime.context.system_prompt`
- **Search Result Limit**: `runtime.context.max_search_results`

## Streaming Control

### langsmith:nostream Tag

The `no_stream` node uses the `langsmith:nostream` tag to disable streaming for a specific LLM call:

```python
model = load_chat_model(runtime.context.model).with_config(
    config={"tags": ["langsmith:nostream"]}
)
```

**Effect:**
- The LangSmith dashboard shows only the completed response without streaming events.
- The client does not receive intermediate events from that node.
- Useful for setting initial context before subgraph execution.

### Subgraph Streaming

Controlled by the `stream_subgraphs` parameter in the API call:

**Default Behavior (stream_subgraphs=False):**
```python
stream = client.runs.stream(
    thread_id=thread_id,
    assistant_id=assistant_id,
    input={"messages": [...]},
    stream_mode=["messages", "values"]
)
```
- Only events from the `subgraph_agent` node are received.
- Internal subgraph nodes (`call_model`, `tools`) are not streamed.

**Enable Subgraph Streaming (stream_subgraphs=True):**
```python
stream = client.runs.stream(
    thread_id=thread_id,
    assistant_id=assistant_id,
    input={"messages": [...]},
    stream_mode=["messages", "values"],
    stream_subgraphs=True  # Also stream internal subgraph events
)
```
- All node events from the subgraph can be received.
- Allows tracking the execution process of internal nodes like `call_model` and `tools`.

## Customization Guide

### 1. Using a Different Subgraph

To use a different graph instead of `react_agent`:

```python
# Import another graph
from other_agent import graph as other_graph

# Add it as a node
builder.add_node("my_subgraph", other_graph)
```

**Note:**
- The State structure of the subgraph and the main graph must be compatible.
- Add a State transformation node if necessary.

### 2. Customizing the Preprocessing Node

Modify the `no_stream` node to add other preprocessing logic:

```python
async def custom_preprocessing(
    state: State, runtime: Runtime[Context]
) -> dict[str, list[AIMessage]]:
    # Custom preprocessing logic
    # e.g., input validation, data transformation, external API calls, etc.

    model = load_chat_model(runtime.context.model)
    # ... custom logic
    return {"messages": [response]}

builder.add_node("preprocessing", custom_preprocessing)
```

### 3. Adding a Postprocessing Node

If additional processing is needed after the subgraph executes:

```python
async def postprocessing(state: State) -> dict:
    # Postprocess the subgraph result
    last_message = state.messages[-1]
    # ... postprocessing logic
    return {"messages": [...]}

builder.add_node("postprocessing", postprocessing)
builder.add_edge("subgraph_agent", "postprocessing")
builder.add_edge("postprocessing", "__end__")
```

### 4. Combining Multiple Subgraphs

Execute multiple subgraphs sequentially or in parallel for complex workflows:

```python
from react_agent import graph as react_graph
from another_agent import graph as another_graph

builder.add_node("agent1", react_graph)
builder.add_node("agent2", another_graph)

# Sequential execution
builder.add_edge("agent1", "agent2")

# Or conditional branching
def route_to_agent(state: State) -> str:
    # Choose a different subgraph based on the state
    if needs_react_agent(state):
        return "agent1"
    return "agent2"

builder.add_conditional_edges("preprocessing", route_to_agent)
```

## Usage Examples

### 1. Basic Execution

```python
from subgraph_agent import graph

# Execute the composite graph including the subgraph
result = await graph.ainvoke({
    "messages": [
        {"role": "user", "content": "What's the weather like?"}
    ]
})

print(result["messages"][-1].content)
```

### 2. Execution via API

Execute the graph registered in `open_langgraph.json`:

```python
# Create Assistant
assistant = await client.assistants.create(
    graph_id="subgraph_agent",
    if_exists="do_nothing"
)

# Create Thread
thread = await client.threads.create()

# Create and stream Run
stream = client.runs.stream(
    thread_id=thread["thread_id"],
    assistant_id=assistant["assistant_id"],
    input={
        "messages": [
            {"role": "user", "content": "Hello!"}
        ]
    },
    stream_mode=["messages", "values"]
)

async for chunk in stream:
    print(chunk)
```

### 3. Internal Subgraph Streaming

Check the detailed execution process of the subgraph:

```python
stream = client.runs.stream(
    thread_id=thread_id,
    assistant_id=assistant_id,
    input={
        "messages": [
            {"role": "user", "content": "Complex query requiring multiple steps"}
        ]
    },
    stream_mode=["messages", "values"],
    stream_subgraphs=True  # Include internal subgraph events
)

langgraph_node_counts = {}

async for chunk in stream:
    # Track the langgraph_node of the event
    if hasattr(chunk, 'langgraph_node'):
        node = chunk.langgraph_node
        langgraph_node_counts[node] = langgraph_node_counts.get(node, 0) + 1

# With stream_subgraphs=True: receive call_model, tools events
# With stream_subgraphs=False: receive only subgraph_agent events
print(langgraph_node_counts)
```

### 4. Customizing Runtime Context

```python
from react_agent.context import Context

# Execute with a custom context
custom_context = Context(
    model="anthropic/claude-3-5-sonnet-20241022",
    system_prompt="You are a helpful assistant specializing in weather.",
    max_search_results=5
)

result = await graph.ainvoke(
    {"messages": [{"role": "user", "content": "Check the weather"}]},
    config={"context": custom_context}
)
```

## Implementation Details

### State Structure (`react_agent.state`)

```python
@dataclass
class InputState:
    messages: Annotated[Sequence[AnyMessage], add_messages] = field(default_factory=list)

@dataclass
class State(InputState):
    is_last_step: IsLastStep = field(default=False)
```

- **messages**: Conversation history managed by the `add_messages` reducer.
- **is_last_step**: Recursion limit flag managed by LangGraph.

### Context Structure (`react_agent.context`)

```python
@dataclass(kw_only=True)
class Context:
    system_prompt: str = field(default=prompts.SYSTEM_PROMPT)
    model: str = field(default="openai/gpt-4o-mini")
    max_search_results: int = field(default=10)
```

Can be overridden by environment variables:
- `SYSTEM_PROMPT`
- `MODEL`
- `MAX_SEARCH_RESULTS`

### Subgraph Execution Mechanism

LangGraph processes the subgraph as follows:

1. The main graph reaches the `subgraph_agent` node.
2. The current State is passed as input to the subgraph.
3. The subgraph (`react_agent`) executes independently:
   - Performs the ReAct cycle.
   - Repeats tool calls and LLM inference.
   - Generates a final AIMessage.
4. The output State of the subgraph is returned to the main graph.
5. The main graph proceeds to the next node (in this case, `__end__`).

## Testing

E2E tests to verify subgraph behavior:

### 1. Event Filtering Test

`tests/e2e/test_streaming/test_event_filtering_and_subgraphs.py::test_langsmith_nostream_event_filtering_e2e`

- Confirms that events from the `no_stream` node with the `langsmith:nostream` tag are filtered.
- Verifies that events from the `subgraph_agent` node are received normally.

### 2. Subgraph Streaming Test

`tests/e2e/test_streaming/test_event_filtering_and_subgraphs.py::test_subgraphs_streaming_parameter_e2e`

- Confirms that the `stream_subgraphs=True` parameter works correctly.
- Verifies that events from internal subgraph nodes (`call_model`) are received.

## Related Documents

- **React Agent**: `/graphs/react_agent/AGENTS.md` - The basic ReAct agent used as a subgraph.
- **HITL Agent**: `/graphs/react_agent_hitl/AGENTS.md` - An example of the Human-in-the-Loop pattern.
- **Architecture**: `/CLAUDE.md` - Overall system architecture and graph integration methods.
