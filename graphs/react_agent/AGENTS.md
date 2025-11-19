# ReAct Agent

## Graph Overview

The ReAct Agent is a LangGraph-based agent that implements the **Reasoning and Acting (ReAct) pattern**. In this pattern, the LLM progressively constructs an answer by iterating through a thought process (Thought) and tool execution (Action) to solve a problem.

### How the ReAct Pattern Works

```
User Question
    ↓
[Reasoning] The LLM analyzes the situation and decides which tool is needed.
    ↓
[Action] The selected tool is executed (e.g., web search).
    ↓
[Observation] The tool's execution result is passed to the LLM.
    ↓
[Reasoning] The result is analyzed to determine if another tool is needed.
    ↓
A final answer is generated, or the cycle repeats.
```

### Key Features

- **Simple Structure**: Continuous execution without complex interrupts.
- **Automatic Tool Selection**: The LLM automatically determines the appropriate tool based on the context.
- **State Management**: Maintains conversation history and execution context via LangGraph's StateGraph.
- **Infinite Loop Prevention**: Safe execution through a recursion limit.

---

## File Structure

The ReAct Agent is composed of 7 modules, each with a clear responsibility:

```
graphs/react_agent/
├── __init__.py         # Package entry point, exports the compiled graph
├── context.py          # Runtime[Context] pattern - defines execution settings
├── graph.py            # Graph definition - nodes, edges, execution flow
├── prompts.py          # System prompt template
├── state.py            # State schema definition (messages, step counter)
├── tools.py            # Tool functions for the agent to use
└── utils.py            # Helper functions (model loading, message processing)
```

### Role of Each File

#### `__init__.py` - Package Entry Point
- Exposes the compiled `graph` object externally.
- Exports it to be referenced in `open_langgraph.json`.

```python
from react_agent.graph import graph

__all__ = ["graph"]
```

#### `context.py` - Runtime Context
- Implements LangGraph's `Runtime[Context]` pattern.
- Defines configuration parameters needed for agent execution.
- Supports automatic loading of environment variables.

**Key Settings:**
- `system_prompt`: Defines the agent's role and behavior.
- `model`: The LLM to use (e.g., "openai/gpt-4o-mini").
- `max_search_results`: The maximum number of results for the search tool.

#### `graph.py` - Graph Architecture
- Defines nodes and edges via the StateGraph builder.
- Implements the core execution flow of the ReAct pattern.
- Conditional routing logic (tool call vs. end).

#### `prompts.py` - Prompt Template
- Defines the agent's system message.
- Supports dynamic variable substitution (e.g., `{system_time}`).

#### `state.py` - State Schema
- `InputState`: External input interface (user messages).
- `State`: The entire execution state (message history, recursion limit flag).
- Message accumulation via the `add_messages` reducer.

#### `tools.py` - Tool Definition
- Functions that the agent can call.
- Current implementation: `search` tool (simulates a web search).
- Accesses user-specific settings via `Runtime[Context]`.

#### `utils.py` - Utility Functions
- `load_chat_model()`: Initializes an LLM from a "provider/model" format.
- `get_message_text()`: Extracts text from a message object.

---

## Graph Architecture

### Node Configuration

The ReAct Agent consists of 2 nodes:

#### 1. `call_model` Node (Reasoning)
**Role**: Calls the LLM to decide the next action.

```python
async def call_model(state: State, runtime: Runtime[Context]) -> dict:
    model = load_chat_model(runtime.context.model).bind_tools(TOOLS)
    system_message = runtime.context.system_prompt.format(
        system_time=datetime.now(tz=UTC).isoformat()
    )
    response = await model.ainvoke(
        [{"role": "system", "content": system_message}, *state.messages]
    )
    return {"messages": [response]}
```

**Processing Flow:**
1. Load model settings from the Runtime Context.
2. Bind the list of tools to the model (to enable tool calls).
3. Format the system prompt (injecting the current time).
4. Call the LLM (system message + conversation history).
5. Return the response (a text answer or a tool call request).

**Recursion Limit Handling:**
- If `state.is_last_step` is True and the LLM still tries to call a tool, it is forcibly terminated.
- A "Sorry, I could not find an answer..." message is returned.

#### 2. `tools` Node (Execution)
**Role**: Actually executes the tool selected by the LLM.

```python
builder.add_node("tools", ToolNode(TOOLS))
```

**Processing Flow:**
1. Extract `tool_calls` from the AIMessage of the previous node (`call_model`).
2. Execute the corresponding tool function for each `tool_call`.
3. Add the tool's execution result to the state as a `ToolMessage`.
4. Automatically return to the `call_model` node.

### Edge Definition

```
__start__ → call_model ⇄ tools
                ↓
            __end__
```

#### 1. Entry Edge
```python
builder.add_edge("__start__", "call_model")
```
- Always starts execution from the `call_model` node when the graph begins.

#### 2. Conditional Edge (`call_model` output)
```python
def route_model_output(state: State) -> Literal["__end__", "tools"]:
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(f"Expected AIMessage, got {type(last_message).__name__}")

    if not last_message.tool_calls:
        return "__end__"  # No tool calls → end

    return "tools"  # Tool calls present → execute tools

builder.add_conditional_edges("call_model", route_model_output)
```

**Routing Logic:**
- **Tool calls present** → Move to the `tools` node (Action step).
- **No tool calls** → Move to `__end__` (final answer is complete).

#### 3. Fixed Edge (`tools` → `call_model`)
```python
builder.add_edge("tools", "call_model")
```
- Always returns to `call_model` after tool execution is complete.
- Implements the ReAct cycle: Action → Observation → Thought.

### State Management

#### InputState (Input Interface)
```python
@dataclass
class InputState:
    messages: Annotated[Sequence[AnyMessage], add_messages] = field(default_factory=list)
```
- The data structure for incoming external input.
- Contains only user messages.

#### State (Entire Execution State)
```python
@dataclass
class State(InputState):
    is_last_step: IsLastStep = field(default=False)
```
- Extends `InputState` to add execution control information.
- `is_last_step`: A recursion limit flag managed by LangGraph.

**`add_messages` Reducer:**
- Accumulates messages in an "append-only" manner.
- Updates (overwrites) messages with the same ID.
- Supports message modification and retry patterns.

---

## Execution Flow

### Example of a Typical Conversation Flow

If a user asks, "What's the weather like today?":

```
Step 1: __start__ → call_model
├─ Input: HumanMessage("What's the weather like today?")
├─ LLM Analysis: "I need the search tool to get weather information."
└─ Output: AIMessage(tool_calls=[{"name": "search", "args": {"query": "weather today"}}])

Step 2: call_model → tools (Conditional Edge)
├─ Decision: `tool_calls` exist → move to the `tools` node.
└─ Tool Execution: search("weather today")

Step 3: `tools` Node Execution
├─ Search tool is called.
├─ Result returned: "Today is sunny with a temperature of 22 degrees."
└─ Output: ToolMessage(content="Today is sunny with a temperature of 22 degrees.")

Step 4: tools → call_model (Fixed Edge)
├─ The tool's execution result is passed to the LLM.
├─ The LLM analyzes the result to generate a final answer.
└─ Output: AIMessage("Today is a sunny day with a temperature of 22 degrees.")

Step 5: call_model → __end__ (Conditional Edge)
├─ Decision: No `tool_calls` → the final answer is complete.
└─ Graph terminates.
```

### Message Accumulation Pattern

At each step, a message is accumulated in the state's `messages` list:

```python
[
    HumanMessage(content="What's the weather like today?"),
    AIMessage(content="", tool_calls=[...]),           # Tool call request
    ToolMessage(content="Search result..."),               # Tool execution result
    AIMessage(content="Today is a sunny day."),      # Final answer
]
```

### Recursion Limit Handling

LangGraph applies a default recursion limit of 25:

```python
if state.is_last_step and response.tool_calls:
    return {
        "messages": [
            AIMessage(
                content="Sorry, I could not find an answer to your question "
                        "in the specified number of steps."
            )
        ]
    }
```

**How it works:**
1. When the step count reaches `recursion_limit - 1`, `is_last_step` becomes `True`.
2. The `call_model` node detects this and forces termination.
3. This prevents a `RecursionError` from occurring when the `recursion_limit` is reached in the next step.

---

## Customization

### 1. Changing the Prompt

**Method A: Modify `prompts.py`**

```python
# graphs/react_agent/prompts.py
SYSTEM_PROMPT = """You are an expert research assistant.
You have access to various tools to help answer questions.

System time: {system_time}

Instructions:
- Always verify information using available tools
- Provide detailed and accurate responses
- Cite sources when possible
"""
```

**Method B: Override with an environment variable**

```bash
# .env file
SYSTEM_PROMPT="You are a specialized financial advisor. System time: {system_time}"
```

### 2. Changing the Model

**Method A: Modify the default value in `context.py`**

```python
# graphs/react_agent/context.py
@dataclass(kw_only=True)
class Context:
    model: str = field(
        default="anthropic/claude-3-5-sonnet-20241022",  # Change the default model
        metadata={"description": "..."}
    )
```

**Method B: Use an environment variable**

```bash
# .env file
MODEL=anthropic/claude-3-5-sonnet-20241022
```

**Method C: Specify in the API request**

```bash
curl -X POST http://localhost:8000/threads/{thread_id}/runs \
  -H "Content-Type: application/json" \
  -d '{
    "assistant_id": "react_agent",
    "input": {"messages": [{"role": "user", "content": "Hello"}]},
    "config": {"configurable": {"model": "anthropic/claude-3-5-sonnet-20241022"}}
  }'
```

### 3. Adding a Tool

**Step 1: Define a new tool function in `tools.py`**

```python
# graphs/react_agent/tools.py
from langgraph.runtime import get_runtime
from react_agent.context import Context

async def calculator(expression: str) -> dict[str, Any]:
    """Calculates a mathematical expression.

    Args:
        expression (str): The mathematical expression to calculate (e.g., "2 + 2 * 3").

    Returns:
        dict: A dictionary containing the calculation result.
    """
    try:
        result = eval(expression)  # Note: Use a safe parser in production
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"expression": expression, "error": str(e)}

async def get_current_time() -> dict[str, str]:
    """Returns the current time."""
    from datetime import datetime, UTC
    now = datetime.now(tz=UTC)
    return {
        "current_time": now.isoformat(),
        "timestamp": int(now.timestamp())
    }
```

**Step 2: Add it to the `TOOLS` list**

```python
# graphs/react_agent/tools.py
TOOLS: list[Callable[..., Any]] = [
    search,
    calculator,        # Add
    get_current_time,  # Add
]
```

**Step 3: Add tool-specific settings to `Context` (optional)**

```python
# graphs/react_agent/context.py
@dataclass(kw_only=True)
class Context:
    system_prompt: str = field(default=prompts.SYSTEM_PROMPT, metadata={...})
    model: str = field(default="openai/gpt-4o-mini", metadata={...})
    max_search_results: int = field(default=10, metadata={...})

    # Add new tool setting
    enable_calculator: bool = field(
        default=True,
        metadata={"description": "Enable calculator tool for math operations"}
    )
```

### 4. Using Context in a Tool

```python
# graphs/react_agent/tools.py
async def search(query: str) -> dict[str, Any]:
    runtime = get_runtime(Context)  # Get the Runtime Context
    max_results = runtime.context.max_search_results  # Use the setting

    # Example of a real Tavily API call
    from tavily import TavilyClient
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    results = client.search(query, max_results=max_results)

    return {
        "query": query,
        "results": results
    }
```

### 5. Adjusting the Recursion Limit

The recursion limit can be set in `open_langgraph.json` or in an API request:

**`open_langgraph.json` setting:**

```json
{
  "graphs": {
    "react_agent": "./graphs/react_agent/__init__.py:graph"
  },
  "default_config": {
    "recursion_limit": 50
  }
}
```

**Setting in an API request:**

```bash
curl -X POST http://localhost:8000/threads/{thread_id}/runs \
  -H "Content-Type: application/json" \
  -d '{
    "assistant_id": "react_agent",
    "input": {"messages": [{"role": "user", "content": "A complex question"}]},
    "config": {"recursion_limit": 50}
  }'
```

---

## Usage Examples

### 1. Register in `open_langgraph.json`

To register the ReAct Agent with the server, add it to `open_langgraph.json`:

```json
{
  "graphs": {
    "react_agent": "./graphs/react_agent/__init__.py:graph"
  },
  "auth": {
    "path": "./auth.py:auth"
  },
  "env": ".env"
}
```

### 2. Run the Server

```bash
# Start the development server
uv run uvicorn src.agent_server.main:app --reload

# Or use Docker
docker compose up open-langgraph
```

### 3. Look up the Assistant

The ReAct Agent is automatically created as a default assistant:

```bash
curl http://localhost:8000/assistants

# Example response:
{
  "data": [
    {
      "assistant_id": "react_agent",
      "graph_id": "react_agent",
      "name": "ReAct Agent",
      "description": "Reasoning and Action agent",
      "created_at": "2024-01-01T00:00:00Z"
    }
  ]
}
```

### 4. Create a Thread

```bash
curl -X POST http://localhost:8000/threads \
  -H "Content-Type: application/json" \
  -d '{}'

# Example response:
{
  "thread_id": "abc-123-def-456",
  "created_at": "2024-01-01T00:00:00Z"
}
```

### 5. Create and Stream a Run

**Non-streaming (regular execution):**

```bash
curl -X POST http://localhost:8000/threads/abc-123-def-456/runs \
  -H "Content-Type: application/json" \
  -d '{
    "assistant_id": "react_agent",
    "input": {
      "messages": [
        {
          "role": "user",
          "content": "What is the weather today in Seoul?"
        }
      ]
    }
  }'
```

**Server-Sent Events (SSE) streaming:**

```bash
curl -X POST http://localhost:8000/threads/abc-123-def-456/runs/stream \
  -H "Content-Type: application/json" \
  -d '{
    "assistant_id": "react_agent",
    "input": {
      "messages": [
        {
          "role": "user",
          "content": "Tell me about LangGraph"
        }
      ]
    }
  }'
```

**Example streaming response:**

```
event: metadata
data: {"run_id": "run-123"}

event: messages/partial
data: {"content": "Let me search for information"}

event: messages/complete
data: {"role": "assistant", "content": "...", "tool_calls": [...]}

event: tools/start
data: {"tool": "search", "input": {"query": "LangGraph"}}

event: tools/complete
data: {"tool": "search", "output": "LangGraph is..."}

event: messages/complete
data: {"role": "assistant", "content": "LangGraph is a framework for building..."}

event: end
data: {}
```

### 6. Check Run Status

```bash
curl http://localhost:8000/threads/abc-123-def-456/runs/run-123

# Example response:
{
  "run_id": "run-123",
  "thread_id": "abc-123-def-456",
  "assistant_id": "react_agent",
  "status": "success",
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-01-01T00:00:05Z"
}
```

### 7. Check Thread State

```bash
curl http://localhost:8000/threads/abc-123-def-456/state

# Example response:
{
  "values": {
    "messages": [
      {
        "role": "user",
        "content": "What is the weather today?"
      },
      {
        "role": "assistant",
        "content": "",
        "tool_calls": [
          {
            "id": "call_123",
            "name": "search",
            "args": {"query": "weather today"}
          }
        ]
      },
      {
        "role": "tool",
        "content": "Sunny, 22°C",
        "tool_call_id": "call_123"
      },
      {
        "role": "assistant",
        "content": "Today's weather is sunny with a temperature of 22°C."
      }
    ]
  },
  "next": []
}
```

### 8. Using a Python Client

```python
import httpx
import json

async def run_react_agent():
    base_url = "http://localhost:8000"

    # 1. Create a thread
    async with httpx.AsyncClient() as client:
        thread_resp = await client.post(f"{base_url}/threads")
        thread_id = thread_resp.json()["thread_id"]

        # 2. Request execution (streaming)
        async with client.stream(
            "POST",
            f"{base_url}/threads/{thread_id}/runs/stream",
            json={
                "assistant_id": "react_agent",
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": "What's 25 * 4 + 17?"
                        }
                    ]
                },
                "config": {
                    "configurable": {
                        "model": "openai/gpt-4o-mini"
                    }
                }
            }
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    print(f"Event: {data}")

# Execute
import asyncio
asyncio.run(run_react_agent())
```

### 9. Running with Custom Settings

```bash
curl -X POST http://localhost:8000/threads/abc-123-def-456/runs \
  -H "Content-Type: application/json" \
  -d '{
    "assistant_id": "react_agent",
    "input": {
      "messages": [
        {"role": "user", "content": "Research quantum computing"}
      ]
    },
    "config": {
      "configurable": {
        "model": "anthropic/claude-3-5-sonnet-20241022",
        "max_search_results": 15,
        "system_prompt": "You are an expert in quantum physics. System time: {system_time}"
      },
      "recursion_limit": 30
    }
  }'
```

---

## Advanced Usage Patterns

### 1. Multi-turn Conversation

The ReAct Agent automatically maintains conversation history:

```bash
# First question
curl -X POST http://localhost:8000/threads/{thread_id}/runs \
  -d '{"assistant_id": "react_agent", "input": {"messages": [{"role": "user", "content": "What is LangGraph?"}]}}'

# Follow-up question (using the same thread_id)
curl -X POST http://localhost:8000/threads/{thread_id}/runs \
  -d '{"assistant_id": "react_agent", "input": {"messages": [{"role": "user", "content": "How is it different from LangChain?"}]}}'
```

The agent remembers the previous conversation context and understands that "it" refers to LangGraph.

### 2. Adding Metadata

```bash
curl -X POST http://localhost:8000/threads \
  -d '{
    "metadata": {
      "user_id": "user-123",
      "session_type": "support",
      "priority": "high"
    }
  }'
```

### 3. Event Replay

If the connection is lost during streaming, you can replay events:

```bash
curl "http://localhost:8000/threads/{thread_id}/runs/{run_id}/stream?after_event_id=event-42"
```

### 4. Observability (Langfuse Integration)

If Langfuse is enabled, all executions are automatically tracked:

```bash
# .env file
LANGFUSE_LOGGING=true
LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_SECRET_KEY=sk-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

LangGraph executions, tool calls, token usage, etc., will be displayed on the Langfuse dashboard.

---

## Troubleshooting

### Problem 1: "Expected AIMessage in output edges" error

**Cause**: The last message in the `route_model_output` function is not an AIMessage.

**Solution**:
- Ensure that the `call_model` node always returns an AIMessage.
- If you've added a custom node, verify the message type.

### Problem 2: Tool is not being called

**Cause**: The model does not support tool calls, or the tool binding failed.

**Solution**:
```python
# Check in utils.py if the model is supported.
# Examples of models that support tool calls:
# - openai/gpt-4, gpt-3.5-turbo
# - anthropic/claude-3-sonnet, claude-3-opus
# - google/gemini-pro
```

### Problem 3: `RecursionError` occurs

**Cause**: The `recursion_limit` was exceeded.

**Solution**:
```bash
# Increase the recursion limit
curl -X POST http://localhost:8000/threads/{thread_id}/runs \
  -d '{"config": {"recursion_limit": 50}, ...}'
```

### Problem 4: Environment variables are not being applied

**Cause**: Failed to load environment variables in `Context.__post_init__`.

**Solution**:
```python
# Debug in context.py
def __post_init__(self) -> None:
    for f in fields(self):
        if not f.init:
            continue
        print(f"Field: {f.name}, Default: {f.default}, Current: {getattr(self, f.name)}")
        env_value = os.environ.get(f.name.upper())
        print(f"Env value for {f.name.upper()}: {env_value}")
```

---

## References

- **LangGraph Official Documentation**: https://langchain-ai.github.io/langgraph/
- **ReAct Paper**: [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- **Open LangGraph Project CLAUDE.md**: `/Users/jhj/Desktop/personal/opensource-langgraph-platform/CLAUDE.md`
- **LangGraph Tool Calling Guide**: https://langchain-ai.github.io/langgraph/how-tos/tool-calling/

---

## Next Steps

Once you understand the ReAct Agent, explore these advanced patterns:

1. **graphs/react_agent_hitl/**: Human-in-the-Loop pattern (requires user approval)
2. **graphs/subgraph_agent/**: Subgraph composition pattern (complex workflows)
3. **Create a Custom Graph**: Design a new graph that meets your project's requirements.

---

## License

This code is part of the Open LangGraph project and is provided under the MIT License.
