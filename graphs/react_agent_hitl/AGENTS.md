# Human-in-the-Loop ReAct Agent

This directory implements a ReAct agent with a **Human-in-the-Loop (HITL)** pattern, which requires human approval before executing tools.

## Overview

### What is the HITL Pattern?

Human-in-the-Loop (HITL) is a pattern where an agent seeks human review and approval before performing critical actions. This agent utilizes LangGraph's `interrupt()` function to pause execution before tool execution, allowing the user to choose one of the following options:

- **accept**: Execute the tool with the original arguments.
- **edit**: Modify the tool arguments and then execute.
- **response**: Cancel the tool execution and substitute it with a user message.
- **ignore**: Cancel the tool execution and end the conversation.

### Differences from the Basic ReAct Agent

| Feature | react_agent | react_agent_hitl |
|---|---|---|
| Tool Execution | Automatic | Requires user approval |
| Interrupt | None | `human_approval` node |
| Tool Modification | Not possible | Can modify arguments before execution |
| User Control | Low | High (reviews all tool calls) |

### Operational Flow

```
Start
  ↓
call_model (LLM decides to call a tool)
  ↓
Tool call present?
  ├─ Yes → human_approval (interrupt, wait for user approval)
  │         ↓
  │      Process user response
  │         ├─ accept → tools (execute tool) → call_model
  │         ├─ edit → tools (execute with modified arguments) → call_model
  │         ├─ response → call_model (with user message)
  │         └─ ignore → END (terminate)
  │
  └─ No → END (final response)
```

## File Structure

### 1. `__init__.py`
**Role**: Package entry point and graph export

```python
from react_agent_hitl.graph import graph

__all__ = ["graph"]
```

- Exposes the `graph` object to be referenced in `open_langgraph.json`.
- Provides package-level documentation.

### 2. `context.py`
**Role**: Defines the runtime settings context.

**Key Components**:
- `Context` dataclass: Configuration parameters required for agent execution.
  - `system_prompt`: Defines the agent's behavior.
  - `model`: The language model to use (e.g., `"openai/gpt-4o-mini"`).
  - `max_search_results`: Maximum number of search results.

**Features**:
- Automatic loading of environment variables: `__post_init__` prioritizes environment variables.
- Integration with LangGraph's template system: Metadata annotation on the `model` field.
- Accessible from graph nodes using the `Runtime[Context]` pattern.

**Usage Example**:
```python
async def call_model(state: State, runtime: Runtime[Context]):
    model = load_chat_model(runtime.context.model)
    prompt = runtime.context.system_prompt.format(...)
```

### 3. `state.py`
**Role**: Defines the graph's state structure.

**Key Components**:
- `InputState`: External interface (input provided by the client).
  - `messages`: Conversation message history (using the `add_messages` reducer).

- `State`: The complete internal state (extends `InputState`).
  - `is_last_step`: Flag indicating if the recursion limit has been reached (a LangGraph-managed variable).

**Message Accumulation Pattern**:
1. `HumanMessage` - User input.
2. `AIMessage` (with `tool_calls`) - Agent's tool call request.
3. **[Interrupt Occurs]** - Awaiting user approval.
4. `ToolMessage(s)` - Tool execution results or cancellation messages.
5. `AIMessage` (without `tool_calls`) - Final response.
6. `HumanMessage` - Next conversation turn.

**State Handling on Interrupt**:
- When `interrupt()` is called, the current state is preserved in a checkpoint.
- If the user modifies a tool, the `AIMessage` is updated.
- If the user chooses to respond, a `HumanMessage` is added.

### 4. `graph.py`
**Role**: Implements the core graph logic and HITL mechanism.

**Key Nodes**:

#### `call_model(state: State, runtime: Runtime[Context])`
Calls the LLM to decide the next action.
- Binds the list of tools to the model.
- Formats the system prompt (including the current time).
- Returns an error message if the recursion limit is reached.

#### `human_approval(state: State)` ⭐
**Core Interrupt Point**

Requests user approval before tool execution.

**Operational Flow**:
1. Find the most recent AI message containing tool calls.
2. Pause execution by calling `interrupt()`.
   ```python
   human_response = interrupt({
       "action_request": {
           "action": "tool_execution",
           "args": {tc["name"]: tc.get("args", {}) for tc in tool_message.tool_calls}
       },
       "config": {
           "allow_respond": True,
           "allow_accept": True,
           "allow_edit": True,
           "allow_ignore": True
       }
   })
   ```
3. Wait for user response (state is saved in a checkpoint).
4. Branch based on the user's response.

#### `route_model_output(state: State)`
Determines the next node based on the model's output.
- If tool calls are present → `human_approval`
- If no tool calls → `END`

**Helper Functions**:
- `_find_tool_message()`: Finds the most recent AI message with tool calls.
- `_create_tool_cancellations()`: Creates tool cancellation messages.
- `_parse_args()`: Parses JSON string arguments.
- `_update_tool_calls()`: Updates tool calls with user-modified arguments.

**Graph Structure**:
```python
builder = StateGraph(State, input_schema=InputState, context_schema=Context)
builder.add_node(call_model)
builder.add_node("tools", ToolNode(TOOLS))
builder.add_node(human_approval)
builder.add_edge("__start__", "call_model")
builder.add_conditional_edges("call_model", route_model_output, path_map=["human_approval", END])
builder.add_edge("tools", "call_model")
graph = builder.compile(name="ReAct Agent")
```

### 5. `prompts.py`
**Role**: Defines the system prompt template.

```python
SYSTEM_PROMPT = """You are a helpful AI assistant.

System time: {system_time}"""
```

- Defines the agent's persona and behavior.
- The `{system_time}` variable is dynamically replaced at runtime.
- Can be customized with a more detailed prompt for production environments.

### 6. `tools.py`
**Role**: Defines the tools the agent can use.

**Key Components**:
- `search(query: str)`: A web search tool.
  - Tavily-based search (currently simulated).
  - Accesses `max_search_results` setting via `Runtime[Context]`.
  - Implemented as an asynchronous function.

```python
async def search(query: str) -> dict[str, Any] | None:
    runtime = get_runtime(Context)
    return {
        "query": query,
        "max_search_results": runtime.context.max_search_results,
        "results": f"Simulated search results for '{query}'"
    }

TOOLS: list[Callable[..., Any]] = [search]
```

**How to Extend**:
- Define new tool functions and add them to the `TOOLS` list.
- Each tool should have a docstring explaining its usage (for the LLM to reference).
- In a production environment, this should be implemented with actual API calls.

### 7. `utils.py`
**Role**: Common utility functions.

**Key Functions**:

#### `get_message_text(msg: BaseMessage) -> str`
Extracts text content from a message.
- Supports simple strings, dictionaries, and multimodal lists.
- Extracts and returns only the text.

#### `load_chat_model(fully_specified_name: str) -> BaseChatModel`
Loads a chat model from a "provider/model" format.
- Examples: `"openai/gpt-4"`, `"anthropic/claude-3-opus"`.
- Useful for specifying models via configuration files or environment variables.

## Interrupt Mechanism in Detail

### 1. Interrupt Trigger

```python
human_response = interrupt({
    "action_request": {
        "action": "tool_execution",
        "args": {tc["name"]: tc.get("args", {}) for tc in tool_message.tool_calls}
    },
    "config": {
        "allow_respond": True,   # User can respond directly
        "allow_accept": True,    # User can approve the tool
        "allow_edit": True,      # User can edit tool arguments
        "allow_ignore": True     # User can deny tool execution
    }
})
```

### 2. Checkpoint Saving

- LangGraph automatically saves the current state to PostgreSQL.
- Can be restored later using the thread ID and checkpoint ID.
- Message history and metadata are all preserved.

### 3. Client Notification

- An interrupt event is sent via an SSE (Server-Sent Events) stream.
- The event includes `action_request` and `config`.
- The client displays an approval/denial UI to the user.

### 4. Awaiting User Response

- Execution is paused, waiting for user input.
- No timeout (waits until the user makes a decision).
- The thread is managed independently of other requests.

## Handling User Responses

### 1. Accept

**Request**:
```json
[{"type": "accept"}]
```

**Processing**:
```python
if response_type == "accept":
    return Command(goto="tools")
```

**Result**:
- The tool is executed with the original arguments.
- Routes to the `tools` node.
- Returns to `call_model` after tool execution.

### 2. Edit

**Request**:
```json
[{
    "type": "edit",
    "args": {
        "args": {
            "search": {
                "query": "modified search query"
            }
        }
    }
}]
```

**Processing**:
```python
elif response_type == "edit" and isinstance(response_args, dict) and "args" in response_args:
    updated_calls = _update_tool_calls(tool_message.tool_calls, response_args)
    updated_message = AIMessage(
        content=tool_message.content,
        tool_calls=updated_calls,
        id=tool_message.id
    )
    return Command(goto="tools", update={"messages": [updated_message]})
```

**Result**:
- The tool arguments are updated with the user-provided values.
- The state is updated with the modified `AIMessage`.
- The tool is executed with the modified arguments.

### 3. Response

**Request**:
```json
[{
    "type": "response",
    "args": "I don't think we need to search for that."
}]
```

**Processing**:
```python
elif response_type == "response":
    tool_responses = _create_tool_cancellations(
        tool_message.tool_calls, "was interrupted for human input"
    )
    human_message = HumanMessage(content=str(response_args))
    return Command(
        goto="call_model",
        update={"messages": tool_responses + [human_message]}
    )
```

**Result**:
- The tool calls are converted into cancellation messages.
- The user's text is added as a `HumanMessage`.
- Routes to `call_model`, where the model responds to the new context.

### 4. Ignore

**Request**:
```json
[{"type": "ignore"}]
```

**Processing**:
```python
else:  # ignore or invalid format
    reason = (
        "cancelled by human operator"
        if response_type == "ignore"
        else "invalid format"
    )
    tool_responses = _create_tool_cancellations(tool_message.tool_calls, reason)
    return Command(goto=END, update={"messages": tool_responses})
```

**Result**:
- The tool calls are converted into cancellation messages.
- The graph execution terminates (`END`).
- The conversation is stopped.

## Resumption Workflow

### 1. Initial Execution

```bash
# Create a thread and start execution
POST /threads/{thread_id}/runs
Content-Type: application/json

{
  "assistant_id": "react_agent_hitl",
  "input": {
    "messages": [
      {"role": "user", "content": "Search for latest AI news"}
    ]
  }
}
```

### 2. Receiving an Interrupt Event

The client receives the following event from the SSE stream:

```json
{
  "event": "interrupt",
  "data": {
    "action_request": {
      "action": "tool_execution",
      "args": {
        "search": {
          "query": "latest AI news"
        }
      }
    },
    "config": {
      "allow_respond": true,
      "allow_accept": true,
      "allow_edit": true,
      "allow_ignore": true
    }
  }
}
```

### 3. User Decision

The client displays an approval UI to the user:
- "Do you want to execute the tool?"
- "Do you want to modify the tool arguments?"
- "Do you want to provide a response directly?"
- "Do you want to cancel the tool execution?"

### 4. Resuming Execution

Resume with the response type chosen by the user:

```bash
POST /threads/{thread_id}/runs/{run_id}
Content-Type: application/json

[{"type": "accept"}]
# or
[{"type": "edit", "args": {"args": {"search": {"query": "modified query"}}}}]
# or
[{"type": "response", "args": "I'll answer directly..."}]
# or
[{"type": "ignore"}]
```

### 5. Execution Completion

- `accept` or `edit`: The model generates a final response after the tool executes.
- `response`: The model generates a response based on the user's message.
- `ignore`: Execution terminates immediately.

## Usage Examples

### Example 1: Basic Approval Flow

**Scenario**: The user requests a search, and approves the agent's tool call.

1. **User Input**:
   ```
   "What's the weather in Seoul?"
   ```

2. **Model Response** (Tool Call):
   ```json
   {
     "tool_calls": [{
       "name": "search",
       "args": {"query": "weather Seoul"}
     }]
   }
   ```

3. **Interrupt Occurs**:
   - The client receives an approval request.
   - Displays "Search for 'weather Seoul'?" to the user.

4. **User Approval**:
   ```json
   [{"type": "accept"}]
   ```

5. **Tool Execution**:
   - The search tool is executed.
   - The result is added to the messages.

6. **Final Response**:
   ```
   "The current weather in Seoul is..."
   ```

### Example 2: Modifying Tool Arguments

**Scenario**: The user makes a search query more specific.

1. **User Input**:
   ```
   "Find information about Python"
   ```

2. **Model Response** (Tool Call):
   ```json
   {
     "tool_calls": [{
       "name": "search",
       "args": {"query": "Python"}
     }]
   }
   ```

3. **Interrupt Occurs**:
   - The user decides that "Python" is too broad.

4. **User Modification**:
   ```json
   [{
     "type": "edit",
     "args": {
       "args": {
         "search": {
           "query": "Python programming language latest features 2024"
         }
       }
     }
   }]
   ```

5. **Modified Tool Execution**:
   - The search is executed with the more specific query.
   - More relevant results are returned.

6. **Final Response**:
   ```
   "Here are the latest Python features in 2024..."
   ```

### Example 3: Providing a Direct Response

**Scenario**: The user provides the answer directly instead of letting the tool run.

1. **User Input**:
   ```
   "What's 2+2?"
   ```

2. **Model Response** (Tool Call):
   ```json
   {
     "tool_calls": [{
       "name": "search",
       "args": {"query": "2+2"}
     }]
   }
   ```
   (The model unnecessarily tries to search.)

3. **Interrupt Occurs**:
   - The user decides the search is unnecessary.

4. **User Response**:
   ```json
   [{
     "type": "response",
     "args": "The answer is 4. No need to search."
   }]
   ```

5. **Model Recalled**:
   - The model is called with the tool cancellation message + user message.
   - The model generates a response with the new context.

6. **Final Response**:
   ```
   "You're right! The answer is 4."
   ```

### Example 4: Canceling Tool Execution

**Scenario**: The user completely rejects the tool execution.

1. **User Input**:
   ```
   "Delete all my files"
   ```

2. **Model Response** (Tool Call):
   ```json
   {
     "tool_calls": [{
       "name": "file_delete",
       "args": {"pattern": "*"}
     }]
   }
   ```

3. **Interrupt Occurs**:
   - A dangerous operation is detected.
   - A warning is displayed to the user.

4. **User Rejection**:
   ```json
   [{"type": "ignore"}]
   ```

5. **Execution Termination**:
   - A tool cancellation message is added.
   - The graph execution terminates immediately.
   - The conversation is safely stopped.

## Production Considerations

### 1. Tool Implementation

The current `search` tool is a simulation. In a real environment:

```python
async def search(query: str) -> dict[str, Any] | None:
    runtime = get_runtime(Context)

    # Tavily API call
    from tavily import TavilyClient
    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

    results = await client.search(
        query=query,
        max_results=runtime.context.max_search_results
    )

    return {
        "query": query,
        "results": results
    }
```

### 2. System Prompt Customization

A more detailed prompt is recommended for production environments:

```python
SYSTEM_PROMPT = """You are a helpful AI assistant specialized in [domain].

Guidelines:
- Always verify information before making tool calls
- Ask for clarification if the request is ambiguous
- Be mindful of sensitive operations

System time: {system_time}

Available tools:
- search: Find information on the web
- [other tools...]
"""
```

### 3. Timeout Handling

Since interrupts wait indefinitely, implementing a timeout at the application level is recommended:

```python
# Client-side timeout
timeout = 300  # 5 minutes
if time_since_interrupt > timeout:
    # Automatically send an ignore response
    await send_resume([{"type": "ignore"}])
```

### 4. Error Handling

Strengthen user response format validation:

```python
def validate_response(response: dict) -> bool:
    response_type = response.get("type")

    if response_type == "edit":
        # Validate args.args structure
        if not isinstance(response.get("args"), dict):
            return False
        if "args" not in response["args"]:
            return False

    elif response_type == "response":
        # Validate text response
        if not response.get("args"):
            return False

    return True
```

### 5. Logging and Monitoring

Track interrupt points:

```python
import logging

logger = logging.getLogger(__name__)

async def human_approval(state: State) -> Command:
    logger.info(f"Interrupt triggered for tools: {[tc['name'] for tc in tool_message.tool_calls]}")

    human_response = interrupt(...)

    logger.info(f"User response: {human_response[0].get('type')}")

    # ...processing logic
```

## Known Limitations

### 1. Command(goto=END) Bug

A known bug in LangGraph can cause `Command(goto=END)` to create an infinite loop.

- **GitHub Issue**: https://github.com/langchain-ai/langgraph/issues/5572
- **Impact**: Can occur when handling the `ignore` response type.
- **Solution**: Wait for a LangGraph update or implement alternative termination logic.

### 2. Multiple Tool Calls

If the model calls multiple tools simultaneously:
- Currently, all tools are approved/rejected/modified at once.
- Future improvement: Extend to allow individual approval per tool.

### 3. Nested Interrupts

Additional interrupts are not possible while an interrupt is in progress:
- Only one interrupt can be handled at a time.
- A new interrupt can occur on the next tool call after resumption.

## References

- **LangGraph Documentation**: https://langchain-ai.github.io/langgraph/
- **Interrupt Guide**: https://langchain-ai.github.io/langgraph/how-tos/human-in-the-loop/
- **Basic ReAct Agent**: `/graphs/react_agent/AGENTS.md`
- **Agent Protocol Spec**: https://github.com/AI-Engineer-Foundation/agent-protocol

## Next Steps

1. **Register in `open_langgraph.json`**:
   ```json
   {
     "graphs": {
       "react_agent_hitl": "./graphs/react_agent_hitl/__init__.py:graph"
     }
   }
   ```

2. **Implement Real Tools**: Integrate the Tavily API in `tools.py`.

3. **Custom Prompts**: Modify `prompts.py` to fit your domain.

4. **UI Implementation**: Develop an approval/rejection interface on the client side.

5. **Testing**: Validate the HITL flow with various scenarios.
