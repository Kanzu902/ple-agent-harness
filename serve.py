# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A minimal harness: one model, three tools, one loop.

The model gets the task and three tools that act in the task container. Each
tool call runs there and its result goes back to the model, until the model
answers without calling a tool. That answer is the final message.
"""

import asyncio
import json
import os
import re
import shlex

import openai
import taskbridge

MODEL = os.environ["MODEL_NAME"]
# The model's highest reasoning level: max for Claude, GLM, GPT-6 and
# DeepSeek-V4, high for the others (Gemini refuses max). REASONING_EFFORT
# overrides it.
MAX_EFFORT_MODELS = ("claude", "glm", "gpt-6", "deepseek-v4")
EFFORT = os.environ.get("REASONING_EFFORT") or (
    "max" if any(m in MODEL.lower() for m in MAX_EFFORT_MODELS) else "high"
)
WRAP_UP_SEC = 240  # ask for the summary this long before the time limit
WRAP_UP = {
    "role": "user",
    "content": "Time is almost up. Stop and reply with your summary.",
}
RESULT_CHARS = 20_000  # a longer tool result loses its middle
# Seconds between retries once the SDK's own quick retries are exhausted.
RETRY_WAITS = (5, 15, 30, 60, 120)
# How endpoints word a 400 saying the conversation no longer fits the context
# window: Claude, OpenAI and vLLM, Gemini, xAI.
CONTEXT_FULL = re.compile(
    r"context.?(length|window)|prompt is too long|maximum prompt length"
    r"|exceed[^.]{0,40}(token|context)",
    re.I,
)
CUT_KEEP = 1000  # characters a tool result keeps once the context is full
# times in a row an empty or cut-off reply gets a "continue" before the run
# stops
MAX_NUDGES = 3
# lower it for a small context window
MAX_OUTPUT_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", "32000"))
# Claude caches a prompt prefix only when marked with cache_control. Each call
# marks the system prompt and the latest message (using 2 of Claude's 4 cache
# breakpoints) so each turn reads the prior conversation from cache. Other
# providers cache automatically without markers.
CACHE_MARK = "claude" in MODEL.lower()

# The evaluator sets OPENAI_BASE_URL and OPENAI_API_KEY for every model.
# The SDK handles quick retries; recover() handles longer backoffs. Because
# calls are non-streaming, a max-effort response can take over 15 minutes
# before any bytes arrive, so the timeout matches the egress proxy's 30-minute
# idle limit.
client = openai.AsyncOpenAI(max_retries=5, timeout=1800)

SYSTEM = """You are an expert engineer working on a task in a Linux container with no internet access. Your tools act inside it, and bash starts in {workdir}.

How to work:
1. Read the whole task and every project document it points to before choosing an approach. When a document exists in several revisions, base your work on the latest issued one and say which one you used; treat files marked superseded, draft or archive as background only.
2. Before you compute anything, write a requirements list to /tmp/requirements.md: every deliverable, and every constraint, code or standard, load factor and limit that the task or the documents state, each with the file and the page or section it comes from.
3. Compute with code, not by hand, and check units at every step. Every input number must come from a named file or be marked as an assumption in your notes. Never fill in a missing value silently. Extract the text of each long document (pdf, docx, xlsx, dxf, ifc) to a file under /tmp once, then search or read slices of that file instead of opening the original again.
4. Write each deliverable exactly where, under the name and in the format the task asks for. Write it early, then improve it. Produce office and binary formats (docx, xlsx, pdf, zip, ...) with code, so they are real files of that format.
5. The time left is shown after the results of your tools. Use it to verify rather than stopping early: do not finish while requirements on your list are unchecked. Before you finish, open every deliverable again, check it against your requirements list item by item, and fix what fails.
6. When the work is done, answer without calling a tool: a short summary of each deliverable (the path of each file), the key results, the assumptions you made, and an honest list of any requirement you could not meet or verify. Never say the work is complete if a requirement is unmet. That answer is graded together with your files.{mcp}"""


def tool(name, description, **params):
  """Builds an OpenAI function-tool definition dictionary.

  Args:
    name: Tool function name.
    description: Human-readable tool description.
    **params: Parameter names mapped to their JSON Schema type strings.

  Returns:
    An OpenAI function-tool specification dictionary.
  """
  return {
      "type": "function",
      "function": {
          "name": name,
          "description": description,
          "parameters": {
              "type": "object",
              "properties": {k: {"type": v} for k, v in params.items()},
              "required": list(params),
          },
      },
  }


TOOLS = [
    tool(
        "bash",
        "Run a bash command (120 s limit) and return its exit code and output.",
        command="string",
    ),
    tool("read_file", "Read a text file.", path="string"),
    tool(
        "write_file",
        "Create or overwrite a file; parent folders are created.",
        path="string",
        content="string",
    ),
]


# --- MCP support, adapted from sample_submissions/custom-openai/serve.py
# (Apache-2.0). Tools the task serves over MCP (documents, spreadsheets,
# email, CAD) are offered to the model as functions.
DIRECT_MCP_CHARS = 40_000  # offer MCP tools directly up to this much JSON
MAX_TOOLS = 128  # functions one request may offer, the built-in ones included
MCP_DIRECT = (
    "\n\nThe task also offers its own tools (the functions named"
    " <server>__<tool>)."
)
MCP_CATALOG = (
    "\n\nThe task also offers its own tools, listed below by server. Call"
    " mcp_find to see a tool's parameters, then mcp_call to run it.\n{catalog}"
)
MCP_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "mcp_find",
            "description": (
                "Show the description and parameters of the task's own tools"
                " that match a query."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A tool name, or words describing what you need.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mcp_call",
            "description": "Run one of the task's own tools.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tool": {
                        "type": "string",
                        "description": "The tool's name as listed.",
                    },
                    "arguments": {"type": "object"},
                },
                "required": ["tool", "arguments"],
            },
        },
    },
]


class MCPServer(taskbridge.MCPClient):
  """Client for one task MCP server and its discovered tools."""

  def __init__(self, session, spec):
    super().__init__(session, spec["url"], client="custom-harness")
    self.name = spec["name"]
    self.tools = []

  async def start(self):
    await self.initialize()
    self.tools = await self.list_tools()

  async def call(self, name, arguments):
    result = await self.call_tool(name, arguments)
    parts = []
    content = (
        result.get("content") if isinstance(result.get("content"), list) else []
    )
    for item in content:
      kind = item.get("type") if isinstance(item, dict) else None
      if kind == "text":
        parts.append(str(item.get("text", "")))
      elif kind in ("resource", "resource_link"):
        parts.append(json.dumps(item)[:RESULT_CHARS])
      else:
        parts.append(f"[{kind or 'unknown'} content not shown]")
    if not parts and result.get("structuredContent") is not None:
      parts.append(json.dumps(result["structuredContent"]))
    text = "\n".join(parts) or "(no content)"
    return f"error: {text}" if result.get("isError") else text


def unique_name(name, taken):
  candidate, n = name[:64], 1
  while candidate in taken:
    n += 1
    suffix = f"_{n}"
    candidate = name[: 64 - len(suffix)] + suffix
  return candidate


async def connect_mcp(session):
  """Connects to the task's MCP servers; returns (tools map, functions, note)."""
  mcp = {}
  for spec in session.mcp_servers:
    server = MCPServer(session, spec)
    try:
      await server.start()
    except (taskbridge.TaskError, RuntimeError, ValueError) as exc:
      await session.log(f"MCP server {server.name} unavailable: {exc}")
      continue
    for t in server.tools:
      name = re.sub(r"[^A-Za-z0-9_-]", "_", f"{server.name}__{t['name']}")
      mcp[unique_name(name, mcp)] = (server, t)
    await session.log(f"MCP server {server.name}: {len(server.tools)} tools")
  if not mcp:
    return {}, [], ""
  functions = [
      {
          "type": "function",
          "function": {
              "name": name,
              "description": (t.get("description") or t["name"])[:1024],
              "parameters": (
                  {
                      k: v
                      for k, v in (t.get("inputSchema") or {}).items()
                      if k != "$schema"
                  }
                  or {"type": "object", "properties": {}}
              ),
          },
      }
      for name, (_, t) in mcp.items()
  ]
  if (
      len(json.dumps(functions)) <= DIRECT_MCP_CHARS
      and len(TOOLS) + len(functions) <= MAX_TOOLS
  ):
    return mcp, functions, MCP_DIRECT
  servers = {}
  for name, (server, _) in mcp.items():
    servers.setdefault(server.name, []).append(name)
  catalog = "\n".join(f"{s}: {', '.join(n)}" for s, n in servers.items())
  return mcp, MCP_TOOLS, MCP_CATALOG.format(catalog=catalog)


def find_mcp(mcp, query):
  """Describes the MCP tools that best match `query`."""
  words = [w for w in re.split(r"[^a-z0-9]+", query.lower()) if w]

  def score(item):
    name, (_, t) = item
    text = f"{name} {t.get('description') or ''}".lower()
    return (
        (query == name) * 100
        + sum(w in name.lower() for w in words) * 3
        + sum(w in text for w in words)
    )

  best = [
      item
      for item in sorted(mcp.items(), key=score, reverse=True)[:8]
      if score(item) > 0
  ]
  if not best:
    return f"no tool matches {query!r}; the names are listed in the system prompt"
  found = [
      f"{name}: {(t.get('description') or '')[:400]}\nparameters:"
      f" {json.dumps(t.get('inputSchema') or {})}"
      for name, (_, t) in best
  ]
  while len(found) > 1 and len("\n\n".join(found)) > RESULT_CHARS:
    found.pop()
  return "\n\n".join(found)


# --- Deliverable guard: before finishing, check that the files the task
# instruction names by absolute path exist.
MAX_GUARDS = 2  # how many times the model is sent back at most
PATH_RE = re.compile(
    r"(?<![:\w/])(/[A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-]+)*"
    r"\.[A-Za-z0-9]{1,6})(?![A-Za-z0-9_/])"
)
GUARD_NOTE = (
    "Before you finish: the task names these files, and they do not exist at"
    " those exact paths yet:\n{files}\n"
    "If one of them is a deliverable, create it at exactly that path now."
    " Use absolute paths: each bash command starts in a new shell, so a cd or"
    " chdir in an earlier command does not carry over. Use ls or find to learn"
    " where your output actually went. If a file is not a deliverable, say so"
    " in your final answer, then finish."
)


def instruction_paths(text):
  """Absolute file paths mentioned in the task instruction, without repeats."""
  found = []
  for p in PATH_RE.findall(text or ""):
    if p not in found:
      found.append(p)
  return found[:40]


async def missing_paths(session, paths):
  """The paths among `paths` that do not exist in the task container."""
  missing = []
  for p in paths:
    try:
      result = await session.exec(f"test -e {shlex.quote(p)}")
    except taskbridge.TaskError:
      continue
    if result["exit_code"] != 0:
      missing.append(p)
  return missing


# --- Old-result trimming: fewer tokens re-sent at every step.
TRIM_KEEP_FULL = 6  # the newest tool results are always kept whole
TRIM_OVER = 3000  # older results longer than this are shortened
TRIM_HEAD = 1500  # characters of a shortened result that stay in the chat
TRIM_NOTE = (
    "\n[... {cut} characters trimmed from this old tool result to save"
    " tokens; {where}. Read it again with read_file, or search it with grep in"
    " bash, if you need it.]"
)


async def trim_old_results(session, messages):
  """Shortens big tool results older than the newest TRIM_KEEP_FULL ones."""
  old = [n for n, m in enumerate(messages) if m["role"] == "tool"]
  old = old[:-TRIM_KEEP_FULL]
  todo = [
      n
      for n in old
      if isinstance(messages[n]["content"], str)
      and len(messages[n]["content"]) > TRIM_OVER
  ]
  if not todo:
    return
  try:
    await session.exec("mkdir -p /tmp/tool_results")
  except taskbridge.TaskError:
    pass
  cut_total = 0
  for n in todo:
    content = messages[n]["content"]
    path = f"/tmp/tool_results/result_{n:04d}.txt"
    try:
      await session.write_file(path, content)
      where = f"the full text is saved at {path}"
    except (taskbridge.TaskError, ValueError):
      where = "it was not saved, so run the tool again"
    cut = len(content) - TRIM_HEAD
    messages[n]["content"] = content[:TRIM_HEAD] + TRIM_NOTE.format(
        cut=cut, where=where
    )
    cut_total += cut
  await session.log(
      f"trim: shortened {len(todo)} old tool result(s), about {cut_total}"
      " characters"
  )


def clip(text):
  """Truncates `text` in the middle if it exceeds `RESULT_CHARS` characters.

  Args:
    text: Tool output string to clip.

  Returns:
    The original string if within `RESULT_CHARS`, or a middle-truncated string.
  """
  if len(text) <= RESULT_CHARS:
    return text
  half = RESULT_CHARS // 2
  return (
      f"{text[:half]}\n[... {len(text) - RESULT_CHARS} characters cut"
      f" ...]\n{text[-half:]}"
  )


def marked(messages):
  """Returns the messages a model call sends.

  For Claude, this is a copy whose system prompt and newest message carry
  `cache_control`. The stored conversation stays as it is, so each turn marks
  its own newest message.

  Args:
    messages: The conversation so far, in OpenAI chat format.

  Returns:
    `messages` itself for other models, or the marked copy for Claude.
  """
  if not CACHE_MARK:
    return messages
  out = list(messages)
  for i in {0, len(out) - 1}:
    m, content = out[i], out[i].get("content")
    if isinstance(content, str) and content:
      out[i] = {
          **m,
          "content": [{
              "type": "text",
              "text": content,
              "cache_control": {"type": "ephemeral"},
          }],
      }
    else:
      out[i] = {**m, "cache_control": {"type": "ephemeral"}}
  return out


async def run_tool(session, fn, mcp=None):
  """Executes a single tool call in the task container and returns its output.

  Rewrites `fn["arguments"]` as canonical JSON (or `"{}"` if invalid) so the
  conversation history remains valid for strict endpoints. Any tool failure is
  returned as an error string for the model.

  Args:
    session: Active `taskbridge.Session` connected to the task container.
    fn: Function-call dictionary containing `name` and `arguments`.

  Returns:
    The clipped tool output or error string to send back to the model.
  """
  name = fn["name"]
  try:
    args = json.loads(fn["arguments"] or "{}")
    if not isinstance(args, dict):
      raise TypeError("not a JSON object")
    fn["arguments"] = json.dumps(args, allow_nan=False)
  except (TypeError, ValueError) as exc:
    fn["arguments"] = "{}"
    return (
        f"error: the arguments are not a JSON object ({exc}); send the call"
        " again"
    )
  try:
    if mcp and name == "mcp_call":
      name, args = str(args.get("tool", "")), args.get("arguments") or {}
      args = json.loads(args) if isinstance(args, str) else args
      if name not in mcp:
        return f"error: no tool named {name!r}; use mcp_find"
    if mcp and name in mcp:
      server, t = mcp[name]
      return clip(await server.call(t["name"], args))
    if mcp and name == "mcp_find":
      return clip(find_mcp(mcp, str(args.get("query", ""))))
    if name == "bash":
      result = await session.exec(args["command"])
      return clip(f"exit code {result['exit_code']}\n{result['output']}")
    if name == "read_file":
      return clip(
          (await session.read_file(args["path"])).decode("utf-8", "replace")
      )
    if name == "write_file":
      return (
          f"wrote {await session.write_file(args['path'], args['content'])}"
          f" bytes to {args['path']}"
      )
    return f"error: there is no tool named {name}"
  # ValueError: content with an unpaired surrogate
  except (taskbridge.TaskError, KeyError, ValueError, RuntimeError) as exc:
    return f"error: {type(exc).__name__}: {exc}"


async def recover(session, messages, exc, waits, until):
  """Handles a failed model call and returns whether to retry.

  On a 400 context-overflow error, shortens the oldest tool results. On a rate
  limit, server error, or connection drop, sleeps for the next interval in
  `RETRY_WAITS` (capped at the wrap-up deadline). Errors are logged rather than
  added to the conversation.

  Args:
    session: Active `taskbridge.Session` for the task.
    messages: The conversation, shortened in place when the context is full.
    exc: The `openai.APIError` the call raised.
    waits: Iterator over the remaining intervals in `RETRY_WAITS`.
    until: Seconds before the time limit at which the wrap-up starts.

  Returns:
    `True` to retry the call, `False` to re-raise.
  """
  status = getattr(exc, "status_code", None)  # None: no HTTP answer
  if status == 400 and CONTEXT_FULL.search(str(exc)):
    return await shorten(session, messages)
  wait = next(waits, None)
  if wait is None or not (
      status is None or status in (408, 409, 429) or status >= 500
  ):
    return False
  left = session.time_left()
  if left is not None and left > until:
    wait = min(wait, left - until)
  await session.log(
      f"model call failed ({str(exc)[:200]}); again in {wait:.0f} s"
  )
  await asyncio.sleep(wait)
  return True


async def shorten(session, messages):
  """Truncates the older half of long tool results to `CUT_KEEP` characters.

  Repeated calls truncate progressively newer results. Claude's thinking blocks
  are kept intact because each block's signature is self-contained and Claude
  rejects tool loops if the latest assistant turn loses its thinking block.

  Args:
    session: Active `taskbridge.Session` for the task.
    messages: The conversation, modified in place.

  Returns:
    Whether any tool result was truncated.
  """
  long = [
      n
      for n, m in enumerate(messages)
      if m["role"] == "tool" and len(m["content"]) > 2 * CUT_KEEP
  ]
  if not long:
    return False
  cut = long[: max(1, len(long) // 2)]
  for n in cut:
    messages[n]["content"] = (
        messages[n]["content"][:CUT_KEEP]
        + "\n[... cut to save context; run the tool again if needed]"
    )
  await session.log(f"context full: cut {len(cut)} old tool results")
  return True


async def solve(session):
  """Runs the minimal tool-calling loop on `session` until completion.

  Args:
    session: Active `taskbridge.Session` for the task.

  Returns:
    The final summary message from the model.
  """
  mcp, extra_tools, note = await connect_mcp(session)
  tools = TOOLS + extra_tools
  messages = [
      {"role": "system", "content": SYSTEM.format(workdir=session.workdir, mcp=note)},
      {"role": "user", "content": session.instruction},
  ]
  # head: the start of an answer cut off at the output limit
  nudges, head, waits = 0, "", iter(RETRY_WAITS)
  guards, wanted = 0, instruction_paths(session.instruction)
  while True:
    left = session.time_left()  # None when the task has no time limit
    wrap_up = left is not None and left < WRAP_UP_SEC
    if wrap_up and messages[-1:] != [WRAP_UP]:  # asked once, also on a retry
      messages.append(dict(WRAP_UP))
    await trim_old_results(session, messages)
    try:  # a call still running when the wrap-up starts is stopped
      reply = await asyncio.wait_for(
          client.chat.completions.create(
              model=MODEL,
              messages=marked(messages),
              tools=tools,
              tool_choice="none" if wrap_up else "auto",
              reasoning_effort=EFFORT,
              max_completion_tokens=MAX_OUTPUT_TOKENS,
          ),
          None if left is None or wrap_up else left - WRAP_UP_SEC,
      )
    except asyncio.TimeoutError:
      continue
    except openai.APIError as exc:
      if await recover(session, messages, exc, waits, WRAP_UP_SEC):
        continue
      raise
    waits = iter(RETRY_WAITS)
    if reply.usage:
      details = getattr(reply.usage, "prompt_tokens_details", None)
      await session.usage(
          MODEL,
          reply.usage.prompt_tokens,
          reply.usage.completion_tokens,
          getattr(details, "cached_tokens", 0) or 0,
      )
    message, cut = (
        reply.choices[0].message,
        reply.choices[0].finish_reason == "length",
    )
    text = head + (message.content or "")
    if text:
      # what taskbridge sends if the time runs out first
      session.final_message = text
    # Send the message back as the endpoint returned it: Gemini refuses a tool
    # call that comes back without its thought signature, which is one of the
    # endpoint's own fields.
    turn = message.model_dump(exclude_none=True)
    if wrap_up or not message.tool_calls:
      if (
          not wrap_up
          and message.content
          and not cut
          and guards < MAX_GUARDS
      ):
        missing = await missing_paths(session, wanted)
        if missing:
          guards += 1
          await session.log(
              f"guard: {len(missing)} file(s) named in the task do not exist"
              " yet"
          )
          messages.append(turn)
          messages.append({
              "role": "user",
              "content": GUARD_NOTE.format(files="\n".join(missing[:20])),
          })
          continue
      if wrap_up or (message.content and not cut) or nudges == MAX_NUDGES:
        return session.final_message
      nudges, head = nudges + 1, text
      if message.content:  # cut off at the output limit
        messages.append(turn)
      messages.append({
          "role": "user",
          "content": (
              "Your reply was cut off; continue where it stopped."
              if cut
              else "Continue the task, and finish with your summary."
          ),
      })
      continue
    nudges, head = 0, ""
    messages.append(turn)
    for call in turn["tool_calls"]:
      await session.log(
          f"{call['function']['name']} {call['function']['arguments'][:200]}"
      )
      content = await run_tool(session, call["function"], mcp)
      messages.append(
          {"role": "tool", "tool_call_id": call["id"], "content": content}
      )
    if left is not None:
      messages[-1][
          "content"
      ] += f"\n[{int(session.time_left() // 60)} minutes left]"


if __name__ == "__main__":
  taskbridge.run(solve)
