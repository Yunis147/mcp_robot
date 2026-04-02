import httpx
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class LLMResponse:
    content: str = ""
    tool_calls: List[Dict] = field(default_factory=list)
    thinking: Optional[str] = None
    usage: Dict = field(default_factory=dict)

class OllamaProvider:
    provider_name = "Ollama"

    def __init__(self, model: str = "qwen3.5:27b"):
        self.model = model
        self.base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")

        try:
            r = httpx.get(f"{self.base_url}/api/tags", timeout=5)
            r.raise_for_status()
            print(f"✅ Ollama connected: {self.base_url} | model: {model}")
        except Exception as e:
            raise ConnectionError(f"Cannot reach Ollama at {self.base_url}: {e}")

    def format_tools(self, tools: List[Dict]) -> List[Dict]:
        """Convert MCP tool schemas → Ollama format."""
        ollama_tools = []
        for tool in tools:
            ollama_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("inputSchema", {
                        "type": "object",
                        "properties": {}
                    })
                }
            })
        return ollama_tools

    def format_tool_calls_for_execution(self, tool_calls: List[Dict]) -> List[Dict]:
        return [
            {
                "name": tc["function"]["name"],
                "id": tc.get("id", tc["function"]["name"]),
                "input": tc["function"].get("arguments", {})
            }
            for tc in tool_calls
        ]

    def format_tool_results_for_conversation(self, tool_calls, tool_outputs):
        """Format tool results back into conversation."""
        results = []
        image_parts = []

        for tool_call, output_parts in zip(tool_calls, tool_outputs):
            text_parts = []
            for part in output_parts:
                if part.get("type") == "text":
                    text_parts.append(part["text"])
                elif part.get("type") == "image":
                    image_parts.append(part)

            results.append({
                "role": "tool",
                "content": "\n".join(text_parts),
                "name": tool_call["name"],
                # attach images directly to tool result
                "_images": image_parts
            })

        return results, image_parts

    def _extract_images_from_content(self, content) -> List[str]:
        """Extract base64 image data from content blocks."""
        images = []
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image":
                    # Claude format: part["source"]["data"]
                    source = part.get("source", {})
                    if source.get("type") == "base64":
                        images.append(source["data"])
        return images

    async def generate_response(
        self,
        messages: List[Dict],
        tools: List[Dict] = None,
        temperature: float = 0.1,
        thinking_enabled: bool = False,
        thinking_budget: int = 0,
    ) -> LLMResponse:

        clean_messages = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")

            if not role:
                continue

            # tool results — extract text and images
            if role == "tool":
                text = str(content) if content else ""
                # check for images attached to tool result
                images = []
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "image":
                            source = part.get("source", {})
                            if source.get("type") == "base64":
                                images.append(source["data"])
                    text = " ".join(
                        p.get("text", "") for p in content
                        if isinstance(p, dict) and p.get("type") == "text"
                    )

                msg_dict = {"role": "tool", "content": text}
                if images:
                    msg_dict["images"] = images
                clean_messages.append(msg_dict)
                continue

            # assistant messages
            if role == "assistant":
                if isinstance(content, list):
                    text_only = " ".join(
                        p.get("text", "") for p in content
                        if isinstance(p, dict) and p.get("type") == "text"
                    )
                    content = text_only
                if content:
                    clean_messages.append({"role": "assistant", "content": content})
                continue

            # user and system messages — keep images!
            if isinstance(content, list):
                text_parts = []
                images = []
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif part.get("type") == "image":
                        source = part.get("source", {})
                        if source.get("type") == "base64":
                            images.append(source["data"])

                text = " ".join(text_parts)
                msg_dict = {"role": role, "content": text}
                if images:
                    msg_dict["images"] = images  # ← Ollama image format
                if text or images:
                    clean_messages.append(msg_dict)
            else:
                if content:
                    clean_messages.append({"role": role, "content": content})

        payload = {
            "model": self.model,
            "messages": clean_messages,
            "tools": tools or [],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": 32768
            }
        }

        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(f"{self.base_url}/api/chat", json=payload)

            print(f"🔍 Ollama status: {r.status_code}")
            if r.status_code != 200:
                print(f"🔍 Ollama error: {r.text[:500]}")

            r.raise_for_status()
            data = r.json()

        if not data:
            return LLMResponse(content="Error: empty response from Ollama", tool_calls=[])

        message = data.get("message", {})
        if not message:
            return LLMResponse(content="Error: no message in Ollama response", tool_calls=[])

        content = message.get("content", "")
        raw_tool_calls = message.get("tool_calls", [])

        tool_calls = []
        for tc in raw_tool_calls:
            fn = tc.get("function", {})
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            tool_calls.append({
                "id": fn.get("name", "tool"),
                "function": {
                    "name": fn.get("name", ""),
                    "arguments": args
                }
            })

        usage = {
            "input_tokens": data.get("prompt_eval_count", 0),
            "output_tokens": data.get("eval_count", 0),
        }

        return LLMResponse(content=content, tool_calls=tool_calls, usage=usage)