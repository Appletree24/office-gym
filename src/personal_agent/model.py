"""Bounded chat-completions client accepting text or a single function call."""
import os
from urllib.error import URLError
from urllib.request import Request, build_opener, ProxyHandler, HTTPRedirectHandler
from officegym.harness.policy import ChatPolicy, render_messages, response_message
from .models import AgentError, Turn, canonical, decode


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class ConversationPolicy(ChatPolicy):
    """Real model usage is reported by the server; no retries or model substitution."""

    def __init__(self, *args, max_response_bytes=1048576, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_response_bytes = max_response_bytes

    def _post(self, payload, timeout):
        headers = {"Content-Type": "application/json"}
        if self.api_key_env:
            key = os.environ.get(self.api_key_env)
            if not key:
                raise AgentError("MODEL_CREDENTIAL_UNSET")
            headers["Authorization"] = "Bearer " + key
        request = Request(self.endpoint, data=canonical(payload).encode(), headers=headers, method="POST")
        opener = build_opener(ProxyHandler({}), NoRedirect())
        with opener.open(request, timeout=timeout) as response:
            body = response.read(self.max_response_bytes + 1)
        if len(body) > self.max_response_bytes:
            raise AgentError("MODEL_RESPONSE_TOO_LARGE")
        try:
            result = decode(body)
            if not isinstance(result, dict):
                raise ValueError("object required")
            return result
        except (ValueError, UnicodeError) as exc:
            raise AgentError("INVALID_MODEL_RESPONSE") from exc

    def generate(self, messages, tools, remaining):
        try:
            response, usage = self._request(render_messages(messages), remaining, tools)
            if usage["completion_tokens"] > min(self.max_output_tokens, remaining["generated_tokens"]):
                raise AgentError("MODEL_OUTPUT_BUDGET_EXCEEDED")
            message = response_message(response)
            calls = message.get("tool_calls", [])
            content = message.get("content")
            if not isinstance(calls, list) or len(calls) > 1:
                raise ValueError("single function call required")
            if response["choices"][0].get("finish_reason") not in (None, "stop", "tool_calls"):
                raise ValueError("incomplete model output")
            if content is not None and not isinstance(content, str):
                raise ValueError("invalid text")
            if calls:
                call = calls[0]
                if call.get("type") != "function":
                    raise ValueError("function call required")
                name = call["function"]["name"]
                args = decode(call["function"]["arguments"])
                if not isinstance(name, str) or not name or not isinstance(args, dict):
                    raise ValueError("invalid arguments")
                return Turn(content, name, args, usage["prompt_tokens"], usage["completion_tokens"])
            if not content or not content.strip():
                raise ValueError("text or tool call required")
            return Turn(content=content, input_tokens=usage["prompt_tokens"], output_tokens=usage["completion_tokens"])
        except AgentError:
            raise
        except (URLError, OSError, TimeoutError) as exc:
            raise AgentError("MODEL_REQUEST_FAILED") from exc
        except Exception as exc:
            raise AgentError("INVALID_MODEL_RESPONSE") from exc
