"""LiteLLM provider implementation for multi-provider support."""

import os
from typing import Any, Optional, Dict, List, Tuple
from urllib.parse import urlparse

import litellm
from litellm import acompletion

from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest


class LiteLLMProvider(LLMProvider):
    """
    LLM provider using LiteLLM for multi-provider support.
    
    Supports OpenRouter, Anthropic, OpenAI, Gemini, and many other providers through
    a unified interface.
    """
    
    def __init__(
        self, 
        api_key: str | None = None, 
        api_base: str | None = None,
        default_model: str = "anthropic/claude-opus-4-5"
    ):
        super().__init__(api_key, api_base)
        self.default_model = default_model
        
        # Detect provider type using robust URL parsing
        self.provider_type = self._detect_provider_type(api_base, default_model)
        
        # Configure LiteLLM based on provider (without global env vars)
        if api_base:
            litellm.api_base = api_base
        
        # Disable LiteLLM logging noise
        litellm.suppress_debug_info = True
    
    def _detect_provider_type(self, api_base: str | None, model: str) -> str:
        """Detect provider type using robust URL parsing and model analysis."""
        if not api_base:
            # Determine from model name
            model_lower = model.lower()
            if 'openrouter' in model_lower:
                return 'openrouter'
            elif 'aihubmix' in model_lower or 'aihubmix' in (api_base or ''):
                return 'aihubmix'
            elif 'deepseek' in model_lower:
                return 'deepseek'
            elif 'anthropic' in model_lower:
                return 'anthropic'
            elif 'openai' in model_lower or 'gpt' in model_lower:
                return 'openai'
            elif 'gemini' in model_lower:
                return 'gemini'
            elif 'zhipu' in model_lower or 'glm' in model_lower or 'zai' in model_lower:
                return 'zhipu'
            elif 'groq' in model_lower:
                return 'groq'
            elif 'moonshot' in model_lower or 'kimi' in model_lower:
                return 'moonshot'
            else:
                return 'custom'
        
        # Parse URL for more reliable detection
        try:
            parsed = urlparse(api_base.lower())
            hostname = parsed.hostname or ''
            
            if 'openrouter' in hostname:
                return 'openrouter'
            elif 'aihubmix' in hostname:
                return 'aihubmix'
            elif 'deepseek' in hostname:
                return 'deepseek'
            elif 'anthropic' in hostname:
                return 'anthropic'
            elif 'openai' in hostname:
                return 'openai'
            elif 'google' in hostname or 'gemini' in hostname:
                return 'gemini'
            elif 'zhipu' in hostname or 'zhipuai' in hostname:
                return 'zhipu'
            elif 'groq' in hostname:
                return 'groq'
            elif 'moonshot' in hostname:
                return 'moonshot'
            else:
                return 'custom'
        except Exception:
            # Fallback to simple string matching
            api_base_lower = api_base.lower()
            if 'openrouter' in api_base_lower:
                return 'openrouter'
            elif 'aihubmix' in api_base_lower:
                return 'aihubmix'
            elif 'deepseek' in api_base_lower:
                return 'deepseek'
            elif 'anthropic' in api_base_lower:
                return 'anthropic'
            elif 'openai' in api_base_lower:
                return 'openai'
            elif 'gemini' in api_base_lower:
                return 'gemini'
            elif 'zhipu' in api_base_lower or 'zhipuai' in api_base_lower:
                return 'zhipu'
            elif 'groq' in api_base_lower:
                return 'groq'
            elif 'moonshot' in api_base_lower:
                return 'moonshot'
            else:
                return 'custom'
    
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """
        Send a chat completion request via LiteLLM.
        
        Args:
            messages: List of message dicts with 'role' and 'content'.
            tools: Optional list of tool definitions in OpenAI format.
            model: Model identifier (e.g., 'anthropic/claude-sonnet-4-5').
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.
        
        Returns:
            LLMResponse with content and/or tool calls.
        """
        model = model or self.default_model
        
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        # Pass API key directly instead of setting global env vars
        if self.api_key:
            kwargs["api_key"] = self.api_key
        
        # Pass api_base directly for custom endpoints
        if self.api_base:
            kwargs["api_base"] = self.api_base
        
        # Handle special cases for specific providers
        if self.provider_type == 'aihubmix':
            # AiHubMix uses OpenAI-compatible API with openai/ prefix
            if not model.startswith("openai/"):
                model_name = model.split('/')[-1] if '/' in model else model
                kwargs["model"] = f"openai/{model_name}"
        
        elif self.provider_type == 'openrouter':
            # OpenRouter requires openrouter/ prefix
            if not model.startswith("openrouter/"):
                kwargs["model"] = f"openrouter/{model}"
        
        elif self.provider_type == 'zhipu':
            # Zhipu/Z.ai models need zai/ prefix
            if not (model.startswith("zai/") or model.startswith("zhipu/")):
                kwargs["model"] = f"zai/{model}"
        
        elif self.provider_type == 'gemini':
            # Gemini models need gemini/ prefix
            if not model.startswith("gemini/"):
                kwargs["model"] = f"gemini/{model}"
        
        elif self.provider_type == 'custom' and self.api_base:
            # Custom vLLM endpoints use hosted_vllm/ prefix per LiteLLM docs
            if not model.startswith("hosted_vllm/"):
                kwargs["model"] = f"hosted_vllm/{model}"
        
        # Handle Kimi/Moonshot temperature requirement
        if self.provider_type == 'moonshot' and temperature != 1.0:
            kwargs["temperature"] = 1.0
        
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        
        try:
            response = await acompletion(**kwargs)
            return self._parse_response(response)
        except Exception as e:
            # Return error as content for graceful handling
            return LLMResponse(
                content=f"Error calling LLM: {str(e)}",
                finish_reason="error",
            )
    
    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse LiteLLM response into our standard format."""
        choice = response.choices[0]
        message = choice.message
        
        tool_calls = []
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                # Parse arguments from JSON string if needed
                args = tc.function.arguments
                if isinstance(args, str):
                    import json
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"raw": args}
                
                tool_calls.append(ToolCallRequest(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                ))
        
        usage = {}
        if hasattr(response, "usage") and response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        
        return LLMResponse(
            content=message.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
        )
    
    def get_default_model(self) -> str:
        """Get the default model."""
        return self.default_model