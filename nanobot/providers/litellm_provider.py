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
    
    # Provider detection patterns - more robust than simple string matching
    _PROVIDER_PATTERNS = {
        'openrouter': [
            lambda base, key: key and key.startswith("sk-or-"),
            lambda base, key: base and 'openrouter' in base.lower()
        ],
        'aihubmix': [
            lambda base, key: base and any(domain in base.lower() 
                                         for domain in ['aihubmix.com', 'aihubmix.cn'])
        ],
        'vllm': [
            lambda base, key: base and not any(
                pattern(base, key) for provider, patterns in [
                    ('openrouter', _PROVIDER_PATTERNS['openrouter']),
                    ('aihubmix', _PROVIDER_PATTERNS['aihubmix'])
                ] for pattern in patterns
            ) and base.startswith(('http://', 'https://'))
        ]
    }
    
    # Model prefixing rules - simplified and more maintainable
    _MODEL_PREFIX_RULES = [
        # (keywords, target_prefix)
        (['glm', 'zhipu'], 'zai'),
        (['qwen', 'dashscope'], 'dashscope'),
        (['moonshot', 'kimi'], 'moonshot'),
        (['gemini'], 'gemini'),
    ]
    
    def __init__(
        self, 
        api_key: str | None = None, 
        api_base: str | None = None,
        default_model: str = "anthropic/claude-opus-4-5",
        extra_headers: dict[str, str] | None = None,
    ):
        super().__init__(api_key, api_base)
        self.default_model = default_model
        self.extra_headers = extra_headers or {}
        
        # Detect provider type using robust pattern matching
        self.provider_type = self._detect_provider_type(api_base, api_key)
        
        # Configure LiteLLM without polluting global environment
        self._configure_litellm(api_key, api_base, default_model)
        
        # Disable LiteLLM logging noise
        litellm.suppress_debug_info = True
    
    def _detect_provider_type(self, api_base: str | None, api_key: str | None) -> str:
        """Detect provider type using robust pattern matching."""
        if not api_base and not api_key:
            return "direct"  # Direct provider usage
            
        for provider, patterns in self._PROVIDER_PATTERNS.items():
            if any(pattern(api_base, api_key) for pattern in patterns):
                return provider
                
        return "direct"
    
    def _configure_litellm(self, api_key: str | None, api_base: str | None, default_model: str) -> None:
        """Configure LiteLLM without setting global environment variables."""
        if not api_key:
            return
            
        # Instead of setting global env vars, we'll pass keys directly to acompletion
        # LiteLLM supports passing keys via kwargs for most providers
        self._api_keys = {}
        
        model_lower = default_model.lower()
        
        # Map model keywords to provider keys
        key_mapping = {
            'deepseek': 'DEEPSEEK_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY', 
            'openai': 'OPENAI_API_KEY',
            'gpt': 'OPENAI_API_KEY',
            'gemini': 'GEMINI_API_KEY',
            'zhipu': 'ZHIPUAI_API_KEY',
            'glm': 'ZHIPUAI_API_KEY',
            'zai': 'ZAI_API_KEY',
            'dashscope': 'DASHSCOPE_API_KEY',
            'qwen': 'DASHSCOPE_API_KEY',
            'groq': 'GROQ_API_KEY',
            'moonshot': 'MOONSHOT_API_KEY',
            'kimi': 'MOONSHOT_API_KEY',
        }
        
        # Set appropriate API key based on model
        for keyword, env_var in key_mapping.items():
            if keyword in model_lower:
                self._api_keys[env_var] = api_key
                break
        else:
            # Default fallback - assume OpenAI-compatible
            self._api_keys['OPENAI_API_KEY'] = api_key
    
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
        
        # Apply model prefixing rules
        model = self._apply_model_prefix(model)
        
        # Apply provider-specific model formatting
        model = self._format_model_for_provider(model)
        
        # Handle provider-specific constraints
        temperature = self._adjust_temperature_for_model(model, temperature)
        
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        # Pass API keys directly instead of using environment variables
        kwargs.update({key.lower(): value for key, value in self._api_keys.items()})
        
        # Pass api_base directly for custom endpoints
        if self.api_base:
            kwargs["api_base"] = self.api_base
        
        # Pass extra headers (e.g. APP-Code for AiHubMix)
        if self.extra_headers:
            kwargs["extra_headers"] = self.extra_headers
        
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
    
    def _apply_model_prefix(self, model: str) -> str:
        """Apply model prefixing rules based on model name."""
        model_lower = model.lower()
        
        # Skip if already prefixed
        skip_prefixes = ['openrouter/', 'hosted_vllm/', 'zai/', 'dashscope/', 'moonshot/', 'gemini/']
        if any(model.startswith(prefix) for prefix in skip_prefixes):
            return model
            
        for keywords, prefix in self._MODEL_PREFIX_RULES:
            if any(kw in model_lower for kw in keywords):
                return f"{prefix}/{model}"
                
        return model
    
    def _format_model_for_provider(self, model: str) -> str:
        """Format model name based on detected provider type."""
        if self.provider_type == 'openrouter' and not model.startswith('openrouter/'):
            return f"openrouter/{model}"
        elif self.provider_type == 'aihubmix':
            # AiHubMix uses OpenAI-compatible API, so strip any existing prefix
            model_name = model.split('/')[-1]
            return f"openai/{model_name}"
        elif self.provider_type == 'vllm':
            return f"hosted_vllm/{model}"
            
        return model
    
    def _adjust_temperature_for_model(self, model: str, temperature: float) -> float:
        """Adjust temperature for models with specific constraints."""
        if "kimi-k2.5" in model.lower():
            return 1.0
        return temperature
    
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