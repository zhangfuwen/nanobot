"""Configuration schema using Pydantic."""

from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings


class WhatsAppConfig(BaseModel):
    """WhatsApp channel configuration."""
    enabled: bool = False
    bridge_url: str = "ws://localhost:3001"
    allow_from: list[str] = Field(default_factory=list)  # Allowed phone numbers


class TelegramConfig(BaseModel):
    """Telegram channel configuration."""
    enabled: bool = False
    token: str = ""  # Bot token from @BotFather
    allow_from: list[str] = Field(default_factory=list)  # Allowed user IDs or usernames
    proxy: str | None = None  # HTTP/SOCKS5 proxy URL, e.g. "http://127.0.0.1:7890" or "socks5://127.0.0.1:1080"


class FeishuConfig(BaseModel):
    """Feishu/Lark channel configuration using WebSocket long connection."""
    enabled: bool = False
    app_id: str = ""  # App ID from Feishu Open Platform
    app_secret: str = ""  # App Secret from Feishu Open Platform
    encrypt_key: str = ""  # Encrypt Key for event subscription (optional)
    verification_token: str = ""  # Verification Token for event subscription (optional)
    allow_from: list[str] = Field(default_factory=list)  # Allowed user open_ids


class DiscordConfig(BaseModel):
    """Discord channel configuration."""
    enabled: bool = False
    token: str = ""  # Bot token from Discord Developer Portal
    allow_from: list[str] = Field(default_factory=list)  # Allowed user IDs
    gateway_url: str = "wss://gateway.discord.gg/?v=10&encoding=json"
    intents: int = 37377  # GUILDS + GUILD_MESSAGES + DIRECT_MESSAGES + MESSAGE_CONTENT


class ChannelsConfig(BaseModel):
    """Configuration for chat channels."""
    whatsapp: WhatsAppConfig = Field(default_factory=WhatsAppConfig)
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    discord: DiscordConfig = Field(default_factory=DiscordConfig)
    feishu: FeishuConfig = Field(default_factory=FeishuConfig)


class AgentDefaults(BaseModel):
    """Default agent configuration."""
    workspace: str = "~/.nanobot/workspace"
    model: str = "anthropic/claude-opus-4-5"
    max_tokens: int = 8192
    temperature: float = 0.7
    max_tool_iterations: int = 20


class AgentsConfig(BaseModel):
    """Agent configuration."""
    defaults: AgentDefaults = Field(default_factory=AgentDefaults)


class ProviderConfig(BaseModel):
    """LLM provider configuration."""
    api_key: str = ""
    api_base: str | None = None
    extra_headers: dict[str, str] | None = None  # Custom headers (e.g. APP-Code for AiHubMix)
    
    @property
    def is_configured(self) -> bool:
        """Check if this provider is properly configured."""
        return bool(self.api_key)


class ProvidersConfig(BaseModel):
    """Configuration for LLM providers."""
    anthropic: ProviderConfig = Field(default_factory=ProviderConfig)
    openai: ProviderConfig = Field(default_factory=ProviderConfig)
    openrouter: ProviderConfig = Field(default_factory=ProviderConfig)
    deepseek: ProviderConfig = Field(default_factory=ProviderConfig)
    groq: ProviderConfig = Field(default_factory=ProviderConfig)
    zhipu: ProviderConfig = Field(default_factory=ProviderConfig)
    dashscope: ProviderConfig = Field(default_factory=ProviderConfig)  # 阿里云通义千问
    vllm: ProviderConfig = Field(default_factory=ProviderConfig)
    gemini: ProviderConfig = Field(default_factory=ProviderConfig)
    moonshot: ProviderConfig = Field(default_factory=ProviderConfig)
    aihubmix: ProviderConfig = Field(default_factory=ProviderConfig)  # AiHubMix API gateway


class GatewayConfig(BaseModel):
    """Gateway/server configuration."""
    host: str = "0.0.0.0"
    port: int = 18790


class WebSearchConfig(BaseModel):
    """Web search tool configuration."""
    api_key: str = ""  # Brave Search API key
    max_results: int = 5


class WebToolsConfig(BaseModel):
    """Web tools configuration."""
    search: WebSearchConfig = Field(default_factory=WebSearchConfig)


class ExecToolConfig(BaseModel):
    """Shell exec tool configuration."""
    timeout: int = 60


class ToolsConfig(BaseModel):
    """Tools configuration."""
    web: WebToolsConfig = Field(default_factory=WebToolsConfig)
    exec: ExecToolConfig = Field(default_factory=ExecToolConfig)
    restrict_to_workspace: bool = False  # If true, restrict all tool access to workspace directory


class Config(BaseSettings):
    """Root configuration for nanobot."""
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    channels: ChannelsConfig = Field(default_factory=ChannelsConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    
    @property
    def workspace_path(self) -> Path:
        """Get expanded workspace path."""
        return Path(self.agents.defaults.workspace).expanduser()
    
    # Default base URLs for API gateways
    _GATEWAY_DEFAULTS = {
        "openrouter": "https://openrouter.ai/api/v1", 
        "aihubmix": "https://aihubmix.com/v1"
    }
    
    # Provider detection mapping with explicit priorities
    _PROVIDER_PRIORITIES = [
        # Gateways first (can serve multiple models)
        ("openrouter", ["openrouter"]),
        ("aihubmix", ["aihubmix"]),
        # Specific providers
        ("anthropic", ["anthropic", "claude"]),
        ("openai", ["openai", "gpt"]),
        ("gemini", ["gemini"]),
        ("zhipu", ["zhipu", "glm", "zai"]),
        ("dashscope", ["dashscope", "qwen"]),
        ("moonshot", ["moonshot", "kimi"]),
        ("deepseek", ["deepseek"]),
        ("groq", ["groq"]),
        ("vllm", ["vllm"]),
    ]

    def get_provider(self, model: str | None = None) -> ProviderConfig | None:
        """
        Get matched provider config based on model name and configuration.
        
        Returns the most appropriate provider based on:
        1. Explicit provider prefix in model (e.g., "openrouter/claude-3.5-sonnet")
        2. Model keywords matching configured providers
        3. Fallback to first available gateway or provider
        
        Args:
            model: Model identifier to match against providers
            
        Returns:
            Configured ProviderConfig or None if no provider is configured
        """
        model_name = (model or self.agents.defaults.model).lower()
        
        # Check for explicit provider prefix (e.g., "openrouter/claude-3.5-sonnet")
        if "/" in model_name:
            provider_prefix = model_name.split("/")[0]
            provider_attr = provider_prefix.replace("-", "_")  # Handle hyphens in provider names
            if hasattr(self.providers, provider_attr):
                provider_config = getattr(self.providers, provider_attr)
                if provider_config.is_configured:
                    return provider_config
        
        # Check model keywords against provider priorities
        for provider_name, keywords in self._PROVIDER_PRIORITIES:
            if any(keyword in model_name for keyword in keywords):
                provider_config = getattr(self.providers, provider_name)
                if provider_config.is_configured:
                    return provider_config
        
        # Fallback: return first configured provider
        all_providers = [
            self.providers.openrouter,
            self.providers.aihubmix,
            self.providers.anthropic,
            self.providers.openai,
            self.providers.deepseek,
            self.providers.gemini,
            self.providers.zhipu,
            self.providers.dashscope,
            self.providers.moonshot,
            self.providers.vllm,
            self.providers.groq,
        ]
        
        for provider in all_providers:
            if provider.is_configured:
                return provider
                
        return None

    def get_api_key(self, model: str | None = None) -> str | None:
        """Get API key for the given model. Falls back to first available key."""
        p = self.get_provider(model)
        return p.api_key if p else None
    
    def get_api_base(self, model: str | None = None) -> str | None:
        """Get API base URL for the given model. Applies default URLs for known gateways."""
        p = self.get_provider(model)
        if p and p.api_base:
            return p.api_base
        # Default URLs for known gateways (openrouter, aihubmix)
        for name, url in self._GATEWAY_DEFAULTS.items():
            if p == getattr(self.providers, name):
                return url
        return None
    
    class Config:
        env_prefix = "NANOBOT_"
        env_nested_delimiter = "__"