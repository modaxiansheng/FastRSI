"""LLM provider abstraction module."""

__all__ = ["LLMProvider", "LLMResponse", "LiteLLMProvider", "OpenAICodexProvider", "AzureOpenAIProvider"]


def __getattr__(name: str):
    if name in {"LLMProvider", "LLMResponse"}:
        from PhyAgentOS.providers import base
        return getattr(base, name)
    if name == "LiteLLMProvider":
        from PhyAgentOS.providers.litellm_provider import LiteLLMProvider
        return LiteLLMProvider
    if name == "OpenAICodexProvider":
        from PhyAgentOS.providers.openai_codex_provider import OpenAICodexProvider
        return OpenAICodexProvider
    if name == "AzureOpenAIProvider":
        from PhyAgentOS.providers.azure_openai_provider import AzureOpenAIProvider
        return AzureOpenAIProvider
    raise AttributeError(name)
