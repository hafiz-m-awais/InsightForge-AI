import os
from enum import Enum
from typing import Optional
from langchain_core.language_models.chat_models import BaseChatModel

# Note: Using modern langchain integrations
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

class LLMProvider(str, Enum):
    OPENROUTER = "openrouter"
    GEMINI = "gemini"
    GROQ = "groq"
    OPENAI = "openai"

class LLMRouter:
    """
    Manages multiple LLM providers and allows seamless switching or fallback.
    OpenRouter uses ChatOpenAI underneath with a custom base_url.
    """
    def __init__(self, default_provider: LLMProvider = LLMProvider.OPENROUTER, model_name: Optional[str] = None):
        self.current_provider = default_provider
        self.model_name = model_name
        self.models = {}
        
    def _initialize_provider(self, provider: LLMProvider, model_name: Optional[str] = None) -> BaseChatModel:
        models_list = []
        
        if provider == LLMProvider.OPENROUTER:
            keys = [k.strip() for k in os.environ.get("OPENROUTER_API_KEYS", os.environ.get("OPENROUTER_API_KEY", "")).split(",") if k.strip()]
            model_strs = [m.strip() for m in os.environ.get("OPENROUTER_MODELS", "anthropic/claude-3-haiku").split(",") if m.strip()]
            if model_name: model_strs = [model_name]
            
            for key in keys:
                for mod in model_strs:
                    models_list.append(ChatOpenAI(
                        base_url="https://openrouter.ai/api/v1",
                        api_key=key,
                        model=mod,
                        temperature=0.2
                    ))
                    
        elif provider == LLMProvider.GEMINI:
            keys = [k.strip() for k in os.environ.get("GEMINI_API_KEYS", os.environ.get("GEMINI_API_KEY", "")).split(",") if k.strip()]
            model_strs = [m.strip() for m in os.environ.get("GEMINI_MODELS", "gemini-pro").split(",") if m.strip()]
            if model_name: model_strs = [model_name]
            
            for key in keys:
                for mod in model_strs:
                    models_list.append(ChatGoogleGenerativeAI(
                        model=mod,
                        google_api_key=key,
                        temperature=0.2
                    ))
                    
        elif provider == LLMProvider.GROQ:
            keys = [k.strip() for k in os.environ.get("GROQ_API_KEYS", os.environ.get("GROQ_API_KEY", "")).split(",") if k.strip()]
            model_strs = [m.strip() for m in os.environ.get("GROQ_MODELS", "llama3-70b-versatile").split(",") if m.strip()]
            if model_name: model_strs = [model_name]
            
            for key in keys:
                for mod in model_strs:
                    models_list.append(ChatGroq(
                        model_name=mod,
                        api_key=key,
                        temperature=0.2
                    ))
                    
        elif provider == LLMProvider.OPENAI:
            keys = [k.strip() for k in os.environ.get("OPENAI_API_KEYS", os.environ.get("OPENAI_API_KEY", "")).split(",") if k.strip()]
            model_strs = [m.strip() for m in os.environ.get("OPENAI_MODELS", "gpt-4-turbo-preview").split(",") if m.strip()]
            if model_name: model_strs = [model_name]
            
            for key in keys:
                for mod in model_strs:
                    models_list.append(ChatOpenAI(
                        model=mod,
                        api_key=key,
                        temperature=0.2
                    ))
        else:
            raise ValueError(f"Unsupported LLM Provider: {provider}")

        if not models_list:
            raise ValueError(f"No API keys found for provider {provider}")

        # Make it iteratable by using Langchain fallbacks
        primary_model = models_list[0]
        if len(models_list) > 1:
            return primary_model.with_fallbacks(models_list[1:])
        return primary_model

    def get_model(self, provider: Optional[LLMProvider] = None, model_name: Optional[str] = None) -> BaseChatModel:
        """
        Get the LLM model instance. Defaults to current_provider if none specified.
        """
        target_provider = provider or self.current_provider
        target_model = model_name or self.model_name
        
        # Cache key based on provider and model name
        cache_key = f"{target_provider.value}_{target_model}"
        
        if cache_key not in self.models:
            self.models[cache_key] = self._initialize_provider(target_provider, target_model)
            
        return self.models[cache_key]
        
    def set_default_provider(self, provider: LLMProvider, model_name: Optional[str] = None):
        self.current_provider = provider
        if model_name:
            self.model_name = model_name

# Global instance
llm_manager = LLMRouter()

def get_llm(provider: Optional[LLMProvider] = None, model_name: Optional[str] = None) -> BaseChatModel:
    """Convenience function to get the requested LLM"""
    return llm_manager.get_model(provider, model_name)
