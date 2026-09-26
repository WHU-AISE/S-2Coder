"""
LLM Client Manager - Unified interface for calling different LLM models
LLM
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = ""
import json
import time
import threading
from typing import Optional, Dict, Any, Union
from enum import Enum
from langchain_openai import ChatOpenAI
from openai import OpenAI
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage

LOCAL_QWENCODER_7B = ""
LOCAL_DEEPSEEK_CODER_67B = ""
DEFAULT_VLLM_TP_SIZE = 1
DEFAULT_VLLM_MAX_MODEL_LEN = 16384
try:
    DEFAULT_VLLM_GPU_MEM_UTIL = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.90"))
except ValueError:
    DEFAULT_VLLM_GPU_MEM_UTIL = 0.90
DEEPSEEK_SAFE_MAX_MODEL_LEN = 16096


class LLMProvider(str, Enum):
    """LLM"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AZURE = "azure"
    DEEPSEEK = "deepseek"
    QWEN = "qwen"
    GEMINI = "gemini"
    DEEPSEEK_CODER_VLLM = "deepseek_coder_vllm"
    QWENCODER_VLLM = "qwencoder_vllm"
    OLLAMA = "ollama"
    CUSTOM = "custom"


class OpenAIModels(str, Enum):
    """OpenAI """
    GPT_4 = "gpt-4"
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"


class AnthropicModels(str, Enum):
    """Anthropic """
    CLAUDE_3_OPUS = "claude-3-opus-20240229"
    CLAUDE_3_SONNET = "claude-3-sonnet-20240229"
    CLAUDE_3_HAIKU = "claude-3-haiku-20240307"


class GoogleModels(str, Enum):
    """Google """
    GEMINI_PRO = "gemini-pro"
    GEMINI_PRO_VISION = "gemini-pro-vision"


class DeepSeekModels(str, Enum):
    """DeepSeek """
    DEEPSEEK_CHAT = "deepseek-chat"
    DEEPSEEK_CODE = "deepseek-coder"


class QwenModels(str, Enum):
    """Qwen """
    QWEN_PLUS = "qwen-plus"
    QWEN_TURBO = "qwen-turbo"
    QWEN_MAX = "qwen-max"

class GeminiModels(str, Enum):
    """Gemini （OpenAI）"""
    GEMINI_2_5_FLASH = "gemini-2.5-flash"
    GEMINI_1_5_PRO = "gemini-1.5-pro"


class OpenSourceCodeModels(str, Enum):
    """（vLLM）"""
    QWENCODER_2_5_7B = "qwencoder-2.5-7b"
    DEEPSEEK_CODER_6_7B = "deepseek-coder-6.7b"


class OllamaModels(str, Enum):
    """Ollama （）"""
    LLAMA2 = "llama2"
    MISTRAL = "mistral"
    NEURAL_CHAT = "neural-chat"
    DOLPHIN_MIX = "dolphin-mixtral"


class LLMConfig(BaseModel):
    """LLM"""
    provider: LLMProvider = Field(default=LLMProvider.OPENAI, description="LLM")
    model: str = Field(description="")
    temperature: float = Field(default=0.7, description="")
    max_tokens: Optional[int] = Field(default=None, description="token")
    top_p: float = Field(default=1.0, description="top_p")
    api_key: Optional[str] = Field(default=None, description="API")
    api_base: Optional[str] = Field(default=None, description="APIURL")
    request_timeout: int = Field(default=60, description="（）")


class VLLMChatClient:
    """vLLM ： LangChain Chat  invoke """

    _MODEL_ALIAS = {
        OpenSourceCodeModels.QWENCODER_2_5_7B.value: LOCAL_QWENCODER_7B,
        OpenSourceCodeModels.DEEPSEEK_CODER_6_7B.value: LOCAL_DEEPSEEK_CODER_67B,
    }
    _MODEL_NUM_HEADS = {
        OpenSourceCodeModels.QWENCODER_2_5_7B.value: 28,
        OpenSourceCodeModels.DEEPSEEK_CODER_6_7B.value: 32,
        LOCAL_QWENCODER_7B: 28,
        LOCAL_DEEPSEEK_CODER_67B: 32,
    }

    @staticmethod
    def _infer_tensor_parallel_size(default: int = DEFAULT_VLLM_TP_SIZE) -> int:
        # Keep a tiny fallback parser, but default is now hardcoded to 3.
        visible = os.getenv("CUDA_VISIBLE_DEVICES", "").strip()
        if visible:
            ids = [x.strip() for x in visible.split(",") if x.strip()]
            if ids:
                return max(1, len(ids))
        return default

    @classmethod
    def _adjust_tp_for_model(cls, model_name: str, tp_size: int) -> int:
        num_heads = cls._MODEL_NUM_HEADS.get(model_name)
        if not num_heads or tp_size <= 1:
            return max(1, tp_size)
        if num_heads % tp_size == 0:
            return tp_size
        for cand in range(tp_size - 1, 0, -1):
            if num_heads % cand == 0:
                print(
                    f"[vLLM] tensor_parallel_size={tp_size} incompatible with heads={num_heads}, "
                    f"fallback to {cand}."
                )
                return cand
        return 1

    @staticmethod
    def _missing_index_files(model_dir: str, index_filename: str) -> list:
        index_path = os.path.join(model_dir, index_filename)
        if not os.path.exists(index_path):
            return []
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                index_obj = json.load(f)
            files = sorted(set(index_obj.get("weight_map", {}).values()))
            return [name for name in files if not os.path.exists(os.path.join(model_dir, name))]
        except Exception:
            return []

    @staticmethod
    def _safe_max_model_len_for_model(model_name: str, requested_len: int) -> int:
        if not model_name:
            return requested_len
        name = str(model_name).lower()
        is_deepseek_coder = (
            "deepseek-coder-6.7b" in name
            or name == OpenSourceCodeModels.DEEPSEEK_CODER_6_7B.value
            or name == LOCAL_DEEPSEEK_CODER_67B.lower()
        )
        if is_deepseek_coder and requested_len > DEEPSEEK_SAFE_MAX_MODEL_LEN:
            print(
                f"[vLLM] Clamp max_model_len from {requested_len} to {DEEPSEEK_SAFE_MAX_MODEL_LEN} "
                "for DeepSeek-Coder-6.7B to avoid KV cache OOM."
            )
            return DEEPSEEK_SAFE_MAX_MODEL_LEN
        return requested_len

    @staticmethod
    def _is_low_memory_startup_error(exc: Exception) -> bool:
        msg = str(exc)
        return "Free memory on device" in msg and "gpu memory utilization" in msg.lower()

    @staticmethod
    def _fallback_gpu_mem_utils(initial_value: float) -> list:
        # Descend through practical defaults when startup fails on VRAM pressure.
        candidates = [0.85, 0.80, 0.75, 0.70, 0.60, 0.50]
        return [v for v in candidates if v < float(initial_value)]

    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 1.0,
        **kwargs: Any,
    ):
        try:
            from vllm import LLM
        except ImportError as exc:
            raise ImportError("Please install vllm first: pip install vllm") from exc

        self.model = self._MODEL_ALIAS.get(model, model)
        self.default_temperature = temperature
        self.default_max_tokens = max_tokens or 512
        self.default_top_p = top_p
        self.default_stop = kwargs.pop("stop", None)

        tensor_parallel_size = int(kwargs.pop("tensor_parallel_size", self._infer_tensor_parallel_size()))
        tensor_parallel_size = self._adjust_tp_for_model(self.model, tensor_parallel_size)
        gpu_memory_utilization = float(kwargs.pop("gpu_memory_utilization", DEFAULT_VLLM_GPU_MEM_UTIL))
        max_model_len = kwargs.pop("max_model_len", DEFAULT_VLLM_MAX_MODEL_LEN)
        max_model_len = self._safe_max_model_len_for_model(self.model, int(max_model_len))
        trust_remote_code = bool(kwargs.pop("trust_remote_code", True))
        enforce_eager = bool(kwargs.pop("enforce_eager", True))
        dtype = kwargs.pop("dtype", os.getenv("VLLM_DTYPE", "auto"))

        engine_kwargs = {
            "model": self.model,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "trust_remote_code": trust_remote_code,
            "enforce_eager": enforce_eager,
            "dtype": dtype,
        }
        # If safetensors index exists but shard files are missing, force loading from
        # PyTorch .bin checkpoints to avoid partial-weight initialization failures.
        if isinstance(self.model, str) and os.path.isdir(self.model):
            missing_safetensor_shards = self._missing_index_files(self.model, "model.safetensors.index.json")
            if missing_safetensor_shards:
                missing_bin_shards = self._missing_index_files(self.model, "pytorch_model.bin.index.json")
                if not missing_bin_shards:
                    engine_kwargs["load_format"] = "pt"
                    print(
                        "[vLLM] Missing safetensors shards detected, falling back to load_format='pt': "
                        + ", ".join(missing_safetensor_shards)
                    )
        if max_model_len is not None:
            engine_kwargs["max_model_len"] = int(max_model_len)
        engine_kwargs.update(kwargs)

        try:
            self.llm = LLM(**engine_kwargs)
        except ValueError as exc:
            if not self._is_low_memory_startup_error(exc):
                raise
            last_exc: Exception = exc
            initial_util = float(engine_kwargs.get("gpu_memory_utilization", DEFAULT_VLLM_GPU_MEM_UTIL))
            for fallback_util in self._fallback_gpu_mem_utils(initial_util):
                try:
                    engine_kwargs["gpu_memory_utilization"] = fallback_util
                    print(
                        f"[vLLM] Startup failed with gpu_memory_utilization={initial_util:.2f}, "
                        f"retrying with {fallback_util:.2f}."
                    )
                    self.llm = LLM(**engine_kwargs)
                    return
                except ValueError as retry_exc:
                    last_exc = retry_exc
                    if not self._is_low_memory_startup_error(retry_exc):
                        raise
            raise last_exc

    @staticmethod
    def _to_prompt_text(prompt: Union[str, Any]) -> str:
        if isinstance(prompt, str):
            return prompt
        if isinstance(prompt, list):
            parts = []
            for item in prompt:
                if isinstance(item, dict):
                    role = str(item.get("role", "")).strip()
                    content = str(item.get("content", "")).strip()
                    if role:
                        parts.append(f"{role}: {content}")
                    elif content:
                        parts.append(content)
                else:
                    parts.append(str(item))
            return "\n".join(p for p in parts if p)
        return str(prompt)

    def invoke(self, prompt: Union[str, Any], **kwargs: Any) -> AIMessage:
        from vllm import SamplingParams

        prompt_text = self._to_prompt_text(prompt)
        temperature = float(kwargs.pop("temperature", self.default_temperature))
        top_p = float(kwargs.pop("top_p", self.default_top_p))
        max_tokens = int(kwargs.pop("max_tokens", self.default_max_tokens))
        stop = kwargs.pop("stop", self.default_stop)

        sampling_kwargs: Dict[str, Any] = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        if stop is not None:
            sampling_kwargs["stop"] = stop

        sampling_params = SamplingParams(**sampling_kwargs)
        outputs = None
        # Instruct/chat models are more stable when using chat template.
        try:
            messages = [{"role": "user", "content": prompt_text}]
            outputs = self.llm.chat(messages, sampling_params=sampling_params, use_tqdm=False)
        except Exception:
            outputs = self.llm.generate([prompt_text], sampling_params=sampling_params, use_tqdm=False)

        text = ""
        if outputs and outputs[0].outputs:
            text = outputs[0].outputs[0].text
        return AIMessage(content=text)


class LLMClientFactory:
    """LLM"""
    
    def __init__(self):
        """"""
        self.clients_cache: Dict[str, Any] = {}
        self._cache_lock = threading.Lock()
    
    @staticmethod
    def create_openai_client(config: LLMConfig) -> ChatOpenAI:
        """
        OpenAI
        
        Args:
            config: LLM
            
        Returns:
            ChatOpenAI
        """
        api_key = config.api_key or os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OpenAI API key not found in config or environment")
        
        api_base = os.getenv("LLM_BASE_URL", "")
        
        kwargs = {
            "model_name": config.model,
            "api_key": api_key,
            "temperature": config.temperature,
            "request_timeout": config.request_timeout,
        }
        
        if config.max_tokens:
            kwargs["max_tokens"] = config.max_tokens
        
        #  base_url  api_base（ langchain_openai）
        if api_base:
            kwargs["base_url"] = api_base
        
        if config.top_p < 1.0:
            kwargs["top_p"] = config.top_p
        
        return ChatOpenAI(**kwargs)
    
    @staticmethod
    def create_anthropic_client(config: LLMConfig) -> Any:
        """
        Anthropic
        
        Args:
            config: LLM
            
        Returns:
            Anthropic
        """
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError("Please install langchain-anthropic: pip install langchain-anthropic")
        
        api_key = config.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key not found in config or environment")
        
        kwargs = {
            "model": config.model,
            "api_key": api_key,
            "temperature": config.temperature,
        }
        
        if config.max_tokens:
            kwargs["max_tokens"] = config.max_tokens
        
        return ChatAnthropic(**kwargs)
    
    @staticmethod
    def create_google_client(config: LLMConfig) -> Any:
        """
        Google
        
        Args:
            config: LLM
            
        Returns:
            Google Generative AI
        """
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError("Please install langchain-google-genai: pip install langchain-google-genai")
        
        api_key = config.api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Google API key not found in config or environment")
        
        kwargs = {
            "model": config.model,
            "google_api_key": api_key,
            "temperature": config.temperature,
            "convert_system_message_to_human": True,
        }
        
        if config.max_tokens:
            kwargs["max_output_tokens"] = config.max_tokens
        
        return ChatGoogleGenerativeAI(**kwargs)
    
    @staticmethod
    def create_deepseek_client(config: LLMConfig) -> ChatOpenAI:
        """
        DeepSeek（OpenAIAPI）
        
        Args:
            config: LLM
            
        Returns:
            DeepSeek ChatOpenAI
        """
        api_key = config.api_key or os.getenv("DEEPSEEK_API_KEY", "")
        base_url = os.getenv("DEEPSEEK_BASE_URL", "")
        
        kwargs = {
            "model_name": "deepseek-chat",
            "api_key": api_key,
            "base_url": base_url,
            "temperature": config.temperature,
            "request_timeout": config.request_timeout,
        }
        
        if config.max_tokens:
            kwargs["max_tokens"] = config.max_tokens
        
        if config.top_p < 1.0:
            kwargs["top_p"] = config.top_p
        
        return ChatOpenAI(**kwargs)
    
    @staticmethod
    def create_qwen_client(config: LLMConfig) -> ChatOpenAI:
        """
        Qwen（OpenAIAPI）

        Args:
            config: LLM

        Returns:
            Qwen ChatOpenAI
        """
        api_key = config.api_key or os.getenv("QWEN_API_KEY", "")
        api_base = os.getenv("QWEN_BASE_URL", "")

        kwargs = {
            "model_name": config.model,
            "api_key": api_key,
            "base_url": api_base,
            "temperature": config.temperature,
            "request_timeout": config.request_timeout,
        }

        if config.max_tokens:
            kwargs["max_tokens"] = config.max_tokens

        if config.top_p < 1.0:
            kwargs["top_p"] = config.top_p

        return ChatOpenAI(**kwargs)

    @staticmethod
    def create_gemini_client(config: LLMConfig) -> ChatOpenAI:
        """
        Gemini（OpenAIAPI）

        Args:
            config: LLM

        Returns:
            Gemini ChatOpenAI
        """
        api_key = config.api_key or os.getenv("GEMINI_API_KEY", "")
        api_base = os.getenv("GEMINI_BASE_URL", "")
        # Force all Gemini requests to use the no-thinking variant.
        model_name = "gemini-2.5-flash-nothinking"

        kwargs = {
            "model_name": model_name,
            "api_key": api_key,
            "base_url": api_base,
            "temperature": 0,
            "request_timeout": config.request_timeout,
            "max_tokens": 1024
        }

        return ChatOpenAI(**kwargs)

    @staticmethod
    def create_qwencoder_vllm_client(config: LLMConfig) -> VLLMChatClient:
        """QwenCoder（vLLM）"""
        return VLLMChatClient(
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
        )

    @staticmethod
    def create_deepseek_coder_vllm_client(config: LLMConfig) -> VLLMChatClient:
        """DeepSeekCoder（vLLM）"""
        return VLLMChatClient(
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
        )
    
    
    @staticmethod
    def create_ollama_client(config: LLMConfig) -> Any:
        """
        Ollama（LLM）
        
        Args:
            config: LLM
            
        Returns:
            Ollama ChatOllama
        """
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            raise ImportError("Please install langchain-ollama: pip install langchain-ollama")
        
        # : config.api_base > OLLAMA_API_BASE > LLM_BASE_URL > 
        api_base = (config.api_base or 
                    os.getenv("OLLAMA_API_BASE") or 
                    os.getenv("LLM_BASE_URL") or 
                    "http://localhost:11434")
        
        kwargs = {
            "model": config.model,
            "base_url": api_base,
            "temperature": config.temperature,
        }
        
        if config.max_tokens:
            kwargs["num_predict"] = config.max_tokens
        
        if config.top_p < 1.0:
            kwargs["top_p"] = config.top_p
        
        return ChatOllama(**kwargs)
    
    @staticmethod
    def create_custom_client(config: LLMConfig) -> ChatOpenAI:
        """
        （OpenAIAPI）
        
        Args:
            config: LLM
            
        Returns:
            OpenAI
        """
        api_key = config.api_key or os.getenv("CUSTOM_API_KEY")
        if not api_key:
            raise ValueError("Custom API key not found in config or environment")
        
        # : config.api_base > CUSTOM_API_BASE > LLM_BASE_URL
        api_base = config.api_base or os.getenv("CUSTOM_API_BASE") or os.getenv("LLM_BASE_URL")
        if not api_base:
            raise ValueError("Custom API base not found in config, CUSTOM_API_BASE or LLM_BASE_URL environment variable")
        
        kwargs = {
            "model_name": config.model,
            "api_key": api_key,
            "base_url": api_base,
            "temperature": config.temperature,
            "request_timeout": config.request_timeout,
        }
        
        if config.max_tokens:
            kwargs["max_tokens"] = config.max_tokens
        
        if config.top_p < 1.0:
            kwargs["top_p"] = config.top_p
        
        return ChatOpenAI(**kwargs)
    
    def create_client(self, config: LLMConfig) -> Any:
        """
        LLM
        
        Args:
            config: LLM
            
        Returns:
            LLM
        """
        # key
        cache_key = f"{config.provider}:{config.model}:{config.temperature}"
        
        with self._cache_lock:
            # ，
            if cache_key in self.clients_cache:
                return self.clients_cache[cache_key]

            # 
            if config.provider == LLMProvider.OPENAI:
                client = self.create_openai_client(config)
            elif config.provider == LLMProvider.ANTHROPIC:
                client = self.create_anthropic_client(config)
            elif config.provider == LLMProvider.GOOGLE:
                client = self.create_google_client(config)
            elif config.provider == LLMProvider.DEEPSEEK:
                client = self.create_deepseek_client(config)
            elif config.provider == LLMProvider.QWEN:
                client = self.create_qwen_client(config)
            elif config.provider == LLMProvider.GEMINI:
                client = self.create_gemini_client(config)
            elif config.provider == LLMProvider.QWENCODER_VLLM:
                client = self.create_qwencoder_vllm_client(config)
            elif config.provider == LLMProvider.DEEPSEEK_CODER_VLLM:
                client = self.create_deepseek_coder_vllm_client(config)
            elif config.provider == LLMProvider.OLLAMA:
                client = self.create_ollama_client(config)
            elif config.provider == LLMProvider.CUSTOM:
                client = self.create_custom_client(config)
            else:
                raise ValueError(f"Unsupported provider: {config.provider}")

            # 
            self.clients_cache[cache_key] = client
            return client
    
    def clear_cache(self):
        """"""
        self.clients_cache.clear()


class LLMClientManager:
    """LLM - """
    
    def __init__(self):
        """"""
        self.factory = LLMClientFactory()
        self.default_config = self._load_default_config()
    
    @staticmethod
    def _normalize_model_name(model: Optional[str]) -> Optional[str]:
        """Normalize non-standard model aliases to valid model ids."""
        if not model:
            return model

        alias_map = {
            "gpt-4o-mini-global": "gpt-4o-mini",
            "gpt-4o-mini-data": "gpt-4o-mini",
            "gpt-4o-global": "gpt-4o",
            "gpt-4o-data": "gpt-4o",
        }
        if model in alias_map:
            normalized = alias_map[model]
            print(f"[LLM] Normalize model alias: {model} -> {normalized}")
            return normalized

        # Generic fallback for custom suffixes like `-global` / `-data`.
        for suffix in ("-global", "-data"):
            if model.endswith(suffix):
                base_model = model[: -len(suffix)]
                if base_model:
                    print(f"[LLM] Normalize model suffix: {model} -> {base_model}")
                    return base_model

        return model

    @staticmethod
    def _load_default_config() -> LLMConfig:
        """"""
        provider_env = os.getenv("LLM_PROVIDER", "openai").lower()
        
        try:
            provider = LLMProvider(provider_env)
        except ValueError:
            provider = LLMProvider.OPENAI
        
        model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        model = LLMClientManager._normalize_model_name(model)
        temperature = float(os.getenv("LLM_TEMPERATURE", "0"))
        max_tokens = os.getenv("LLM_MAX_TOKENS")
        
        return LLMConfig(
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=int(max_tokens) if max_tokens else None,
        )
    
    def get_client(
        self,
        provider: Optional[LLMProvider] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Any:
        """
        LLM
        
        Args:
            provider: LLM
            model: 
            temperature: 
            max_tokens: token
            **kwargs: 
            
        Returns:
            LLM
        """
        config = LLMConfig(
            provider=provider or self.default_config.provider,
            model=self._normalize_model_name(model or self.default_config.model),
            temperature=temperature if temperature is not None else self.default_config.temperature,
            max_tokens=max_tokens or self.default_config.max_tokens,
            **kwargs
        )
        return self.factory.create_client(config)
    
    def get_openai_client(self, model: str = "gpt-4o-mini", temperature: float = 0.0, **kwargs) -> ChatOpenAI:
        """
        OpenAI - 
        
        Args:
            model: 
            temperature: 
            **kwargs: 
            
        Returns:
            ChatOpenAI
        """
        return self.get_client(
            provider=LLMProvider.OPENAI,
            model=model,
            temperature=temperature,
            **kwargs
        )
    
    def get_anthropic_client(self, model: str = "claude-3-sonnet-20240229", temperature: float = 0.7, **kwargs) -> Any:
        """
        Anthropic - 
        
        Args:
            model: 
            temperature: 
            **kwargs: 
            
        Returns:
            Anthropic
        """
        return self.get_client(
            provider=LLMProvider.ANTHROPIC,
            model=model,
            temperature=temperature,
            **kwargs
        )
    
    def get_google_client(self, model: str = "gemini-pro", temperature: float = 0.7, **kwargs) -> Any:
        """
        Google - 
        
        Args:
            model: 
            temperature: 
            **kwargs: 
            
        Returns:
            Google
        """
        return self.get_client(
            provider=LLMProvider.GOOGLE,
            model=model,
            temperature=temperature,
            **kwargs
        )
    
    def get_deepseek_client(self, model: str = "deepseek-chat", temperature: float = 0.7, **kwargs) -> ChatOpenAI:
        """
        DeepSeek - 

        Args:
            model: 
            temperature: 
            **kwargs: 

        Returns:
            DeepSeek
        """
        return self.get_client(
            provider=LLMProvider.DEEPSEEK,
            model=model,
            temperature=temperature,
            **kwargs
        )

    def get_qwen_client(self, model: str = "qwen-plus", temperature: float = 0.7, **kwargs) -> ChatOpenAI:
        """
        Qwen - 

        Args:
            model: 
            temperature: 
            **kwargs: 

        Returns:
            Qwen
        """
        return self.get_client(
            provider=LLMProvider.QWEN,
            model=model,
            temperature=temperature,
            quantization="fp8",
            **kwargs
        )

    def get_gemini_client(self, model: str = GeminiModels.GEMINI_2_5_FLASH.value, temperature: float = 0.7, **kwargs) -> ChatOpenAI:
        """
        Gemini - 

        Args:
            model: 
            temperature: 
            **kwargs: 

        Returns:
            Gemini
        """
        return self.get_client(
            provider=LLMProvider.GEMINI,
            model=model,
            temperature=temperature,
            **kwargs
        )

    def get_qwencoder_vllm_client(
        self,
        model: str = OpenSourceCodeModels.QWENCODER_2_5_7B.value,
        temperature: float = 0.0,
        **kwargs,
    ) -> VLLMChatClient:
        """QwenCoder（vLLM）"""
        return self.get_client(
            provider=LLMProvider.QWENCODER_VLLM,
            model=model,
            temperature=temperature,
            quantization="fp8",
            **kwargs,
        )

    def get_deepseekcoder_vllm_client(
        self,
        model: str = OpenSourceCodeModels.DEEPSEEK_CODER_6_7B.value,
        temperature: float = 0.0,
        **kwargs,
    ) -> VLLMChatClient:
        """DeepSeekCoder（vLLM）"""
        return self.get_client(
            provider=LLMProvider.DEEPSEEK_CODER_VLLM,
            model=model,
            temperature=temperature,
            **kwargs,
        )

    def get_ollama_client(self, model: str = "llama2", temperature: float = 0.7, **kwargs) -> Any:
        """
        Ollama - （LLM）
        
        Args:
            model: 
            temperature: 
            **kwargs: 
            
        Returns:
            Ollama
        """
        return self.get_client(
            provider=LLMProvider.OLLAMA,
            model=model,
            temperature=temperature,
            **kwargs
        )
    
    def get_custom_client(self, model: str = "custom", temperature: float = 0.7, **kwargs) -> ChatOpenAI:
        """
         - （OpenAIAPI）
        
        Args:
            model: 
            temperature: 
            **kwargs: 
            
        Returns:
            
        """
        return self.get_client(
            provider=LLMProvider.CUSTOM,
            model=model,
            temperature=temperature,
            **kwargs
        )
    
    @staticmethod
    def list_openai_models() -> list:
        """OpenAI"""
        return [model.value for model in OpenAIModels]
    
    @staticmethod
    def list_anthropic_models() -> list:
        """Anthropic"""
        return [model.value for model in AnthropicModels]
    
    @staticmethod
    def list_google_models() -> list:
        """Google"""
        return [model.value for model in GoogleModels]
    
    @staticmethod
    def list_deepseek_models() -> list:
        """DeepSeek"""
        return [model.value for model in DeepSeekModels]
    
    
    @staticmethod
    def list_qwen_models() -> list:
        """Qwen"""
        return [model.value for model in QwenModels]

    @staticmethod
    def list_gemini_models() -> list:
        """Gemini"""
        return [model.value for model in GeminiModels]

    @staticmethod
    def list_ollama_models() -> list:
        """Ollama"""
        return [model.value for model in OllamaModels]

    @staticmethod
    def list_open_source_code_models() -> list:
        """（vLLM）"""
        return [model.value for model in OpenSourceCodeModels]


# 
_global_manager: Optional[LLMClientManager] = None


def get_global_llm_manager() -> LLMClientManager:
    """LLM"""
    global _global_manager
    if _global_manager is None:
        _global_manager = LLMClientManager()
    return _global_manager


def get_default_llm_client() -> Any:
    """LLM"""
    manager = get_global_llm_manager()
    return manager.get_client()


# 
def get_openai_client(model: str = "gpt-4o-mini", temperature: float = 0.0, **kwargs) -> ChatOpenAI:
    """OpenAI"""
    manager = get_global_llm_manager()
    return manager.get_openai_client(model=model, temperature=temperature, **kwargs)


def get_anthropic_client(model: str = "claude-3-sonnet-20240229", temperature: float = 0.7) -> Any:
    """Anthropic"""
    manager = get_global_llm_manager()
    return manager.get_anthropic_client(model=model, temperature=temperature)


def get_google_client(model: str = "gemini-pro", temperature: float = 0.7) -> Any:
    """Google"""
    manager = get_global_llm_manager()
    return manager.get_google_client(model=model, temperature=temperature)


def get_deepseek_client(model: str = "deepseek-chat", temperature: float = 0.7) -> ChatOpenAI:
    """DeepSeek"""
    manager = get_global_llm_manager()
    return manager.get_deepseek_client(model=model, temperature=temperature)


def get_qwen_client(model: str = "qwen-plus", temperature: float = 0.7) -> ChatOpenAI:
    """Qwen"""
    manager = get_global_llm_manager()
    return manager.get_qwen_client(model=model, temperature=temperature)


def get_gemini_client(model: str = GeminiModels.GEMINI_2_5_FLASH.value, temperature: float = 0.7) -> ChatOpenAI:
    """Gemini"""
    manager = get_global_llm_manager()
    return manager.get_gemini_client(model=model, temperature=temperature)


def get_qwencoder_vllm_client(
    model: str = OpenSourceCodeModels.QWENCODER_2_5_7B.value,
    temperature: float = 0.0,
    **kwargs,
) -> VLLMChatClient:
    """QwenCoder（vLLM）"""
    manager = get_global_llm_manager()
    return manager.get_qwencoder_vllm_client(model=model, temperature=temperature, **kwargs)


def get_deepseekcoder_vllm_client(
    model: str = OpenSourceCodeModels.DEEPSEEK_CODER_6_7B.value,
    temperature: float = 0.0,
    **kwargs,
) -> VLLMChatClient:
    """DeepSeekCoder（vLLM）"""
    manager = get_global_llm_manager()
    return manager.get_deepseekcoder_vllm_client(model=model, temperature=temperature, **kwargs)


def get_ollama_client(model: str = "llama2", temperature: float = 0.7) -> Any:
    """Ollama（LLM）"""
    manager = get_global_llm_manager()
    return manager.get_ollama_client(model=model, temperature=temperature)


def get_custom_client(model: str = "custom", temperature: float = 0.7, **kwargs) -> ChatOpenAI:
    """（OpenAIAPI）"""
    manager = get_global_llm_manager()
    return manager.get_custom_client(model=model, temperature=temperature, **kwargs)


def get_embedding(text: str, model: str = "text-embedding-3-small", timeout: int = 30) -> list:
    """
     embedding 
    
    Args:
        text: 
        model: embedding  (: text-embedding-3-small)
        timeout:  (: 30)
        
    Returns:
        embedding 
    """
    try:
        api_key = os.getenv("OPENAI_API_KEY", "")
        base_url = os.getenv("LLM_BASE_URL", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        
        client = OpenAI(api_key=api_key, base_url=base_url)

        # 
        start_time = time.time()
        
        response = client.embeddings.create(
            input=text,
            model=model,
            timeout=timeout  # 
        )
        
        # 
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout:
            raise TimeoutError(f":  {timeout} ")
        
        #  embedding 
        return response.data[0].embedding
        
    except TimeoutError as e:
        print(f"❌  embedding : {e}")
        raise
    
    except Exception as e:
        print(f"❌  embedding : {e}")
        raise
