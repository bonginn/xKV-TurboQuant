import logging
from typing import Dict, Optional

try:
    from loguru import logger
except ImportError:  # loguru is optional in offline inference environments.
    logger = logging.getLogger(__name__)

from transformers.cache_utils import Cache, DynamicCache
from .attn_patch.mistral import enable_mistral_xKV_eval
from .attn_patch.llama import enable_llama_xKV_eval
from .attn_patch.qwen import enable_qwen_xKV_eval
from .attn_patch.deepseek_v2 import enable_deepseek_v2_xKV_eval
from .customized_cache import method_to_cache_obj
from .configurations import xKVConfig
from .runtime_audit import create_runtime_audit

def prepare_cache(method: str, config, runtime_audit=None):
    cache_obj: Cache = method_to_cache_obj.get(method, None)
    # HF creates a fresh cache for every generate() call.  Keep deterministic
    # TurboQuant rotations/codebooks alive at model-patch scope so RULER does
    # not repeat a large QR decomposition for every sample.
    turboquant_registry = {}

    def _prepare_cache_for_generation(
        self, generation_config, model_kwargs: Dict, *args, **kwargs
    ) -> bool:
        """
        Prepares the cache for generation (if applicable), given `generate`'s paramaterization. If a cache is
        instantiated, writes it to `model_kwargs`, under the name expected by the model.
        """
        #config.num_layers = self.config.num_hidden_layers
        if cache_obj is None:
            model_kwargs["past_key_values"] = DynamicCache()
        else:
            model_kwargs["past_key_values"] = cache_obj(
                config,
                turboquant_registry=turboquant_registry,
                runtime_audit=runtime_audit,
            )

    return _prepare_cache_for_generation



class KVCompress:
    """
    A helper to patch your model with a FakeLayerMergingCache 
    using a user-provided or YAML-loaded xKVConfig.
    """

    def __init__(self, xKV_config: Optional[xKVConfig] = None, yaml_path: Optional[str] = None):
        """
        If xKV_config is given, use that.
        Else if yaml_path is given, load from that YAML file.
        Otherwise, raise an error or set a default fallback.
        """
        if xKV_config is not None:
            self.config = xKV_config
        elif yaml_path is not None:
            self.config = xKVConfig.from_yaml(yaml_path)
        else:
            raise ValueError("Must provide either xKV_config or yaml_path.")
    
    def __call__(self, model):
        return self.enable_xKV_patch(model)
    
    def enable_xKV_patch(self, model):
        logger.info("Enabling xKV patch for model: {}".format(model.config.architectures))
        if "mistral" in model.config.model_type:
            enable_mistral_xKV_eval(model)
        elif "llama" in model.config.model_type:
            enable_llama_xKV_eval(model)
        elif "qwen" in model.config.model_type:
            enable_qwen_xKV_eval(model)    
        elif "deepseek_v2" in model.config.model_type:
            enable_deepseek_v2_xKV_eval(model)
        else:
            raise ValueError("Model type not supported for xKV patch: {}".format(model.config.model_type))
        
        model.kv_compress_config = self.config
        runtime_audit = create_runtime_audit(
            self.config,
            model_num_layers=model.config.num_hidden_layers,
        )
        if runtime_audit is not None:
            model.xkv_runtime_audit = runtime_audit
        elif hasattr(model, "xkv_runtime_audit"):
            delattr(model, "xkv_runtime_audit")
        prepare_cache_fn = prepare_cache(
            "xKV",
            self.config,
            runtime_audit=runtime_audit,
        )
        model._prepare_cache_for_generation = prepare_cache_fn.__get__(
            model, model.__class__
        )
        
        return model
