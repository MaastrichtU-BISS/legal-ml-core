"""LLM assistance for annotation.

The package holds the capability only. Policy stays with the host, which injects
``FEATURE_GATE`` (who may use the shared key) and ``USAGE_METER`` (how shared-key
tokens are counted). See docs/adr/0002 and 0003 in the BAIGLE repo.
"""

from .access import free_model_for, purge_owner_llm_data
from .models_list import is_free_model
from .metering import DBUsageMeter, UsageMeter

__all__ = [
    "DBUsageMeter",
    "UsageMeter",
    "free_model_for",
    "is_free_model",
    "purge_owner_llm_data",
]
