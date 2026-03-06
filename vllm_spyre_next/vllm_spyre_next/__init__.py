"""vLLM-Spyre Next: Next-generation vLLM integration for IBM Spyre AI accelerator.

This package provides a vLLM platform plugin for Spyre hardware, implementing:
- Phase 1: Core compiler interface (SpyreCompiler)
- Phase 2: Artifact caching with 3-level strategy
- Phase 3 (optional): Graph wrapper for runtime optimization

The integration follows vLLM-Ascend's plugin architecture pattern, enabling
out-of-tree (OOT) integration without modifying vLLM core code.

Entry Point:
    The register() function is called by vLLM's plugin system during
    initialization. It is configured in pyproject.toml:
    
    [project.entry-points."vllm.platform_plugins"]
    spyre_next = "vllm_spyre_next:register"

Usage:
    # vLLM automatically discovers and loads the plugin
    # No explicit import needed in user code
    
    from vllm import LLM
    
    # vLLM detects Spyre hardware and uses SpyreCompiler
    llm = LLM(model="meta-llama/Llama-2-7b-hf")
    outputs = llm.generate("Hello, my name is")
"""

from vllm_spyre_next.platform import register

__all__ = ["register"]

# Version information
__version__ = "0.1.0"
__author__ = "IBM Research"
__description__ = "vLLM integration for IBM Spyre AI accelerator"