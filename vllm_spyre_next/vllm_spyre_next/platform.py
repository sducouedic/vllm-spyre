"""Spyre platform plugin for vLLM.

This module provides the SpyrePlatform class that registers Spyre as a
hardware backend for vLLM, following the plugin architecture pattern
established by vLLM-Ascend.
"""

from typing import Optional
from vllm.platforms.interface import Platform, PlatformEnum
from vllm.config import VllmConfig


class SpyrePlatform(Platform):
    """Platform plugin for IBM Spyre AI accelerator.
    
    This class integrates Spyre hardware with vLLM by:
    1. Registering the SpyreCompiler backend
    2. Configuring compilation settings for Spyre
    3. Optionally providing graph wrapper for runtime optimization
    
    The platform follows vLLM's plugin architecture, allowing out-of-tree
    (OOT) integration without modifying vLLM core code.
    """
    
    _enum = PlatformEnum.SPYRE
    device_name: str = "spyre"
    device_type: str = "spyre"
    
    @classmethod
    def get_compile_backend(cls) -> str:
        """Return Spyre compiler backend class path.
        
        This method tells vLLM which compiler to use for Spyre hardware.
        The returned string is a fully-qualified class path that vLLM
        will dynamically import and instantiate.
        
        Returns:
            str: Fully-qualified path to SpyreCompiler class
        """
        return "vllm_spyre_next.compilation.compiler_interface.SpyreCompiler"
    
    @classmethod
    def get_static_graph_wrapper_cls(cls) -> Optional[str]:
        """Return graph wrapper class path (optional).
        
        Graph wrappers are runtime optimizations that capture execution
        graphs per batch size to eliminate repeated kernel launch overhead.
        
        Phase 1-2: Return None (no graph wrapper)
        - Focus on core compilation and caching first
        - Measure baseline performance
        
        Phase 3 (optional): Enable graph wrapper if benchmarks show benefit
        - Uncomment the return statement below
        - Implement SpyreGraphWrapper in compilation/spyre_graph.py
        - Only proceed if launchKernel() overhead > 5% of execution time
        
        Returns:
            Optional[str]: Fully-qualified path to SpyreGraphWrapper class,
                or None if graph wrapper is not needed
        """
        # Phase 1-2: No graph wrapper
        return None
        
        # Phase 3 (if needed): Enable graph wrapper
        # return "vllm_spyre_next.compilation.spyre_graph.SpyreGraphWrapper"
    
    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        """Configure compilation settings for Spyre.
        
        This method is called by vLLM during initialization to configure
        compilation settings specific to Spyre hardware. It:
        
        1. Disables standard Inductor (uses torch-spyre's Inductor extension)
        2. Configures piecewise compilation splitting operations
        3. Sets Spyre-specific hardware parameters
        
        Args:
            vllm_config: vLLM configuration to update
        """
        compilation_config = vllm_config.compilation_config
        
        # Disable standard Inductor - use torch-spyre's Inductor extension
        # torch-spyre provides its own Inductor backend that generates
        # SuperDSC JSON instead of standard Triton/C++ code
        compilation_config.use_inductor = False
        
        # Configure piecewise compilation if enabled
        # Piecewise compilation splits the model into subgraphs at specific
        # operations to enable better optimization and caching
        if compilation_config.cudagraph_mode.has_piecewise_cudagraphs():
            # Set splitting operations for Spyre
            # These operations are good split points because they:
            # - Have clear input/output boundaries
            # - Benefit from separate compilation
            # - Enable better cache reuse
            compilation_config.splitting_ops = [
                "aten.mm.default",      # Matrix multiplication
                "aten.addmm.default",   # Add + matrix multiplication
                "aten.bmm.default",     # Batch matrix multiplication
            ]
        
        # Spyre-specific hardware configuration
        # These settings affect compilation and must match the hardware
        compilation_config.spyre_enable_static_kernel = True
        compilation_config.spyre_core_count = 32
        compilation_config.spyre_stick_alignment = 128


def register() -> None:
    """Entry point for vLLM platform plugin.
    
    This function is called by vLLM's plugin system during initialization.
    It registers the SpyrePlatform with vLLM's platform registry.
    
    The entry point is configured in pyproject.toml:
    [project.entry-points."vllm.platform_plugins"]
    spyre_next = "vllm_spyre_next:register"
    """
    from vllm.platforms import register_platform
    register_platform(SpyrePlatform)