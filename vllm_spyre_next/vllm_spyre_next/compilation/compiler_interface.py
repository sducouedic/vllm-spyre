"""Spyre backend compiler implementing vLLM's CompilerInterface.

This module provides the SpyreCompiler class that integrates torch-spyre's
compilation stack with vLLM's caching infrastructure.
"""

from typing import Any, Callable, Optional
import hashlib
import pickle
from pathlib import Path
import torch
from vllm.compilation.compiler_interface import CompilerInterface
from vllm.config import VllmConfig
from vllm.logger import logger

from .caching import SpyreCompiledArtifacts, SpyreArtifactCache


class SpyreCompiler(CompilerInterface):
    """Spyre backend compiler implementing vLLM's CompilerInterface.
    
    Delegates to torch-spyre's compilation stack:
    1. torch.compile(backend="spyre") triggers Inductor extension
    2. Inductor generates SuperDSC JSON
    3. DeepTools compiles to .g2 binaries
    4. Returns callable that executes via launchKernel()
    
    This compiler integrates with vLLM's 3-level caching:
    - Level 1: Inductor disk cache (torch-spyre's existing cache)
    - Level 2: In-memory SHA256 deduplication (SpyreArtifactCache)
    - Level 3: AOT serialization (SpyreSerializableFunction)
    """
    
    def __init__(self, vllm_config: VllmConfig):
        """Initialize SpyreCompiler with vLLM configuration.
        
        Args:
            vllm_config: vLLM configuration containing compilation settings
        """
        super().__init__(vllm_config)
        self.compilation_config = vllm_config.compilation_config
        
        # Spyre-specific configuration
        self.enable_static_kernel = getattr(
            vllm_config.compilation_config,
            'spyre_enable_static_kernel',
            True
        )
        self.core_count = 32  # Spyre hardware constant
        self.stick_alignment = 128  # Spyre memory alignment
        
        # Initialize in-memory artifact cache (Level 2)
        self._artifact_cache: Optional[SpyreArtifactCache] = None
        
    def compile(
        self,
        graph_module: torch.fx.GraphModule,
        example_inputs: list[torch.Tensor],
        **kwargs
    ) -> Callable:
        """Compile FX graph to Spyre executable with 3-level caching.
        
        This method implements a 3-level cache lookup strategy:
        1. Level 2 (fastest): Check in-memory SpyreArtifactCache
        2. Level 1 (fast): Check torch-spyre disk cache (.g2 binaries)
        3. Cache miss (slow): Compile via torch-spyre stack
        
        The torch-spyre compilation stack:
        1. Uses Inductor extension to generate SuperDSC JSON
        2. Invokes DeepTools compiler to produce .g2 binaries
        3. Returns a callable that executes via launchKernel()
        
        Args:
            graph_module: FX graph from Dynamo tracing
            example_inputs: Example tensors for shape inference
            **kwargs: Additional compilation arguments
            
        Returns:
            SpyreCompiledArtifacts: Wrapper containing compiled callable
                and metadata for caching
        """
        # Create artifacts object for hash computation
        temp_artifacts = SpyreCompiledArtifacts(
            compiled_fn=None,  # Will be filled later
            graph_module=graph_module,
            example_inputs=example_inputs,
            compiler_config=self._get_compiler_config()
        )
        artifact_hash = temp_artifacts.artifact_hash
        
        # Level 2: Check in-memory cache (fastest - O(1) hash lookup)
        if self._artifact_cache is not None and self._artifact_cache.has(artifact_hash):
            logger.info(f"Cache hit (Level 2 - in-memory) for hash {artifact_hash[:8]}...")
            return self._artifact_cache.load(artifact_hash)
        
        # Level 1: Check torch-spyre disk cache (fast - avoids compilation)
        try:
            from torch_spyre.caching import (
                generate_cache_key,
                get_cache_dir,
                cache_exists,
                load_cached_artifacts
            )
            
            cache_key = generate_cache_key(graph_module)
            cache_dir = get_cache_dir(cache_key)
            
            if cache_exists(cache_dir):
                logger.info(f"Cache hit (Level 1 - torch-spyre disk) for key {cache_key[:8]}...")
                compiled_fn = load_cached_artifacts(cache_dir)
                
                # Create artifacts and store in Level 2 cache
                artifacts = SpyreCompiledArtifacts(
                    compiled_fn=compiled_fn,
                    graph_module=graph_module,
                    example_inputs=example_inputs,
                    compiler_config=self._get_compiler_config()
                )
                
                if self._artifact_cache is not None:
                    self._artifact_cache.store(artifacts)
                
                return artifacts
        except ImportError:
            logger.debug("torch-spyre caching functions not available, skipping Level 1 cache")
        except Exception as e:
            logger.warning(f"Error checking torch-spyre cache: {e}, falling back to compilation")
        
        # Cache miss - compile via torch-spyre stack
        logger.info(f"Cache miss for hash {artifact_hash[:8]}..., compiling...")
        
        # Delegate to torch-spyre's compilation stack
        # This triggers: Inductor extension → SuperDSC JSON → DeepTools → .g2
        compiled_fn = torch.compile(
            graph_module,
            backend="spyre",
            fullgraph=True,
            dynamic=False  # Spyre benefits from static shapes
        )
        
        # Wrap in SpyreCompiledArtifacts for caching
        artifacts = SpyreCompiledArtifacts(
            compiled_fn=compiled_fn,
            graph_module=graph_module,
            example_inputs=example_inputs,
            compiler_config=self._get_compiler_config()
        )
        
        # Store in Level 2 cache for subsequent access
        if self._artifact_cache is not None:
            self._artifact_cache.store(artifacts)
        
        return artifacts
    
    def initialize_cache(self) -> None:
        """Initialize the in-memory artifact cache (Level 2).
        
        This method sets up the SpyreArtifactCache for SHA256-based
        deduplication of compiled artifacts. Should be called before
        the first compilation.
        """
        if self._artifact_cache is None:
            self._artifact_cache = SpyreArtifactCache()
            logger.info("Initialized SpyreArtifactCache (Level 2 in-memory cache)")
    
    def load(self, path: Path) -> Callable:
        """Load compiled artifacts from AOT cache (Level 3).
        
        This method deserializes artifacts from disk and restores them
        into the in-memory cache. It attempts to use torch-spyre's disk
        cache (Level 1) before recompiling.
        
        Args:
            path: Path to serialized artifacts file
            
        Returns:
            Callable: Compiled function ready for execution
            
        Raises:
            FileNotFoundError: If the cache file doesn't exist
            pickle.UnpicklingError: If deserialization fails
        """
        if not path.exists():
            raise FileNotFoundError(f"AOT cache file not found: {path}")
        
        logger.info(f"Loading artifacts from AOT cache (Level 3): {path}")
        
        # Read serialized data
        with open(path, 'rb') as f:
            data = f.read()
        
        # Deserialize artifacts (will check torch-spyre cache internally)
        artifacts = SpyreCompiledArtifacts.deserialize(data)
        
        # Store in Level 2 cache for subsequent access
        if self._artifact_cache is not None:
            self._artifact_cache.store(artifacts)
        
        return artifacts
    
    def compute_hash(self) -> str:
        """Compute cache key for Spyre compilation.
        
        Inherits vLLM's base hash (environment + vllm_config + source code)
        and adds Spyre-specific factors that affect compilation output.
        
        Returns:
            str: Combined hash string in format "{base_hash}_{spyre_hash}"
        """
        base_hash = self.vllm_config.compute_hash()
        
        # Add Spyre-specific hash factors
        spyre_factors = {
            'enable_static_kernel': self.enable_static_kernel,
            'core_count': self.core_count,
            'stick_alignment': self.stick_alignment,
            'torch_spyre_version': self._get_torch_spyre_version(),
            'deeptools_version': self._get_deeptools_version(),
        }
        
        # Create deterministic hash
        spyre_hash = hashlib.sha256(
            str(sorted(spyre_factors.items())).encode()
        ).hexdigest()[:16]
        
        return f"{base_hash}_{spyre_hash}"
    
    def _get_compiler_config(self) -> dict:
        """Get Spyre compiler configuration.
        
        Returns:
            dict: Configuration dictionary for SpyreCompiledArtifacts
        """
        return {
            'enable_static_kernel': self.enable_static_kernel,
            'core_count': self.core_count,
            'stick_alignment': self.stick_alignment,
        }
    
    def _get_torch_spyre_version(self) -> str:
        """Get torch-spyre version for cache invalidation.
        
        Returns:
            str: torch-spyre version string or "unknown"
        """
        try:
            import torch_spyre
            return getattr(torch_spyre, '__version__', 'unknown')
        except (ImportError, AttributeError):
            return "unknown"
    
    def _get_deeptools_version(self) -> str:
        """Get DeepTools compiler version.
        
        TODO: Implement version detection by calling dxp_standalone --version
        or reading from environment/config.
        
        Returns:
            str: DeepTools version string or "unknown"
        """
        # TODO: Implement version detection
        # Could run: subprocess.run(["dxp_standalone", "--version"], ...)
        return "unknown"