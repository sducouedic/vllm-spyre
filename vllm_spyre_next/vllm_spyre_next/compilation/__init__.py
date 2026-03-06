"""Compilation module for vLLM-Spyre Next.

This module provides the compilation infrastructure for Spyre hardware:

Phase 1 - Core Compiler Interface:
    - SpyreCompiler: Implements vLLM's CompilerInterface
    - Delegates to torch-spyre's compilation stack
    - Generates cache keys with Spyre-specific factors

Phase 2 - Artifact Caching:
    - SpyreCompiledArtifacts: Manages compiled artifacts with SHA256 deduplication
    - SpyreArtifactCache: In-memory cache with two-level storage
    - 3-level caching strategy:
        * Level 1: Inductor disk cache (.g2 binaries)
        * Level 2: In-memory SHA256 deduplication
        * Level 3: AOT serialization (future)

Phase 3 - Graph Wrapper (Optional):
    - SpyreGraphWrapper: Runtime optimization for graph capture/replay
    - Only implement if benchmarks show launchKernel() overhead > 5%

Exports:
    SpyreCompiler: Main compiler class
    SpyreCompiledArtifacts: Artifact management
    SpyreArtifactCache: In-memory cache
"""

from vllm_spyre_next.compilation.compiler_interface import SpyreCompiler
from vllm_spyre_next.compilation.caching import (
    SpyreCompiledArtifacts,
    SpyreArtifactCache,
)

__all__ = [
    "SpyreCompiler",
    "SpyreCompiledArtifacts",
    "SpyreArtifactCache",
]