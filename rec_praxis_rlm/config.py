"""Configuration models for rec_praxis_rlm package."""

import os
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# Default maximum output characters for safe code execution
# Prevents excessive memory usage from unbounded output
DEFAULT_MAX_OUTPUT_CHARS = 10000


class MemoryConfig(BaseModel):
    """Configuration for ProceduralMemory.

    Attributes:
        storage_path: Path to JSONL file for persistent storage
        top_k: Number of top experiences to retrieve
        similarity_threshold: Minimum similarity score (0.0-1.0)
        env_weight: Weight for environmental feature similarity (0.0-1.0)
        goal_weight: Weight for goal similarity (0.0-1.0)
        require_success: If True, only retrieve successful experiences
        embedding_model: Name of sentence-transformers model
        embedding_api_fallback: API provider for embedding fallback (openai, cohere, voyage)
        result_size_limit: Maximum size of result string in bytes
    """

    storage_path: str = Field(
        default="./memory.jsonl",
        description="Path to JSONL file for persistent storage",
    )
    top_k: int = Field(
        default=6,
        ge=1,
        le=100,
        description="Number of top experiences to retrieve",
    )
    similarity_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score",
    )
    env_weight: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Weight for environmental feature similarity",
    )
    goal_weight: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Weight for goal similarity",
    )
    require_success: bool = Field(
        default=False,
        description="If True, only retrieve successful experiences",
    )
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Name of sentence-transformers model",
    )
    embedding_api_fallback: Optional[str] = Field(
        default=None,
        description="API provider for embedding fallback",
    )
    result_size_limit: int = Field(
        default=50000,
        description="Maximum size of result string in bytes",
    )
    faiss_memory_limit_mb: int = Field(
        default=500,
        ge=10,
        le=10000,
        description="Maximum memory for FAISS index in MB (default 500MB)",
    )

    @field_validator("storage_path")
    @classmethod
    def validate_storage_path(cls, v: str) -> str:
        """Validate storage path to prevent path traversal attacks.

        Args:
            v: Storage path to validate

        Returns:
            Validated storage path

        Raises:
            ValueError: If path contains dangerous traversal patterns
        """
        # Allow :memory: for in-memory databases
        if v == ":memory:":
            return v

        # Resolve to absolute path to detect traversal
        try:
            abs_path = Path(v).resolve()
        except (ValueError, OSError) as e:
            raise ValueError(f"Invalid storage path '{v}': {e}")

        # Check for path traversal by comparing resolved path with input
        # If resolved path escapes the expected directory, it's suspicious
        input_path = Path(v)

        # Reject absolute paths that try to escape to sensitive directories
        dangerous_prefixes = ["/etc", "/sys", "/proc", "/dev", "/root", "C:\\Windows", "C:\\Program Files"]
        abs_str = str(abs_path)
        for prefix in dangerous_prefixes:
            if abs_str.startswith(prefix):
                raise ValueError(
                    f"Storage path '{v}' resolves to dangerous location '{abs_path}'. "
                    f"Paths cannot point to system directories."
                )

        # Warn about suspicious patterns (but don't reject - could be legitimate)
        if ".." in v:
            # Parent traversal detected - make sure resolved path is safe
            cwd = Path.cwd().resolve()
            if not str(abs_path).startswith(str(cwd)):
                raise ValueError(
                    f"Storage path '{v}' attempts to escape current working directory. "
                    f"Resolved to '{abs_path}' which is outside '{cwd}'."
                )

        return v

    @model_validator(mode="after")
    def validate_weight_sum(self) -> "MemoryConfig":
        """Validate that env_weight + goal_weight sum to 1.0."""
        weight_sum = self.env_weight + self.goal_weight
        if abs(weight_sum - 1.0) > 0.001:  # Allow small floating point error
            raise ValueError(f"env_weight + goal_weight must sum to 1.0, got {weight_sum:.3f}")
        return self

    @classmethod
    def for_code_review(cls, storage_path: str = "./memory_code_review.jsonl") -> "MemoryConfig":
        """Create configuration optimized for code review tasks.

        Settings:
        - Higher similarity threshold (0.7) for precise matches
        - Requires successful experiences only
        - Prioritizes goal similarity (code quality patterns)
        - Top 4 most relevant experiences

        Args:
            storage_path: Path to store code review memories

        Returns:
            MemoryConfig instance optimized for code review
        """
        return cls(
            storage_path=storage_path,
            top_k=4,
            similarity_threshold=0.7,
            env_weight=0.3,
            goal_weight=0.7,
            require_success=True,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            result_size_limit=30000,
        )

    @classmethod
    def for_security_audit(cls, storage_path: str = "./memory_security.jsonl") -> "MemoryConfig":
        """Create configuration optimized for security auditing.

        Settings:
        - Lower similarity threshold (0.4) to catch diverse vulnerabilities
        - Includes failed experiences (learn from past false positives)
        - Balanced env/goal weights
        - Top 8 experiences for broader context

        Args:
            storage_path: Path to store security audit memories

        Returns:
            MemoryConfig instance optimized for security auditing
        """
        return cls(
            storage_path=storage_path,
            top_k=8,
            similarity_threshold=0.4,
            env_weight=0.5,
            goal_weight=0.5,
            require_success=False,  # Learn from false positives
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            result_size_limit=50000,
        )

    @classmethod
    def for_web_scraping(cls, storage_path: str = "./memory_scraping.jsonl") -> "MemoryConfig":
        """Create configuration optimized for web scraping tasks.

        Settings:
        - Medium similarity threshold (0.5)
        - Prioritizes environmental similarity (site structure)
        - Requires successful experiences
        - Top 6 experiences

        Args:
            storage_path: Path to store web scraping memories

        Returns:
            MemoryConfig instance optimized for web scraping
        """
        return cls(
            storage_path=storage_path,
            top_k=6,
            similarity_threshold=0.5,
            env_weight=0.7,
            goal_weight=0.3,
            require_success=True,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            result_size_limit=100000,  # Web pages can be large
        )

    @classmethod
    def for_testing(cls, storage_path: str = "./memory_testing.jsonl") -> "MemoryConfig":
        """Create configuration optimized for test generation tasks.

        Settings:
        - High similarity threshold (0.75) for precise test patterns
        - Requires successful experiences
        - Heavily prioritizes goal similarity (test coverage patterns)
        - Top 5 experiences

        Args:
            storage_path: Path to store testing memories

        Returns:
            MemoryConfig instance optimized for test generation
        """
        return cls(
            storage_path=storage_path,
            top_k=5,
            similarity_threshold=0.75,
            env_weight=0.2,
            goal_weight=0.8,
            require_success=True,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            result_size_limit=40000,
        )


class ReplConfig(BaseModel):
    """Configuration for RLMContext.

    Attributes:
        max_output_chars: Maximum characters to capture from code execution
        max_search_matches: Maximum number of search results to return
        search_context_chars: Number of context characters before/after match
        execution_timeout_seconds: Timeout for safe code execution
        regex_timeout_seconds: Timeout for regex search operations (ReDoS protection)
        enable_sandbox: If True, use sandboxed execution (recommended)
        log_executions: If True, log all code executions for audit trail
        allowed_builtins: List of allowed built-in functions
    """

    max_output_chars: int = Field(
        default=DEFAULT_MAX_OUTPUT_CHARS,
        description="Maximum characters to capture from code execution",
    )
    max_search_matches: int = Field(
        default=100,
        description="Maximum number of search results to return",
    )
    search_context_chars: int = Field(
        default=200,
        description="Number of context characters before/after match",
    )
    execution_timeout_seconds: float = Field(
        default=5.0,
        ge=0.1,
        le=60.0,
        description="Timeout for safe code execution",
    )
    regex_timeout_seconds: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Timeout for regex search operations (ReDoS protection)",
    )
    enable_sandbox: bool = Field(
        default=True,
        description="If True, use sandboxed execution",
    )
    log_executions: bool = Field(
        default=True,
        description="If True, log all code executions for audit trail",
    )
    allowed_builtins: list[str] = Field(
        default_factory=lambda: [
            "len",
            "range",
            "sum",
            "max",
            "min",
            "abs",
            "round",
            "sorted",
            "enumerate",
            "zip",
            "map",
            "filter",
            "all",
            "any",
            "str",
            "int",
            "float",
            "bool",
            "list",
            "dict",
            "set",
            "tuple",
        ],
        description="List of allowed built-in functions",
    )


class PlannerConfig(BaseModel):
    """Configuration for PraxisRLMPlanner.

    Attributes:
        lm_model: Language model identifier (e.g., 'openai/gpt-4o-mini')
        api_key: API key for LLM provider (optional, falls back to env vars)
        temperature: Sampling temperature (0.0 = deterministic, 2.0 = very random)
        max_iters: Maximum number of ReAct iterations
        enable_mlflow_tracing: If True, enable MLflow automatic tracing
        log_traces_from_compile: If True, log optimizer compilation traces
        optimizer: Optimizer to use (miprov2, simba, grpo, gepa)
        optimizer_auto_level: Optimizer automation level (light, medium, heavy)
        use_toon_adapter: If True, use TOON format for 40% token reduction (experimental)
    """

    lm_model: str = Field(
        default="openai/gpt-4o-mini",
        description="Language model identifier",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for LLM provider (optional, falls back to env vars)",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    max_iters: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of ReAct iterations",
    )
    enable_mlflow_tracing: bool = Field(
        default=True,
        description="If True, enable MLflow automatic tracing",
    )
    log_traces_from_compile: bool = Field(
        default=False,
        description="If True, log optimizer compilation traces",
    )
    optimizer: str = Field(
        default="miprov2",
        description="Optimizer to use",
    )
    optimizer_auto_level: Literal["light", "medium", "heavy"] = Field(
        default="medium",
        description="Optimizer automation level",
    )
    use_toon_adapter: bool = Field(
        default=False,
        description="If True, use TOON format for 40% token reduction (experimental, requires dspy-toon)",
    )
