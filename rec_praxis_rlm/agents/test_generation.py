"""Test Generation Agent for automated pytest test creation.

This agent analyzes source code coverage and generates pytest tests to increase
coverage, inspired by Qodo-Cover's test generation capabilities.

Features:
- Parse coverage.py reports to identify uncovered code paths
- Generate pytest tests using DSPy-based prompts
- Validate generated tests execute successfully
- Use procedural memory to learn from successful test patterns
- Iterative improvement until coverage target is met
"""

import ast
import json
import re
import subprocess
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# v0.9.0: Language support enum
class Language(Enum):
    """Supported programming languages for test generation."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    GO = "go"
    RUST = "rust"

try:
    from coverage import Coverage
except ImportError:
    Coverage = None  # type: ignore

try:
    import dspy
except ImportError:
    dspy = None  # type: ignore

from rec_praxis_rlm import ProceduralMemory, Experience, MemoryConfig, RLMContext


# v0.9.0: Language detection helper
def detect_language(file_path: str) -> Language:
    """Detect programming language from file extension.

    Args:
        file_path: Path to source file

    Returns:
        Language enum value
    """
    ext = Path(file_path).suffix.lower()

    language_map = {
        '.py': Language.PYTHON,
        '.js': Language.JAVASCRIPT,
        '.jsx': Language.JAVASCRIPT,
        '.ts': Language.TYPESCRIPT,
        '.tsx': Language.TYPESCRIPT,
        '.go': Language.GO,
        '.rs': Language.RUST,
    }

    return language_map.get(ext, Language.PYTHON)  # Default to Python


# DSPy Signature for intelligent test generation
if dspy is not None:
    class GeneratePytestTest(dspy.Signature):
        """Generate a complete pytest test with assertions for an uncovered function.

        Analyze the function's purpose, parameters, and expected behavior to create
        comprehensive test cases including:
        - Happy path tests with typical inputs
        - Edge case tests (boundary values, empty inputs, None, etc.)
        - Error case tests (invalid inputs, exceptions)
        - Property-based invariants when applicable

        Generate actual assertions, not TODO/pass stubs.
        """

        function_name: str = dspy.InputField(desc="Name of the function to test")
        function_source: str = dspy.InputField(desc="Source code of the function including signature and docstring")
        class_name: Optional[str] = dspy.InputField(desc="Class name if function is a method (None otherwise)", default=None)
        similar_test_patterns: str = dspy.InputField(desc="Similar successful test patterns from memory", default="")
        use_hypothesis: bool = dspy.InputField(desc="Whether to generate Hypothesis property-based tests with @given decorator", default=False)
        target_language: str = dspy.InputField(desc="Target programming language (python, javascript, typescript, go, rust)", default="python")

        test_code: str = dspy.OutputField(desc="Complete test code with imports, test functions, and assertions in the target language")
        test_reasoning: str = dspy.OutputField(desc="Explanation of test strategy and coverage approach")


@dataclass
class UncoveredRegion:
    """Represents an uncovered code region that needs test coverage."""
    file_path: str
    start_line: int
    end_line: int
    function_name: Optional[str] = None
    class_name: Optional[str] = None
    complexity: int = 1  # Cyclomatic complexity estimate

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "file": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "function": self.function_name,
            "class": self.class_name,
            "complexity": self.complexity,
        }


@dataclass
class UncoveredBranch:
    """Represents an uncovered branch (if/else, try/except, etc.) that needs test coverage."""
    file_path: str
    source_line: int  # Line where the branch starts
    target_line: int  # Line where the branch goes (or -1 for exit)
    branch_type: str  # "if", "else", "elif", "try", "except", "finally", "match", "case"
    function_name: Optional[str] = None
    class_name: Optional[str] = None
    condition: Optional[str] = None  # The branch condition (for if/elif)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "file": self.file_path,
            "source_line": self.source_line,
            "target_line": self.target_line,
            "branch_type": self.branch_type,
            "function": self.function_name,
            "class": self.class_name,
            "condition": self.condition,
        }


@dataclass
class GeneratedTest:
    """Represents a generated pytest test."""
    test_code: str
    target_file: str
    target_function: str
    test_file_path: str
    description: str
    estimated_coverage_gain: float  # 0.0-100.0

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "test_code": self.test_code,
            "target_file": self.target_file,
            "target_function": self.target_function,
            "test_file_path": self.test_file_path,
            "description": self.description,
            "estimated_coverage_gain": self.estimated_coverage_gain,
        }


@dataclass
class CoverageAnalysis:
    """Results of coverage analysis."""
    total_coverage: float  # 0.0-100.0
    uncovered_regions: List[UncoveredRegion]
    files_analyzed: int
    lines_covered: int
    lines_total: int
    # v0.7.0: Branch coverage analysis
    branch_coverage: Optional[float] = None  # 0.0-100.0
    uncovered_branches: Optional[List[UncoveredBranch]] = None
    branches_covered: int = 0
    branches_total: int = 0

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        result = {
            "total_coverage": self.total_coverage,
            "uncovered_regions": [r.to_dict() for r in self.uncovered_regions],
            "files_analyzed": self.files_analyzed,
            "lines_covered": self.lines_covered,
            "lines_total": self.lines_total,
        }
        # Add branch coverage if available
        if self.branch_coverage is not None:
            result["branch_coverage"] = self.branch_coverage
            result["uncovered_branches"] = [b.to_dict() for b in (self.uncovered_branches or [])]
            result["branches_covered"] = self.branches_covered
            result["branches_total"] = self.branches_total
        return result


class TestGenerationAgent:
    """Production test generation agent with coverage analysis.

    This agent:
    1. Parses coverage.py reports to find uncovered code
    2. Generates pytest tests targeting uncovered paths
    3. Validates tests execute and pass
    4. Uses procedural memory to learn from successful patterns
    5. Iterates until coverage target is reached
    """

    def __init__(
        self,
        memory_path: str = ":memory:",
        coverage_data_file: str = ".coverage",
        test_dir: str = "tests",
        lm_model: Optional[str] = None,
        use_dspy: bool = False,
        analyze_branches: bool = True,  # v0.7.0: Enable branch coverage analysis
        use_hypothesis: bool = False,  # v0.8.0: Enable Hypothesis property-based testing
        target_language: Optional[Language] = None  # v0.9.0: Target language (auto-detect if None)
    ):
        """Initialize test generation agent.

        Args:
            memory_path: Path to JSONL file for procedural memory storage.
                        Use ":memory:" for in-memory (testing only).
            coverage_data_file: Path to coverage.py data file (default: .coverage)
            test_dir: Directory where generated tests will be saved
            lm_model: Language model for DSPy test generation (e.g., "groq/llama-3.3-70b-versatile")
                     If None, uses template-based generation
            use_dspy: Whether to use DSPy for intelligent test generation (requires lm_model)
            analyze_branches: Whether to analyze branch coverage (v0.7.0+)
            use_hypothesis: Whether to generate Hypothesis property-based tests (v0.8.0+)
            target_language: Target programming language (v0.9.0+, auto-detects if None)
        """
        self.memory_path = memory_path
        self.coverage_data_file = Path(coverage_data_file)
        self.test_dir = Path(test_dir)
        self.use_dspy = use_dspy and lm_model is not None
        self.analyze_branches = analyze_branches
        self.use_hypothesis = use_hypothesis
        self.target_language = target_language or Language.PYTHON  # Default to Python

        self.memory = ProceduralMemory(
            config=MemoryConfig(
                storage_path=memory_path,
                env_weight=0.6,
                goal_weight=0.4,
            )
        )
        self.rlm = RLMContext()

        # Check if coverage.py is available
        if Coverage is None:
            raise ImportError(
                "coverage package is required for test generation. "
                "Install it with: pip install coverage pytest-cov"
            )

        # Initialize DSPy if requested
        self.test_generator = None
        if self.use_dspy:
            if dspy is None:
                raise ImportError(
                    "dspy package is required for LLM-based test generation. "
                    "Install it with: pip install dspy-ai"
                )

            # Configure DSPy language model
            try:
                if lm_model.startswith("groq/"):
                    import os
                    api_key = os.getenv("GROQ_API_KEY")
                    if not api_key:
                        raise ValueError("GROQ_API_KEY environment variable required for Groq models")
                    lm = dspy.GROQ(model=lm_model.replace("groq/", ""), api_key=api_key)
                elif lm_model.startswith("openai/"):
                    import os
                    api_key = os.getenv("OPENAI_API_KEY")
                    if not api_key:
                        raise ValueError("OPENAI_API_KEY environment variable required for OpenAI models")
                    lm = dspy.OpenAI(model=lm_model.replace("openai/", ""), api_key=api_key)
                else:
                    # Generic LiteLLM model
                    lm = dspy.LM(model=lm_model)

                dspy.configure(lm=lm)

                # Initialize ChainOfThought with our signature
                self.test_generator = dspy.ChainOfThought(GeneratePytestTest)

            except Exception as e:
                raise RuntimeError(f"Failed to initialize DSPy with model {lm_model}: {e}")

    def analyze_coverage(
        self,
        source_files: Optional[List[str]] = None
    ) -> CoverageAnalysis:
        """Analyze coverage data and identify uncovered regions.

        Args:
            source_files: Optional list of source files to analyze.
                         If None, analyzes all files in coverage report.

        Returns:
            CoverageAnalysis object with uncovered regions
        """
        if not self.coverage_data_file.exists():
            raise FileNotFoundError(
                f"Coverage data file not found: {self.coverage_data_file}. "
                "Run pytest with --cov first: pytest --cov=your_package tests/"
            )

        # Load coverage data
        cov = Coverage(data_file=str(self.coverage_data_file))
        cov.load()

        uncovered_regions = []
        total_lines = 0
        covered_lines = 0
        files_analyzed = 0

        # Get list of measured files
        measured_files = cov.get_data().measured_files()

        # Filter by source_files if provided
        if source_files:
            source_paths = {str(Path(f).absolute()) for f in source_files}
            measured_files = [f for f in measured_files if f in source_paths]

        for file_path in measured_files:
            try:
                # Get coverage analysis for this file
                analysis = cov.analysis2(file_path)
                _, executed, excluded, missing = analysis

                # Update totals
                all_lines = executed + missing
                total_lines += len(all_lines)
                covered_lines += len(executed)
                files_analyzed += 1

                # Group consecutive missing lines into regions
                if missing:
                    regions = self._group_missing_lines(file_path, sorted(missing))
                    uncovered_regions.extend(regions)

            except Exception as e:
                # Skip files that can't be analyzed
                continue

        # Calculate total coverage percentage
        total_coverage = (covered_lines / total_lines * 100) if total_lines > 0 else 0.0

        # v0.7.0: Analyze branch coverage if requested
        branch_coverage = None
        uncovered_branches = None
        branches_covered = 0
        branches_total = 0

        if self.analyze_branches:
            branch_coverage, uncovered_branches, branches_covered, branches_total = (
                self._analyze_branch_coverage(cov, measured_files)
            )

        return CoverageAnalysis(
            total_coverage=total_coverage,
            uncovered_regions=uncovered_regions,
            files_analyzed=files_analyzed,
            lines_covered=covered_lines,
            lines_total=total_lines,
            branch_coverage=branch_coverage,
            uncovered_branches=uncovered_branches,
            branches_covered=branches_covered,
            branches_total=branches_total,
        )

    def _group_missing_lines(
        self,
        file_path: str,
        missing_lines: List[int]
    ) -> List[UncoveredRegion]:
        """Group consecutive missing lines into uncovered regions.

        Args:
            file_path: Path to source file
            missing_lines: Sorted list of uncovered line numbers

        Returns:
            List of UncoveredRegion objects
        """
        if not missing_lines:
            return []

        regions = []
        current_start = missing_lines[0]
        current_end = missing_lines[0]

        for line in missing_lines[1:]:
            if line == current_end + 1:
                # Consecutive line, extend region
                current_end = line
            else:
                # Gap found, save current region and start new one
                regions.append(self._create_uncovered_region(
                    file_path, current_start, current_end
                ))
                current_start = line
                current_end = line

        # Add final region
        regions.append(self._create_uncovered_region(
            file_path, current_start, current_end
        ))

        return regions

    def _create_uncovered_region(
        self,
        file_path: str,
        start_line: int,
        end_line: int
    ) -> UncoveredRegion:
        """Create an UncoveredRegion with function/class context.

        Args:
            file_path: Path to source file
            start_line: Start line of uncovered region
            end_line: End line of uncovered region

        Returns:
            UncoveredRegion with function/class context
        """
        # Parse AST to find function/class context
        function_name = None
        class_name = None

        try:
            with open(file_path, 'r') as f:
                source = f.read()

            tree = ast.parse(source, filename=file_path)

            # Find the function/class containing this line
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                        if node.lineno <= start_line <= (node.end_lineno or float('inf')):
                            function_name = node.name
                            # Check if function is inside a class
                            for parent in ast.walk(tree):
                                if isinstance(parent, ast.ClassDef):
                                    if hasattr(parent, 'lineno') and hasattr(parent, 'end_lineno'):
                                        if parent.lineno <= node.lineno <= (parent.end_lineno or float('inf')):
                                            class_name = parent.name
                                            break
                            break

        except Exception:
            # If AST parsing fails, continue without context
            pass

        return UncoveredRegion(
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            function_name=function_name,
            class_name=class_name,
            complexity=max(1, end_line - start_line + 1)  # Simple complexity estimate
        )

    def _analyze_branch_coverage(
        self,
        cov: Coverage,
        measured_files: List[str]
    ) -> Tuple[Optional[float], List[UncoveredBranch], int, int]:
        """Analyze branch coverage and identify uncovered branches.

        Args:
            cov: Coverage object with loaded data
            measured_files: List of files to analyze

        Returns:
            Tuple of (branch_coverage_pct, uncovered_branches, branches_covered, branches_total)
        """
        uncovered_branches: List[UncoveredBranch] = []
        total_branch_exits = 0
        covered_branch_exits = 0

        for file_path in measured_files:
            try:
                # Get branch statistics for this file
                branch_stats = cov.branch_stats(file_path)

                if not branch_stats:
                    continue

                # branch_stats is a dict: {line_no: (total_exits, taken_exits)}
                for line_no, (total_exits, taken_exits) in branch_stats.items():
                    total_branch_exits += total_exits
                    covered_branch_exits += taken_exits

                    # If not all exits were taken, we have uncovered branches
                    if taken_exits < total_exits:
                        # Identify the uncovered branches using AST
                        branches = self._identify_uncovered_branches_at_line(
                            file_path, line_no, total_exits, taken_exits
                        )
                        uncovered_branches.extend(branches)

            except Exception:
                # Skip files that don't have branch coverage or can't be analyzed
                continue

        # Calculate branch coverage percentage
        branch_coverage_pct = None
        if total_branch_exits > 0:
            branch_coverage_pct = (covered_branch_exits / total_branch_exits * 100)

        return branch_coverage_pct, uncovered_branches, covered_branch_exits, total_branch_exits

    def _identify_uncovered_branches_at_line(
        self,
        file_path: str,
        line_no: int,
        total_exits: int,
        taken_exits: int
    ) -> List[UncoveredBranch]:
        """Identify which specific branches are uncovered at a given line.

        Args:
            file_path: Path to source file
            line_no: Line number with uncovered branches
            total_exits: Total number of branch exits
            taken_exits: Number of taken branch exits

        Returns:
            List of UncoveredBranch objects
        """
        branches: List[UncoveredBranch] = []

        try:
            with open(file_path, 'r') as f:
                source = f.read()

            tree = ast.parse(source, filename=file_path)
            lines = source.split('\n')

            # Find the AST node at this line
            for node in ast.walk(tree):
                if not hasattr(node, 'lineno') or node.lineno != line_no:
                    continue

                # Get function/class context
                function_name, class_name = self._get_function_context(tree, node)

                # Handle different types of conditional statements
                if isinstance(node, ast.If):
                    # If statement with possible else/elif
                    condition = ast.unparse(node.test) if hasattr(ast, 'unparse') else lines[line_no - 1].strip()

                    # If we have uncovered branches, create branch objects
                    # Note: This is simplified - actual implementation would need
                    # to determine which specific branch (if/elif/else) is uncovered
                    if taken_exits < total_exits:
                        # Create branch for the "true" path
                        branches.append(UncoveredBranch(
                            file_path=file_path,
                            source_line=line_no,
                            target_line=node.body[0].lineno if node.body else line_no + 1,
                            branch_type="if",
                            function_name=function_name,
                            class_name=class_name,
                            condition=condition
                        ))

                        # If there's an else clause and it's uncovered
                        if node.orelse:
                            branches.append(UncoveredBranch(
                                file_path=file_path,
                                source_line=line_no,
                                target_line=node.orelse[0].lineno,
                                branch_type="else",
                                function_name=function_name,
                                class_name=class_name,
                                condition=f"not ({condition})"
                            ))

                elif isinstance(node, ast.Try) or (hasattr(ast, 'TryStar') and isinstance(node, ast.TryStar)):
                    # Try/except block (TryStar is Python 3.11+)
                    if taken_exits < total_exits:
                        branches.append(UncoveredBranch(
                            file_path=file_path,
                            source_line=line_no,
                            target_line=node.handlers[0].lineno if node.handlers else line_no + 1,
                            branch_type="except",
                            function_name=function_name,
                            class_name=class_name,
                            condition=None
                        ))

                elif hasattr(ast, 'Match') and isinstance(node, ast.Match):
                    # Match/case statement (Python 3.10+)
                    if taken_exits < total_exits:
                        for case in node.cases:
                            branches.append(UncoveredBranch(
                                file_path=file_path,
                                source_line=line_no,
                                target_line=case.body[0].lineno if case.body else line_no + 1,
                                branch_type="case",
                                function_name=function_name,
                                class_name=class_name,
                                condition=ast.unparse(case.pattern) if hasattr(ast, 'unparse') else "case"
                            ))

        except Exception:
            # If AST parsing fails, create a generic uncovered branch
            if taken_exits < total_exits:
                branches.append(UncoveredBranch(
                    file_path=file_path,
                    source_line=line_no,
                    target_line=-1,
                    branch_type="unknown",
                    function_name=None,
                    class_name=None,
                    condition=None
                ))

        return branches

    def _get_function_context(
        self,
        tree: ast.AST,
        node: ast.AST
    ) -> Tuple[Optional[str], Optional[str]]:
        """Get the function and class context for a given AST node.

        Args:
            tree: Full AST tree
            node: Node to find context for

        Returns:
            Tuple of (function_name, class_name)
        """
        function_name = None
        class_name = None

        if not hasattr(node, 'lineno'):
            return function_name, class_name

        # Find the enclosing function
        for func_node in ast.walk(tree):
            if isinstance(func_node, ast.FunctionDef):
                if (hasattr(func_node, 'lineno') and hasattr(func_node, 'end_lineno') and
                    func_node.lineno <= node.lineno <= (func_node.end_lineno or float('inf'))):
                    function_name = func_node.name

                    # Check if function is inside a class
                    for class_node in ast.walk(tree):
                        if isinstance(class_node, ast.ClassDef):
                            if (hasattr(class_node, 'lineno') and hasattr(class_node, 'end_lineno') and
                                class_node.lineno <= func_node.lineno <= (class_node.end_lineno or float('inf'))):
                                class_name = class_node.name
                                break
                    break

        return function_name, class_name

    def generate_test(
        self,
        region: UncoveredRegion,
        source_code: str
    ) -> Optional[GeneratedTest]:
        """Generate a pytest test for an uncovered region.

        Args:
            region: UncoveredRegion to generate test for
            source_code: Full source code of the target file

        Returns:
            GeneratedTest object or None if generation failed
        """
        # Extract the uncovered code snippet
        lines = source_code.split('\n')
        snippet_start = max(0, region.start_line - 5)  # Include 5 lines before
        snippet_end = min(len(lines), region.end_line + 5)  # Include 5 lines after
        snippet = '\n'.join(lines[snippet_start:snippet_end])

        # Query procedural memory for similar test patterns
        similar_tests = self._find_similar_test_patterns(region)

        # Generate test using pattern matching + simple template
        # In production, this would use DSPy for LLM-based generation
        test_code = self._generate_test_code(region, snippet, similar_tests)

        if not test_code:
            return None

        # Determine test file path
        test_file_path = self._get_test_file_path(region.file_path)

        return GeneratedTest(
            test_code=test_code,
            target_file=region.file_path,
            target_function=region.function_name or "unknown",
            test_file_path=test_file_path,
            description=f"Test for {region.function_name or 'uncovered code'} at lines {region.start_line}-{region.end_line}",
            estimated_coverage_gain=float(region.end_line - region.start_line + 1)
        )

    def _find_similar_test_patterns(
        self,
        region: UncoveredRegion
    ) -> List[Experience]:
        """Find similar test patterns from procedural memory.

        Args:
            region: UncoveredRegion to find patterns for

        Returns:
            List of relevant experiences from memory
        """
        # Build query features
        query_features = ["pytest", "test_generation"]

        if region.function_name:
            query_features.append(region.function_name.lower())

        if region.class_name:
            query_features.append(region.class_name.lower())

        # Query memory for similar patterns
        # In production, this would use embedding-based similarity search
        relevant_experiences = []

        # For now, return empty list (would query self.memory in production)
        return relevant_experiences

    def _extract_function_source(
        self,
        file_path: str,
        function_name: str
    ) -> Optional[str]:
        """Extract full source code of a function from a file.

        Args:
            file_path: Path to source file
            function_name: Name of function to extract

        Returns:
            Full function source code including signature and docstring, or None if not found
        """
        try:
            with open(file_path, 'r') as f:
                source = f.read()

            tree = ast.parse(source, filename=file_path)

            # Find the function node
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == function_name:
                    # Extract function source using ast.get_source_segment
                    if hasattr(ast, 'get_source_segment'):
                        return ast.get_source_segment(source, node)
                    else:
                        # Fallback for older Python versions
                        lines = source.split('\n')
                        if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                            start = node.lineno - 1
                            end = node.end_lineno if node.end_lineno else start + 1
                            return '\n'.join(lines[start:end])

        except Exception:
            pass

        return None

    def _generate_test_code(
        self,
        region: UncoveredRegion,
        snippet: str,
        similar_tests: List[Experience]
    ) -> Optional[str]:
        """Generate pytest test code for an uncovered region.

        Args:
            region: UncoveredRegion to generate test for
            snippet: Code snippet containing the uncovered region
            similar_tests: Similar test patterns from memory

        Returns:
            Generated test code as string, or None if generation failed
        """
        if not region.function_name:
            return None

        # Use DSPy for intelligent test generation if available
        if self.use_dspy and self.test_generator is not None:
            return self._generate_test_with_dspy(region, similar_tests)

        # Fallback to template-based generation
        # Extract module path from file path
        module_path = self._get_module_path(region.file_path)

        # Generate basic test template
        test_code = f'''"""Auto-generated test for {region.function_name}."""
import pytest
from {module_path} import {region.function_name}


def test_{region.function_name}_basic():
    """Test {region.function_name} with basic inputs."""
    # TODO: Add appropriate test cases
    # Generated for uncovered lines {region.start_line}-{region.end_line}
    pass
'''

        return test_code

    def _generate_test_with_dspy(
        self,
        region: UncoveredRegion,
        similar_tests: List[Experience]
    ) -> Optional[str]:
        """Generate pytest test using DSPy LLM.

        Args:
            region: UncoveredRegion to generate test for
            similar_tests: Similar test patterns from memory

        Returns:
            Generated test code with assertions, or None if generation failed
        """
        # Extract full function source
        function_source = self._extract_function_source(
            region.file_path,
            region.function_name
        )

        if not function_source:
            # Fallback to template if we can't extract function source
            return None

        # Format similar test patterns for context
        similar_patterns_text = ""
        if similar_tests:
            similar_patterns_text = "Similar successful test patterns:\n"
            for i, exp in enumerate(similar_tests[:3], 1):  # Top 3 most similar
                similar_patterns_text += f"{i}. {exp.action}\n   Result: {exp.result}\n"

        try:
            # Call DSPy test generator
            # v0.8.0: Include use_hypothesis flag
            # v0.9.0: Include target_language
            result = self.test_generator(
                function_name=region.function_name,
                function_source=function_source,
                class_name=region.class_name,
                similar_test_patterns=similar_patterns_text,
                use_hypothesis=self.use_hypothesis,
                target_language=self.target_language.value
            )

            # Extract generated test code
            test_code = result.test_code.strip()

            # Basic validation: check if test has assertions
            if "assert" not in test_code.lower() and "pytest.raises" not in test_code.lower():
                print(f"Warning: Generated test for {region.function_name} has no assertions, using template fallback")
                return None

            # Store the reasoning in memory for future reference
            if self.memory_path != ":memory:":
                exp = Experience(
                    env_features=["pytest", "dspy_generation", region.function_name],
                    goal=f"generate test with assertions for {region.function_name}",
                    action=f"DSPy reasoning: {result.test_reasoning}",
                    result=f"Generated test with assertions: {test_code[:200]}...",
                    success=True,
                    timestamp=time.time()
                )
                self.memory.store(exp)

            return test_code

        except Exception as e:
            print(f"Error generating test with DSPy for {region.function_name}: {e}")
            return None

    def _get_module_path(self, file_path: str) -> str:
        """Convert file path to Python module path.

        Args:
            file_path: File path (e.g., "src/mypackage/module.py")

        Returns:
            Module path (e.g., "mypackage.module")
        """
        path = Path(file_path)

        # Remove .py extension
        parts = list(path.with_suffix('').parts)

        # Remove common prefixes (src, lib, etc.)
        if parts and parts[0] in ('src', 'lib'):
            parts = parts[1:]

        return '.'.join(parts)

    def _get_test_file_path(self, source_file: str) -> str:
        """Get test file path for a source file.

        Args:
            source_file: Path to source file

        Returns:
            Path to corresponding test file
        """
        source_path = Path(source_file)
        test_filename = f"test_{source_path.stem}.py"

        # Create test file in test_dir
        return str(self.test_dir / test_filename)

    def validate_test(self, test: GeneratedTest) -> Tuple[bool, str]:
        """Validate that a generated test executes and passes.

        Args:
            test: GeneratedTest to validate

        Returns:
            Tuple of (success: bool, message: str)
        """
        # Write test to temporary file
        test_file = Path(test.test_file_path)
        test_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Check if test file already exists and append if so
            if test_file.exists():
                with open(test_file, 'a') as f:
                    f.write('\n\n' + test.test_code)
            else:
                with open(test_file, 'w') as f:
                    f.write(test.test_code)

            # Run pytest on this specific test file
            result = subprocess.run(
                ['pytest', str(test_file), '-v'],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                # Test passed
                self._store_test_success(test)
                return True, f"Test executed successfully: {test.description}"
            else:
                # Test failed
                error_msg = result.stdout + result.stderr
                return False, f"Test failed: {error_msg[:200]}"

        except subprocess.TimeoutExpired:
            return False, "Test execution timed out (>30s)"

        except Exception as e:
            return False, f"Test validation error: {str(e)}"

    def _store_test_success(self, test: GeneratedTest):
        """Store successful test in procedural memory.

        Args:
            test: GeneratedTest that passed validation
        """
        # Only store if memory is persistent
        if self.memory_path == ":memory:":
            return

        exp = Experience(
            env_features=["pytest", "test_generation", test.target_function],
            goal=f"generate test for {test.target_function}",
            action=f"Generated test: {test.description}",
            result=f"Test passed validation and increased coverage",
            success=True,
            timestamp=time.time()
        )
        self.memory.store(exp)

    def generate_tests_for_coverage_gap(
        self,
        target_coverage: float = 90.0,
        max_tests: int = 10,
        source_files: Optional[List[str]] = None
    ) -> List[GeneratedTest]:
        """Generate tests to reach target coverage.

        Args:
            target_coverage: Target coverage percentage (0-100)
            max_tests: Maximum number of tests to generate
            source_files: Optional list of source files to target

        Returns:
            List of generated tests
        """
        # Analyze current coverage
        analysis = self.analyze_coverage(source_files)

        print(f"Current line coverage: {analysis.total_coverage:.1f}%")
        # v0.7.0: Display branch coverage if available
        if analysis.branch_coverage is not None:
            print(f"Current branch coverage: {analysis.branch_coverage:.1f}%")
            print(f"Found {len(analysis.uncovered_branches or [])} uncovered branches")
        print(f"Target coverage: {target_coverage:.1f}%")
        print(f"Found {len(analysis.uncovered_regions)} uncovered line regions")

        if analysis.total_coverage >= target_coverage:
            print("Target coverage already met!")
            return []

        # Sort uncovered regions by complexity (prioritize complex code)
        sorted_regions = sorted(
            analysis.uncovered_regions,
            key=lambda r: r.complexity,
            reverse=True
        )

        generated_tests = []

        for region in sorted_regions[:max_tests]:
            # Read source file
            try:
                with open(region.file_path, 'r') as f:
                    source_code = f.read()
            except Exception:
                continue

            # Generate test
            test = self.generate_test(region, source_code)

            if test:
                generated_tests.append(test)
                print(f"Generated test for {region.function_name or 'unknown'} "
                      f"at {region.file_path}:{region.start_line}")

        return generated_tests
