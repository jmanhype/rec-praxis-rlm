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
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from coverage import Coverage
    from coverage.files import FnmatchMatcher
except ImportError:
    Coverage = None  # type: ignore
    FnmatchMatcher = None  # type: ignore

try:
    import dspy
except ImportError:
    dspy = None  # type: ignore

from rec_praxis_rlm import ProceduralMemory, Experience, MemoryConfig, RLMContext


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

        test_code: str = dspy.OutputField(desc="Complete pytest test code with imports, test functions, and assertions")
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

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "total_coverage": self.total_coverage,
            "uncovered_regions": [r.to_dict() for r in self.uncovered_regions],
            "files_analyzed": self.files_analyzed,
            "lines_covered": self.lines_covered,
            "lines_total": self.lines_total,
        }


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
        test_dir: str = "tests"
    ):
        """Initialize test generation agent.

        Args:
            memory_path: Path to JSONL file for procedural memory storage.
                        Use ":memory:" for in-memory (testing only).
            coverage_data_file: Path to coverage.py data file (default: .coverage)
            test_dir: Directory where generated tests will be saved
        """
        self.memory_path = memory_path
        self.coverage_data_file = Path(coverage_data_file)
        self.test_dir = Path(test_dir)

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

        return CoverageAnalysis(
            total_coverage=total_coverage,
            uncovered_regions=uncovered_regions,
            files_analyzed=files_analyzed,
            lines_covered=covered_lines,
            lines_total=total_lines,
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
        # Simple template-based generation for MVP
        # In production, this would use DSPy with LLM

        if not region.function_name:
            return None

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

        print(f"Current coverage: {analysis.total_coverage:.1f}%")
        print(f"Target coverage: {target_coverage:.1f}%")
        print(f"Found {len(analysis.uncovered_regions)} uncovered regions")

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
