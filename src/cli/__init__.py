"""
Command-line interfaces for GPN-1.

Exports:
    - train: Training CLI entry point
    - evaluate: Evaluation CLI
    - visualize: Sample generation CLI
"""

# CLI modules are imported at runtime to avoid circular dependencies
__all__ = [
    "train",
    "evaluate",
    "visualize",
]
