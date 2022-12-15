"""
Class for general experiment argument_parsing.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MultiRunArguments:
    """
    Arguments pertaining to running multiple experiments.
    """
    base_output_dir: str = field(
        default=None,
        metadata={"help": "Base directory to output experiment results to."}
    )
    personalize_targets: str = field(
        default=None,
        metadata={"help": "Path to file with names of target values for personalization (e.g., specific user names)."
                          "If None, will personalize based on all values present in test data."}
    )
