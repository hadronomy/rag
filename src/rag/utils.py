"""Utility functions for the RAG application."""

import multiprocessing
import os
import warnings


def setup_multiprocessing():
    """
    Setup multiprocessing to prevent semaphore leaks.

    This function should be called early in the application lifecycle
    to ensure proper multiprocessing configuration.
    """
    # Set the start method to 'spawn' which is more compatible
    # with CUDA and prevents semaphore leaks
    if multiprocessing.get_start_method(allow_none=True) != "spawn":
        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            # Start method has already been set, which is fine
            pass

    # Suppress specific multiprocessing warnings
    warnings.filterwarnings(
        "ignore", category=UserWarning, module="multiprocessing.resource_tracker"
    )

    # Set environment variables to prevent CUDA multiprocessing issues
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def cleanup_torch_resources():
    """Clean up PyTorch resources to prevent memory leaks."""
    import gc

    import torch

    # Force garbage collection
    gc.collect()

    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Clear all CUDA streams
        for device_id in range(torch.cuda.device_count()):
            with torch.cuda.device(device_id):
                torch.cuda.empty_cache()
