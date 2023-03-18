"""
Utility functions.

Authors:
Justin Butler & Garrett Toews
"""
import torch
import logging


def check_device():
    """
    A utility function to determine GPU/CPU device type based on individual computer hardware.

    Returns: The pytorch device to be used during runtime.

    """
    # For MacOSX M1 GPU
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        # this ensures that the current MacOS version is at least 12.3+
        # and ensures that the current current PyTorch installation was built with MPS activated.
        device = torch.device("mps")
        logging.info("Using MPS GPU")
    # cuda for NVIDIA GPU
    elif torch.cuda.is_available():
        device = torch.device('cuda:0')
        logging.info("Using CUDA GPU")
    else:
        device = torch.device('cpu')
        logging.info("Using CPU")
    return device
