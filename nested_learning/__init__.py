"""
Nested Learning: The Illusion of Deep Learning Architectures
============================================================

A JAX implementation of the Nested Learning framework from the NeurIPS 2025 paper
by Behrouz, Razaviyayn, Zhong, and Mirrokni (Google Research).

Core idea: A model's architecture and its optimizer are the SAME thing — just
different "levels" of optimization, each with its own information flow and
update speed. Adding a new optimizer level is like adding a new layer.

Modules:
    associative_memory  - Foundation: key-value memory that learns via gradient descent
    deep_optimizer      - Replaces simple momentum with a learned neural network
    self_modifying      - Layers that rewrite their own weights during the forward pass
    continuum_memory    - Multi-frequency memory system (fast, medium, slow updates)
    hope                - The full HOPE model combining everything
    train               - Training utilities for nested optimization
"""

from nested_learning.associative_memory import AssociativeMemory
from nested_learning.deep_optimizer import DeepMomentumGD, MemoryMLP
from nested_learning.self_modifying import SelfModifyingLinear
from nested_learning.continuum_memory import ContinuumMemorySystem
from nested_learning.hope import HOPE, HOPEBlock
