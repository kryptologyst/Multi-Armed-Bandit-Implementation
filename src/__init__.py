"""
Multi-Armed Bandit Implementation Package.

A comprehensive implementation of multi-armed bandit algorithms with
modern Python practices, extensive visualization, and comparison tools.
"""

__version__ = "1.0.0"
__author__ = "AI Assistant"
__email__ = "ai@example.com"

from .envs.bandit_env import MultiArmedBanditEnv
from .agents.bandit_agents import (
    BanditAgent, EpsilonGreedyAgent, DecayingEpsilonGreedyAgent,
    UCBAgent, ThompsonSamplingAgent, SoftmaxAgent, GradientBanditAgent,
    create_agent
)
from .trainer import BanditTrainer

__all__ = [
    "MultiArmedBanditEnv",
    "BanditAgent", "EpsilonGreedyAgent", "DecayingEpsilonGreedyAgent",
    "UCBAgent", "ThompsonSamplingAgent", "SoftmaxAgent", "GradientBanditAgent",
    "create_agent", "BanditTrainer"
]
