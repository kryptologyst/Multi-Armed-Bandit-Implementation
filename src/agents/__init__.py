"""
Bandit Agents Package.

This package contains implementations of various multi-armed bandit algorithms.
"""

from .bandit_agents import (
    BanditAgent, EpsilonGreedyAgent, DecayingEpsilonGreedyAgent,
    UCBAgent, ThompsonSamplingAgent, SoftmaxAgent, GradientBanditAgent,
    create_agent, BanditStats
)

__all__ = [
    "BanditAgent", "EpsilonGreedyAgent", "DecayingEpsilonGreedyAgent",
    "UCBAgent", "ThompsonSamplingAgent", "SoftmaxAgent", "GradientBanditAgent",
    "create_agent", "BanditStats"
]
