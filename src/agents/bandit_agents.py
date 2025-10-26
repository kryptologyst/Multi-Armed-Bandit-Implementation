"""
Multi-Armed Bandit Agent Implementations.

This module provides various bandit algorithms including epsilon-greedy, UCB, 
Thompson Sampling, and more advanced techniques.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BanditStats:
    """Statistics for bandit algorithm performance."""
    total_rewards: float = 0.0
    total_regret: float = 0.0
    step_count: int = 0
    arm_counts: np.ndarray = None
    arm_rewards: np.ndarray = None
    arm_estimates: np.ndarray = None
    
    def __post_init__(self):
        if self.arm_counts is None:
            self.arm_counts = np.array([])
        if self.arm_rewards is None:
            self.arm_rewards = np.array([])
        if self.arm_estimates is None:
            self.arm_estimates = np.array([])


class BanditAgent(ABC):
    """Abstract base class for bandit agents."""
    
    def __init__(self, num_arms: int, seed: Optional[int] = None):
        self.num_arms = num_arms
        self.rng = np.random.default_rng(seed)
        self.stats = BanditStats()
        self.stats.arm_counts = np.zeros(num_arms, dtype=int)
        self.stats.arm_rewards = np.zeros(num_arms, dtype=float)
        self.stats.arm_estimates = np.zeros(num_arms, dtype=float)
    
    @abstractmethod
    def select_action(self) -> int:
        """Select an action (arm) to pull."""
        pass
    
    def update(self, action: int, reward: float) -> None:
        """Update the agent's estimates based on the observed reward."""
        self.stats.step_count += 1
        self.stats.total_rewards += reward
        self.stats.arm_counts[action] += 1
        self.stats.arm_rewards[action] += reward
        
        # Update estimate using incremental average
        n = self.stats.arm_counts[action]
        self.stats.arm_estimates[action] += (1/n) * (reward - self.stats.arm_estimates[action])
    
    def get_stats(self) -> BanditStats:
        """Get current statistics."""
        return self.stats


class EpsilonGreedyAgent(BanditAgent):
    """
    Epsilon-Greedy bandit agent.
    
    With probability epsilon, explores randomly; otherwise exploits the best-known arm.
    
    Args:
        num_arms: Number of arms
        epsilon: Exploration probability
        seed: Random seed
    """
    
    def __init__(self, num_arms: int, epsilon: float = 0.1, seed: Optional[int] = None):
        super().__init__(num_arms, seed)
        self.epsilon = epsilon
        logger.info(f"Initialized EpsilonGreedyAgent with epsilon={epsilon}")
    
    def select_action(self) -> int:
        """Select action using epsilon-greedy strategy."""
        if self.rng.random() < self.epsilon:
            # Explore: choose random arm
            return self.rng.integers(0, self.num_arms)
        else:
            # Exploit: choose best estimated arm
            return int(np.argmax(self.stats.arm_estimates))


class DecayingEpsilonGreedyAgent(EpsilonGreedyAgent):
    """
    Epsilon-Greedy agent with decaying exploration rate.
    
    The exploration rate decreases over time to focus more on exploitation
    as the agent learns.
    
    Args:
        num_arms: Number of arms
        initial_epsilon: Initial exploration probability
        decay_rate: Rate of decay for epsilon
        min_epsilon: Minimum epsilon value
        seed: Random seed
    """
    
    def __init__(
        self, 
        num_arms: int, 
        initial_epsilon: float = 1.0,
        decay_rate: float = 0.99,
        min_epsilon: float = 0.01,
        seed: Optional[int] = None
    ):
        super().__init__(num_arms, initial_epsilon, seed)
        self.initial_epsilon = initial_epsilon
        self.decay_rate = decay_rate
        self.min_epsilon = min_epsilon
        logger.info(f"Initialized DecayingEpsilonGreedyAgent with decay_rate={decay_rate}")
    
    def select_action(self) -> int:
        """Select action using decaying epsilon-greedy strategy."""
        # Update epsilon based on decay
        self.epsilon = max(
            self.min_epsilon,
            self.initial_epsilon * (self.decay_rate ** self.stats.step_count)
        )
        return super().select_action()


class UCBAgent(BanditAgent):
    """
    Upper Confidence Bound (UCB) bandit agent.
    
    Selects arms based on upper confidence bounds, balancing exploration
    and exploitation through confidence intervals.
    
    Args:
        num_arms: Number of arms
        confidence_level: Confidence level for UCB (typically 2.0)
        seed: Random seed
    """
    
    def __init__(self, num_arms: int, confidence_level: float = 2.0, seed: Optional[int] = None):
        super().__init__(num_arms, seed)
        self.confidence_level = confidence_level
        logger.info(f"Initialized UCBAgent with confidence_level={confidence_level}")
    
    def select_action(self) -> int:
        """Select action using UCB strategy."""
        # If any arm hasn't been tried, select it
        untried_arms = np.where(self.stats.arm_counts == 0)[0]
        if len(untried_arms) > 0:
            return int(untried_arms[0])
        
        # Calculate UCB values
        total_pulls = self.stats.step_count
        ucb_values = (
            self.stats.arm_estimates + 
            self.confidence_level * np.sqrt(np.log(total_pulls) / self.stats.arm_counts)
        )
        
        return int(np.argmax(ucb_values))


class ThompsonSamplingAgent(BanditAgent):
    """
    Thompson Sampling bandit agent.
    
    Uses Bayesian inference to sample from posterior distributions
    of arm rewards and select the arm with highest sampled value.
    
    Args:
        num_arms: Number of arms
        prior_alpha: Prior alpha parameter for Beta distribution
        prior_beta: Prior beta parameter for Beta distribution
        seed: Random seed
    """
    
    def __init__(
        self, 
        num_arms: int, 
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        seed: Optional[int] = None
    ):
        super().__init__(num_arms, seed)
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.alpha = np.full(num_arms, prior_alpha)
        self.beta = np.full(num_arms, prior_beta)
        logger.info(f"Initialized ThompsonSamplingAgent with Beta({prior_alpha}, {prior_beta}) priors")
    
    def select_action(self) -> int:
        """Select action using Thompson Sampling."""
        # Sample from Beta distributions
        sampled_values = self.rng.beta(self.alpha, self.beta)
        return int(np.argmax(sampled_values))
    
    def update(self, action: int, reward: float) -> None:
        """Update Beta distribution parameters."""
        super().update(action, reward)
        
        # Update Beta distribution parameters
        if reward > 0:
            self.alpha[action] += 1
        else:
            self.beta[action] += 1


class SoftmaxAgent(BanditAgent):
    """
    Softmax (Boltzmann) bandit agent.
    
    Selects arms based on softmax probabilities derived from estimated rewards.
    Higher temperature leads to more exploration.
    
    Args:
        num_arms: Number of arms
        temperature: Temperature parameter (higher = more exploration)
        seed: Random seed
    """
    
    def __init__(self, num_arms: int, temperature: float = 1.0, seed: Optional[int] = None):
        super().__init__(num_arms, seed)
        self.temperature = temperature
        logger.info(f"Initialized SoftmaxAgent with temperature={temperature}")
    
    def select_action(self) -> int:
        """Select action using softmax strategy."""
        # Calculate softmax probabilities
        exp_values = np.exp(self.stats.arm_estimates / self.temperature)
        probabilities = exp_values / np.sum(exp_values)
        
        # Sample action based on probabilities
        return int(self.rng.choice(self.num_arms, p=probabilities))


class GradientBanditAgent(BanditAgent):
    """
    Gradient Bandit agent.
    
    Uses gradient ascent to update action preferences based on
    the average reward baseline.
    
    Args:
        num_arms: Number of arms
        learning_rate: Learning rate for gradient updates
        seed: Random seed
    """
    
    def __init__(self, num_arms: int, learning_rate: float = 0.1, seed: Optional[int] = None):
        super().__init__(num_arms, seed)
        self.learning_rate = learning_rate
        self.preferences = np.zeros(num_arms)
        self.average_reward = 0.0
        logger.info(f"Initialized GradientBanditAgent with learning_rate={learning_rate}")
    
    def select_action(self) -> int:
        """Select action using gradient bandit strategy."""
        # Calculate softmax probabilities from preferences
        exp_preferences = np.exp(self.preferences - np.max(self.preferences))
        probabilities = exp_preferences / np.sum(exp_preferences)
        
        # Sample action
        return int(self.rng.choice(self.num_arms, p=probabilities))
    
    def update(self, action: int, reward: float) -> None:
        """Update preferences using gradient ascent."""
        super().update(action, reward)
        
        # Update average reward
        self.average_reward = self.stats.total_rewards / self.stats.step_count
        
        # Calculate softmax probabilities
        exp_preferences = np.exp(self.preferences - np.max(self.preferences))
        probabilities = exp_preferences / np.sum(exp_preferences)
        
        # Update preferences
        for arm in range(self.num_arms):
            if arm == action:
                self.preferences[arm] += self.learning_rate * (reward - self.average_reward) * (1 - probabilities[arm])
            else:
                self.preferences[arm] -= self.learning_rate * (reward - self.average_reward) * probabilities[arm]


def create_agent(agent_type: str, num_arms: int, **kwargs) -> BanditAgent:
    """
    Factory function to create bandit agents.
    
    Args:
        agent_type: Type of agent to create
        num_arms: Number of arms
        **kwargs: Additional parameters for the agent
        
    Returns:
        Initialized bandit agent
    """
    agent_map = {
        "epsilon_greedy": EpsilonGreedyAgent,
        "decaying_epsilon_greedy": DecayingEpsilonGreedyAgent,
        "ucb": UCBAgent,
        "thompson_sampling": ThompsonSamplingAgent,
        "softmax": SoftmaxAgent,
        "gradient": GradientBanditAgent,
    }
    
    if agent_type not in agent_map:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    return agent_map[agent_type](num_arms, **kwargs)
