"""
Multi-Armed Bandit Environment Implementation.

This module provides a modern, gymnasium-compatible implementation of the multi-armed bandit problem
with support for various reward distributions and configurations.
"""

from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import logging

logger = logging.getLogger(__name__)


class MultiArmedBanditEnv(gym.Env):
    """
    Multi-Armed Bandit Environment compatible with gymnasium.
    
    This environment simulates a slot machine with multiple arms, each providing
    rewards drawn from different distributions. The agent must learn which arm
    provides the highest expected reward.
    
    Args:
        num_arms: Number of arms in the bandit
        reward_distribution: Type of reward distribution ('gaussian', 'bernoulli', 'uniform')
        reward_params: Parameters for the reward distribution
        seed: Random seed for reproducibility
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(
        self,
        num_arms: int = 10,
        reward_distribution: str = "gaussian",
        reward_params: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        
        self.num_arms = num_arms
        self.reward_distribution = reward_distribution.lower()
        self.reward_params = reward_params or {}
        self.seed = seed
        
        # Set up random number generator
        self.rng = np.random.default_rng(seed)
        
        # Initialize true reward parameters for each arm
        self._initialize_arms()
        
        # Action space: discrete actions corresponding to arm indices
        self.action_space = spaces.Discrete(num_arms)
        
        # Observation space: empty (bandit problem has no state)
        self.observation_space = spaces.Box(
            low=0, high=0, shape=(0,), dtype=np.float32
        )
        
        # Track statistics
        self.total_rewards = 0.0
        self.step_count = 0
        self.arm_counts = np.zeros(num_arms, dtype=int)
        self.arm_rewards = np.zeros(num_arms, dtype=float)
        
        logger.info(f"Initialized MultiArmedBanditEnv with {num_arms} arms")
        if self.reward_distribution == 'gaussian':
            logger.info(f"True arm means: {self.true_means}")
        elif self.reward_distribution == 'bernoulli':
            logger.info(f"True arm probabilities: {self.true_probs}")
        elif self.reward_distribution == 'uniform':
            logger.info(f"True arm ranges: {[(low, high) for low, high in zip(self.true_lows, self.true_highs)]}")
    
    def _initialize_arms(self) -> None:
        """Initialize the true reward parameters for each arm."""
        if self.reward_distribution == "gaussian":
            # Gaussian rewards: N(mean, std^2)
            means = self.reward_params.get("means", None)
            stds = self.reward_params.get("stds", 1.0)
            
            if means is None:
                # Random means from N(0, 1)
                self.true_means = self.rng.normal(0, 1, self.num_arms)
            else:
                self.true_means = np.array(means)
            
            if isinstance(stds, (int, float)):
                self.true_stds = np.full(self.num_arms, stds)
            else:
                self.true_stds = np.array(stds)
                
        elif self.reward_distribution == "bernoulli":
            # Bernoulli rewards: probability of success
            probs = self.reward_params.get("probs", None)
            if probs is None:
                # Random probabilities
                self.true_probs = self.rng.uniform(0.1, 0.9, self.num_arms)
            else:
                self.true_probs = np.array(probs)
                
        elif self.reward_distribution == "uniform":
            # Uniform rewards: U(a, b)
            lows = self.reward_params.get("lows", None)
            highs = self.reward_params.get("highs", None)
            
            if lows is None:
                self.true_lows = self.rng.uniform(-2, 0, self.num_arms)
            else:
                self.true_lows = np.array(lows)
                
            if highs is None:
                self.true_highs = self.rng.uniform(0, 2, self.num_arms)
            else:
                self.true_highs = np.array(highs)
        else:
            raise ValueError(f"Unknown reward distribution: {self.reward_distribution}")
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        # Reset statistics
        self.total_rewards = 0.0
        self.step_count = 0
        self.arm_counts.fill(0)
        self.arm_rewards.fill(0.0)
        
        # Return empty observation (no state in bandit problem)
        return np.array([], dtype=np.float32), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Arm index to pull (0 to num_arms-1)
            
        Returns:
            observation: Empty array (no state)
            reward: Reward from pulling the selected arm
            terminated: Always False (episodic)
            truncated: Always False (no truncation)
            info: Additional information about the step
        """
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        
        # Generate reward based on distribution
        reward = self._generate_reward(action)
        
        # Update statistics
        self.total_rewards += reward
        self.step_count += 1
        self.arm_counts[action] += 1
        self.arm_rewards[action] += reward
        
        # Calculate average reward for this arm
        avg_reward = self.arm_rewards[action] / self.arm_counts[action]
        
        info = {
            "arm_counts": self.arm_counts.copy(),
            "arm_rewards": self.arm_rewards.copy(),
            "arm_averages": self.arm_rewards / np.maximum(self.arm_counts, 1),
            "total_reward": self.total_rewards,
            "step_count": self.step_count,
            "best_arm": np.argmax(self.true_means if self.reward_distribution == "gaussian" 
                                 else self.true_probs if self.reward_distribution == "bernoulli"
                                 else (self.true_lows + self.true_highs) / 2),
        }
        
        return np.array([], dtype=np.float32), reward, False, False, info
    
    def _generate_reward(self, arm: int) -> float:
        """Generate reward for the given arm based on the reward distribution."""
        if self.reward_distribution == "gaussian":
            return float(self.rng.normal(self.true_means[arm], self.true_stds[arm]))
        elif self.reward_distribution == "bernoulli":
            return float(self.rng.binomial(1, self.true_probs[arm]))
        elif self.reward_distribution == "uniform":
            return float(self.rng.uniform(self.true_lows[arm], self.true_highs[arm]))
        else:
            raise ValueError(f"Unknown reward distribution: {self.reward_distribution}")
    
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render the environment."""
        if mode == "human":
            print(f"Step: {self.step_count}, Total Reward: {self.total_rewards:.3f}")
            print(f"Arm counts: {self.arm_counts}")
            print(f"Arm averages: {self.arm_rewards / np.maximum(self.arm_counts, 1)}")
        elif mode == "rgb_array":
            # Return a simple visualization as RGB array
            # This is a placeholder - in practice, you'd create an actual image
            return np.zeros((100, 100, 3), dtype=np.uint8)
        return None
    
    def get_optimal_action(self) -> int:
        """Get the optimal action (arm with highest expected reward)."""
        if self.reward_distribution == "gaussian":
            return int(np.argmax(self.true_means))
        elif self.reward_distribution == "bernoulli":
            return int(np.argmax(self.true_probs))
        elif self.reward_distribution == "uniform":
            return int(np.argmax((self.true_lows + self.true_highs) / 2))
        else:
            raise ValueError(f"Unknown reward distribution: {self.reward_distribution}")
    
    def get_regret(self, action: int) -> float:
        """Calculate regret for a given action."""
        optimal_reward = self._get_optimal_reward()
        actual_reward = self._get_expected_reward(action)
        return optimal_reward - actual_reward
    
    def _get_optimal_reward(self) -> float:
        """Get the expected reward of the optimal arm."""
        if self.reward_distribution == "gaussian":
            return float(np.max(self.true_means))
        elif self.reward_distribution == "bernoulli":
            return float(np.max(self.true_probs))
        elif self.reward_distribution == "uniform":
            return float(np.max((self.true_lows + self.true_highs) / 2))
        else:
            raise ValueError(f"Unknown reward distribution: {self.reward_distribution}")
    
    def _get_expected_reward(self, arm: int) -> float:
        """Get the expected reward for a given arm."""
        if self.reward_distribution == "gaussian":
            return float(self.true_means[arm])
        elif self.reward_distribution == "bernoulli":
            return float(self.true_probs[arm])
        elif self.reward_distribution == "uniform":
            return float((self.true_lows[arm] + self.true_highs[arm]) / 2)
        else:
            raise ValueError(f"Unknown reward distribution: {self.reward_distribution}")
    
    def close(self) -> None:
        """Close the environment."""
        pass
