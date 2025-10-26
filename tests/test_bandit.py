"""
Unit tests for Multi-Armed Bandit implementation.

This module contains comprehensive tests for environments, agents, and trainer components.
"""

import pytest
import numpy as np
from unittest.mock import patch
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.envs.bandit_env import MultiArmedBanditEnv
from src.agents.bandit_agents import (
    EpsilonGreedyAgent, UCBAgent, ThompsonSamplingAgent,
    SoftmaxAgent, GradientBanditAgent, create_agent
)
from src.trainer import BanditTrainer


class TestMultiArmedBanditEnv:
    """Test cases for MultiArmedBanditEnv."""
    
    def test_gaussian_env_creation(self):
        """Test Gaussian bandit environment creation."""
        env = MultiArmedBanditEnv(num_arms=5, reward_distribution='gaussian', seed=42)
        
        assert env.num_arms == 5
        assert env.reward_distribution == 'gaussian'
        assert len(env.true_means) == 5
        assert len(env.true_stds) == 5
        assert env.action_space.n == 5
    
    def test_bernoulli_env_creation(self):
        """Test Bernoulli bandit environment creation."""
        env = MultiArmedBanditEnv(num_arms=3, reward_distribution='bernoulli', seed=42)
        
        assert env.num_arms == 3
        assert env.reward_distribution == 'bernoulli'
        assert len(env.true_probs) == 3
        assert all(0 <= p <= 1 for p in env.true_probs)
    
    def test_uniform_env_creation(self):
        """Test Uniform bandit environment creation."""
        env = MultiArmedBanditEnv(num_arms=4, reward_distribution='uniform', seed=42)
        
        assert env.num_arms == 4
        assert env.reward_distribution == 'uniform'
        assert len(env.true_lows) == 4
        assert len(env.true_highs) == 4
        assert all(low < high for low, high in zip(env.true_lows, env.true_highs))
    
    def test_reset(self):
        """Test environment reset functionality."""
        env = MultiArmedBanditEnv(num_arms=3, seed=42)
        
        # Take some steps
        env.step(0)
        env.step(1)
        
        # Reset
        obs, info = env.reset(seed=42)
        
        assert len(obs) == 0  # Empty observation for bandit
        assert env.step_count == 0
        assert env.total_rewards == 0.0
        assert np.all(env.arm_counts == 0)
    
    def test_step_gaussian(self):
        """Test step function for Gaussian environment."""
        env = MultiArmedBanditEnv(num_arms=2, reward_distribution='gaussian', seed=42)
        
        obs, reward, terminated, truncated, info = env.step(0)
        
        assert len(obs) == 0
        assert isinstance(reward, float)
        assert not terminated
        assert not truncated
        assert 'arm_counts' in info
        assert info['arm_counts'][0] == 1
        assert info['step_count'] == 1
    
    def test_step_bernoulli(self):
        """Test step function for Bernoulli environment."""
        env = MultiArmedBanditEnv(num_arms=2, reward_distribution='bernoulli', seed=42)
        
        obs, reward, terminated, truncated, info = env.step(0)
        
        assert reward in [0, 1]  # Binary reward
        assert info['arm_counts'][0] == 1
    
    def test_get_optimal_action(self):
        """Test optimal action identification."""
        env = MultiArmedBanditEnv(
            num_arms=3, 
            reward_distribution='gaussian',
            reward_params={'means': [1.0, 2.0, 0.5]},
            seed=42
        )
        
        optimal_action = env.get_optimal_action()
        assert optimal_action == 1  # Arm with highest mean
    
    def test_get_regret(self):
        """Test regret calculation."""
        env = MultiArmedBanditEnv(
            num_arms=2,
            reward_distribution='gaussian',
            reward_params={'means': [1.0, 2.0]},
            seed=42
        )
        
        regret_optimal = env.get_regret(1)  # Optimal arm
        regret_suboptimal = env.get_regret(0)  # Suboptimal arm
        
        assert regret_optimal == 0.0
        assert regret_suboptimal == 1.0


class TestBanditAgents:
    """Test cases for bandit agents."""
    
    def test_epsilon_greedy_agent(self):
        """Test epsilon-greedy agent."""
        agent = EpsilonGreedyAgent(num_arms=3, epsilon=0.1, seed=42)
        
        assert agent.num_arms == 3
        assert agent.epsilon == 0.1
        assert len(agent.stats.arm_estimates) == 3
    
    def test_epsilon_greedy_action_selection(self):
        """Test epsilon-greedy action selection."""
        agent = EpsilonGreedyAgent(num_arms=3, epsilon=0.0, seed=42)  # Pure exploitation
        
        # Initially all estimates are 0, so should select arm 0
        action = agent.select_action()
        assert action in [0, 1, 2]
        
        # Update estimates
        agent.update(0, 1.0)
        agent.update(1, 2.0)
        
        # With epsilon=0, should select arm 1 (highest estimate)
        action = agent.select_action()
        assert action == 1
    
    def test_ucb_agent(self):
        """Test UCB agent."""
        agent = UCBAgent(num_arms=3, confidence_level=2.0, seed=42)
        
        assert agent.num_arms == 3
        assert agent.confidence_level == 2.0
    
    def test_ucb_action_selection(self):
        """Test UCB action selection."""
        agent = UCBAgent(num_arms=3, confidence_level=2.0, seed=42)
        
        # First action should be arm 0 (untried arms)
        action = agent.select_action()
        assert action == 0
        
        # Update and test again
        agent.update(0, 1.0)
        action = agent.select_action()
        assert action in [1, 2]  # Should select untried arm
    
    def test_thompson_sampling_agent(self):
        """Test Thompson Sampling agent."""
        agent = ThompsonSamplingAgent(num_arms=3, seed=42)
        
        assert agent.num_arms == 3
        assert len(agent.alpha) == 3
        assert len(agent.beta) == 3
        assert np.all(agent.alpha == 1.0)  # Prior alpha
        assert np.all(agent.beta == 1.0)   # Prior beta
    
    def test_thompson_sampling_update(self):
        """Test Thompson Sampling update."""
        agent = ThompsonSamplingAgent(num_arms=2, seed=42)
        
        # Update with success
        agent.update(0, 1.0)
        assert agent.alpha[0] == 2.0
        assert agent.beta[0] == 1.0
        
        # Update with failure
        agent.update(0, 0.0)
        assert agent.alpha[0] == 2.0
        assert agent.beta[0] == 2.0
    
    def test_softmax_agent(self):
        """Test Softmax agent."""
        agent = SoftmaxAgent(num_arms=3, temperature=1.0, seed=42)
        
        assert agent.num_arms == 3
        assert agent.temperature == 1.0
    
    def test_gradient_agent(self):
        """Test Gradient Bandit agent."""
        agent = GradientBanditAgent(num_arms=3, learning_rate=0.1, seed=42)
        
        assert agent.num_arms == 3
        assert agent.learning_rate == 0.1
        assert len(agent.preferences) == 3
    
    def test_agent_update(self):
        """Test agent update functionality."""
        agent = EpsilonGreedyAgent(num_arms=2, epsilon=0.1, seed=42)
        
        # Initial state
        assert agent.stats.arm_counts[0] == 0
        assert agent.stats.arm_estimates[0] == 0.0
        
        # Update
        agent.update(0, 1.0)
        
        assert agent.stats.arm_counts[0] == 1
        assert agent.stats.arm_estimates[0] == 1.0
        assert agent.stats.step_count == 1
        assert agent.stats.total_rewards == 1.0
    
    def test_create_agent_factory(self):
        """Test agent factory function."""
        agent = create_agent('epsilon_greedy', num_arms=3, epsilon=0.1)
        assert isinstance(agent, EpsilonGreedyAgent)
        
        agent = create_agent('ucb', num_arms=3, confidence_level=2.0)
        assert isinstance(agent, UCBAgent)
        
        with pytest.raises(ValueError):
            create_agent('unknown_agent', num_arms=3)


class TestBanditTrainer:
    """Test cases for BanditTrainer."""
    
    def test_trainer_creation(self):
        """Test trainer creation."""
        trainer = BanditTrainer(seed=42)
        
        assert trainer.seed == 42
        assert trainer.results == {}
    
    def test_single_agent_training(self):
        """Test single agent training."""
        env = MultiArmedBanditEnv(num_arms=3, seed=42)
        agent = EpsilonGreedyAgent(num_arms=3, epsilon=0.1, seed=42)
        trainer = BanditTrainer(seed=42)
        
        results = trainer.train_single_agent(agent, env, num_steps=100, agent_name='test')
        
        assert 'rewards' in results
        assert 'regrets' in results
        assert 'actions' in results
        assert len(results['rewards']) == 100
        assert len(results['regrets']) == 100
        assert len(results['actions']) == 100
        assert results['num_steps'] == 100
        assert results['agent_name'] == 'test'
    
    def test_agent_comparison(self):
        """Test multi-agent comparison."""
        env = MultiArmedBanditEnv(num_arms=3, seed=42)
        agents = {
            'epsilon_greedy': EpsilonGreedyAgent(num_arms=3, epsilon=0.1, seed=42),
            'ucb': UCBAgent(num_arms=3, confidence_level=2.0, seed=42)
        }
        trainer = BanditTrainer(seed=42)
        
        results = trainer.compare_agents(agents, env, num_steps=50, num_runs=2)
        
        assert 'epsilon_greedy' in results
        assert 'ucb' in results
        assert 'avg_reward_mean' in results['epsilon_greedy']
        assert 'total_regret_mean' in results['epsilon_greedy']
    
    def test_results_serialization(self):
        """Test results serialization."""
        trainer = BanditTrainer()
        trainer.results = {
            'test_agent': {
                'avg_reward_mean': 1.5,
                'total_regret_mean': 10.0,
                'runs': [{'rewards': np.array([1.0, 2.0])}]
            }
        }
        
        serializable = trainer._make_serializable(trainer.results)
        
        # Check that numpy arrays are converted to lists
        assert isinstance(serializable['test_agent']['runs'][0]['rewards'], list)
        assert serializable['test_agent']['runs'][0]['rewards'] == [1.0, 2.0]


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_training(self):
        """Test complete training pipeline."""
        # Create environment
        env = MultiArmedBanditEnv(num_arms=5, reward_distribution='gaussian', seed=42)
        
        # Create agent
        agent = EpsilonGreedyAgent(num_arms=5, epsilon=0.1, seed=42)
        
        # Create trainer
        trainer = BanditTrainer(seed=42)
        
        # Train
        results = trainer.train_single_agent(agent, env, num_steps=200, agent_name='test')
        
        # Verify results
        assert results['avg_reward'] > 0
        assert results['total_regret'] >= 0
        assert results['training_time'] > 0
        
        # Verify agent learned something
        assert np.sum(agent.stats.arm_counts) == 200
        assert agent.stats.total_rewards > 0
    
    def test_multiple_algorithms_comparison(self):
        """Test comparison of multiple algorithms."""
        env = MultiArmedBanditEnv(num_arms=4, seed=42)
        
        agents = {
            'epsilon_greedy': EpsilonGreedyAgent(num_arms=4, epsilon=0.1, seed=42),
            'ucb': UCBAgent(num_arms=4, confidence_level=2.0, seed=42),
            'thompson_sampling': ThompsonSamplingAgent(num_arms=4, seed=42)
        }
        
        trainer = BanditTrainer(seed=42)
        results = trainer.compare_agents(agents, env, num_steps=100, num_runs=3)
        
        # Verify all agents are present
        assert len(results) == 3
        assert all(agent_name in results for agent_name in agents.keys())
        
        # Verify statistical measures are present
        for agent_name in results:
            assert 'avg_reward_mean' in results[agent_name]
            assert 'total_regret_mean' in results[agent_name]
            assert 'runs' in results[agent_name]


if __name__ == "__main__":
    pytest.main([__file__])
