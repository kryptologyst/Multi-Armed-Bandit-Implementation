"""
Training and Evaluation Module for Multi-Armed Bandit Algorithms.

This module provides comprehensive training, evaluation, and comparison
functionality for bandit algorithms with extensive logging and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import logging
import time
from pathlib import Path
import json
import yaml

from .envs.bandit_env import MultiArmedBanditEnv
from .agents.bandit_agents import BanditAgent, create_agent, BanditStats

logger = logging.getLogger(__name__)


class BanditTrainer:
    """
    Trainer class for multi-armed bandit algorithms.
    
    Provides comprehensive training, evaluation, and comparison functionality
    with logging, visualization, and checkpointing capabilities.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        log_dir: Optional[str] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            config: Configuration dictionary
            log_dir: Directory for logging results
            seed: Random seed for reproducibility
        """
        self.config = config or {}
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        self.seed = seed
        
        # Set up logging
        self._setup_logging()
        
        # Results storage
        self.results: Dict[str, Any] = {}
        
        logger.info("Initialized BanditTrainer")
    
    def _setup_logging(self) -> None:
        """Set up logging configuration."""
        log_file = self.log_dir / "training.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def train_single_agent(
        self,
        agent: BanditAgent,
        env: MultiArmedBanditEnv,
        num_steps: int,
        agent_name: str = "agent"
    ) -> Dict[str, Any]:
        """
        Train a single agent on the environment.
        
        Args:
            agent: Bandit agent to train
            env: Multi-armed bandit environment
            num_steps: Number of training steps
            agent_name: Name for logging purposes
            
        Returns:
            Dictionary containing training results
        """
        logger.info(f"Training {agent_name} for {num_steps} steps")
        
        # Reset environment and agent
        env.reset(seed=self.seed)
        
        # Storage for results
        rewards = np.zeros(num_steps)
        regrets = np.zeros(num_steps)
        actions = np.zeros(num_steps, dtype=int)
        cumulative_rewards = np.zeros(num_steps)
        cumulative_regrets = np.zeros(num_steps)
        
        start_time = time.time()
        
        for step in range(num_steps):
            # Select action
            action = agent.select_action()
            
            # Take step in environment
            _, reward, _, _, info = env.step(action)
            
            # Update agent
            agent.update(action, reward)
            
            # Calculate regret
            optimal_action = env.get_optimal_action()
            optimal_reward = env._get_expected_reward(optimal_action)
            actual_reward = env._get_expected_reward(action)
            regret = optimal_reward - actual_reward
            
            # Store results
            rewards[step] = reward
            regrets[step] = regret
            actions[step] = action
            cumulative_rewards[step] = np.sum(rewards[:step+1])
            cumulative_regrets[step] = np.sum(regrets[:step+1])
        
        training_time = time.time() - start_time
        
        # Calculate final statistics
        final_stats = agent.get_stats()
        avg_reward = np.mean(rewards)
        total_regret = np.sum(regrets)
        
        results = {
            "agent_name": agent_name,
            "rewards": rewards,
            "regrets": regrets,
            "actions": actions,
            "cumulative_rewards": cumulative_rewards,
            "cumulative_regrets": cumulative_regrets,
            "final_stats": final_stats,
            "avg_reward": avg_reward,
            "total_regret": total_regret,
            "training_time": training_time,
            "num_steps": num_steps,
            "optimal_action": env.get_optimal_action(),
            "optimal_reward": env._get_optimal_reward(),
        }
        
        logger.info(f"{agent_name} training completed in {training_time:.2f}s")
        logger.info(f"Average reward: {avg_reward:.3f}, Total regret: {total_regret:.3f}")
        
        return results
    
    def compare_agents(
        self,
        agents: Dict[str, BanditAgent],
        env: MultiArmedBanditEnv,
        num_steps: int,
        num_runs: int = 1
    ) -> Dict[str, Any]:
        """
        Compare multiple agents on the same environment.
        
        Args:
            agents: Dictionary mapping agent names to agent instances
            env: Multi-armed bandit environment
            num_steps: Number of training steps
            num_runs: Number of independent runs for statistical significance
            
        Returns:
            Dictionary containing comparison results
        """
        logger.info(f"Comparing {len(agents)} agents over {num_runs} runs")
        
        all_results = {}
        
        for run in range(num_runs):
            logger.info(f"Starting run {run + 1}/{num_runs}")
            
            run_results = {}
            for agent_name, agent in agents.items():
                # Create a fresh copy of the agent for each run
                # Map class names to agent types
                class_name = agent.__class__.__name__.lower()
                agent_type_map = {
                    'epsilongreedyagent': 'epsilon_greedy',
                    'decayingepsilongreedyagent': 'decaying_epsilon_greedy',
                    'ucbagent': 'ucb',
                    'thompsonsamplingagent': 'thompson_sampling',
                    'softmaxagent': 'softmax',
                    'gradientbanditagent': 'gradient'
                }
                agent_type = agent_type_map.get(class_name, class_name.replace('agent', ''))
                
                agent_copy = create_agent(
                    agent_type,
                    env.num_arms,
                    **{k: v for k, v in agent.__dict__.items() 
                       if k not in ['num_arms', 'rng', 'stats', 'alpha', 'beta', 'preferences']}
                )
                
                results = self.train_single_agent(
                    agent_copy, env, num_steps, f"{agent_name}_run_{run}"
                )
                run_results[agent_name] = results
            
            all_results[f"run_{run}"] = run_results
        
        # Aggregate results across runs
        aggregated_results = self._aggregate_results(all_results)
        
        self.results = aggregated_results
        return aggregated_results
    
    def _aggregate_results(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate results across multiple runs."""
        agent_names = list(next(iter(all_results.values())).keys())
        aggregated = {}
        
        for agent_name in agent_names:
            agent_results = [run_results[agent_name] for run_results in all_results.values()]
            
            # Aggregate metrics
            avg_rewards = np.array([r["avg_reward"] for r in agent_results])
            total_regrets = np.array([r["total_regret"] for r in agent_results])
            training_times = np.array([r["training_time"] for r in agent_results])
            
            # Calculate statistics
            aggregated[agent_name] = {
                "avg_reward_mean": np.mean(avg_rewards),
                "avg_reward_std": np.std(avg_rewards),
                "total_regret_mean": np.mean(total_regrets),
                "total_regret_std": np.std(total_regrets),
                "training_time_mean": np.mean(training_times),
                "training_time_std": np.std(training_times),
                "runs": agent_results,
            }
        
        return aggregated
    
    def plot_results(
        self,
        results: Optional[Dict[str, Any]] = None,
        save_path: Optional[str] = None,
        show_plot: bool = True
    ) -> None:
        """
        Plot training results with comprehensive visualizations.
        
        Args:
            results: Results dictionary (uses self.results if None)
            save_path: Path to save plots
            show_plot: Whether to display plots
        """
        if results is None:
            results = self.results
        
        if not results:
            logger.warning("No results to plot")
            return
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Multi-Armed Bandit Algorithm Comparison', fontsize=16)
        
        # Plot 1: Cumulative Rewards
        ax1 = axes[0, 0]
        for agent_name, agent_results in results.items():
            if "runs" in agent_results:
                # Multiple runs - plot mean and std
                runs_data = agent_results["runs"]
                cumulative_rewards = np.array([r["cumulative_rewards"] for r in runs_data])
                mean_rewards = np.mean(cumulative_rewards, axis=0)
                std_rewards = np.std(cumulative_rewards, axis=0)
                
                steps = np.arange(len(mean_rewards))
                ax1.plot(steps, mean_rewards, label=agent_name, linewidth=2)
                ax1.fill_between(steps, mean_rewards - std_rewards, 
                               mean_rewards + std_rewards, alpha=0.3)
            else:
                # Single run
                ax1.plot(agent_results["cumulative_rewards"], 
                        label=agent_name, linewidth=2)
        
        ax1.set_title('Cumulative Rewards')
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Cumulative Reward')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Cumulative Regrets
        ax2 = axes[0, 1]
        for agent_name, agent_results in results.items():
            if "runs" in agent_results:
                runs_data = agent_results["runs"]
                cumulative_regrets = np.array([r["cumulative_regrets"] for r in runs_data])
                mean_regrets = np.mean(cumulative_regrets, axis=0)
                std_regrets = np.std(cumulative_regrets, axis=0)
                
                steps = np.arange(len(mean_regrets))
                ax2.plot(steps, mean_regrets, label=agent_name, linewidth=2)
                ax2.fill_between(steps, mean_regrets - std_regrets, 
                               mean_regrets + std_regrets, alpha=0.3)
            else:
                ax2.plot(agent_results["cumulative_regrets"], 
                        label=agent_name, linewidth=2)
        
        ax2.set_title('Cumulative Regrets')
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Cumulative Regret')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Average Rewards Comparison
        ax3 = axes[1, 0]
        agent_names = list(results.keys())
        avg_rewards = []
        avg_reward_stds = []
        
        for agent_name in agent_names:
            agent_results = results[agent_name]
            if "avg_reward_mean" in agent_results:
                avg_rewards.append(agent_results["avg_reward_mean"])
                avg_reward_stds.append(agent_results["avg_reward_std"])
            else:
                avg_rewards.append(agent_results["avg_reward"])
                avg_reward_stds.append(0)
        
        bars = ax3.bar(agent_names, avg_rewards, yerr=avg_reward_stds, 
                      capsize=5, alpha=0.7)
        ax3.set_title('Average Rewards Comparison')
        ax3.set_ylabel('Average Reward')
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, reward in zip(bars, avg_rewards):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{reward:.3f}', ha='center', va='bottom')
        
        # Plot 4: Total Regrets Comparison
        ax4 = axes[1, 1]
        total_regrets = []
        total_regret_stds = []
        
        for agent_name in agent_names:
            agent_results = results[agent_name]
            if "total_regret_mean" in agent_results:
                total_regrets.append(agent_results["total_regret_mean"])
                total_regret_stds.append(agent_results["total_regret_std"])
            else:
                total_regrets.append(agent_results["total_regret"])
                total_regret_stds.append(0)
        
        bars = ax4.bar(agent_names, total_regrets, yerr=total_regret_stds, 
                      capsize=5, alpha=0.7, color='red')
        ax4.set_title('Total Regrets Comparison')
        ax4.set_ylabel('Total Regret')
        ax4.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, regret in zip(bars, total_regrets):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{regret:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plots saved to {save_path}")
        
        if show_plot:
            plt.show()
    
    def save_results(self, filepath: str) -> None:
        """Save results to JSON file."""
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = self._make_serializable(self.results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert numpy arrays to lists for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif hasattr(obj, '__dict__'):
            # Handle custom objects like BanditStats
            return {k: self._make_serializable(v) for k, v in obj.__dict__.items()}
        else:
            return obj
    
    def load_config(self, config_path: str) -> None:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
    
    def save_config(self, config_path: str) -> None:
        """Save current configuration to YAML file."""
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
        logger.info(f"Configuration saved to {config_path}")
