#!/usr/bin/env python3
"""
Demonstration script for the Multi-Armed Bandit implementation.

This script showcases the key features of the modernized bandit framework
including multiple algorithms, comprehensive evaluation, and visualization.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.envs.bandit_env import MultiArmedBanditEnv
from src.agents.bandit_agents import create_agent
from src.trainer import BanditTrainer


def main():
    """Run a comprehensive demonstration of the bandit framework."""
    print("🎰 Multi-Armed Bandit Framework Demonstration")
    print("=" * 50)
    
    # Create environment
    print("\n1. Creating Environment")
    print("-" * 20)
    env = MultiArmedBanditEnv(
        num_arms=10,
        reward_distribution='gaussian',
        reward_params={'stds': 1.0},
        seed=42
    )
    print(f"✓ Created {env.num_arms}-armed Gaussian bandit")
    print(f"✓ Optimal arm: {env.get_optimal_action()}")
    print(f"✓ Optimal reward: {env._get_optimal_reward():.3f}")
    
    # Create multiple agents
    print("\n2. Creating Agents")
    print("-" * 20)
    agents = {
        'Epsilon-Greedy (ε=0.1)': create_agent('epsilon_greedy', num_arms=10, epsilon=0.1),
        'UCB (c=2.0)': create_agent('ucb', num_arms=10, confidence_level=2.0),
        'Thompson Sampling': create_agent('thompson_sampling', num_arms=10),
        'Softmax (τ=1.0)': create_agent('softmax', num_arms=10, temperature=1.0),
    }
    
    for name, agent in agents.items():
        print(f"✓ Created {name}")
    
    # Create trainer
    print("\n3. Training and Evaluation")
    print("-" * 20)
    trainer = BanditTrainer(seed=42)
    
    # Run comparison
    print("Running comparison over 5 independent runs...")
    results = trainer.compare_agents(agents, env, num_steps=1000, num_runs=5)
    
    # Display results
    print("\n4. Results Summary")
    print("-" * 20)
    print(f"{'Algorithm':<20} {'Avg Reward':<15} {'Total Regret':<15}")
    print("-" * 50)
    
    for agent_name, agent_results in results.items():
        avg_reward = agent_results['avg_reward_mean']
        avg_regret = agent_results['total_regret_mean']
        print(f"{agent_name:<20} {avg_reward:<15.3f} {avg_regret:<15.3f}")
    
    # Create visualizations
    print("\n5. Creating Visualizations")
    print("-" * 20)
    print("Generating comprehensive plots...")
    
    # Set up plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Multi-Armed Bandit Algorithm Comparison', fontsize=16)
    
    # Plot 1: Cumulative Rewards
    ax1 = axes[0, 0]
    for agent_name, agent_results in results.items():
        runs_data = agent_results["runs"]
        cumulative_rewards = np.array([r["cumulative_rewards"] for r in runs_data])
        mean_rewards = np.mean(cumulative_rewards, axis=0)
        std_rewards = np.std(cumulative_rewards, axis=0)
        
        steps = np.arange(len(mean_rewards))
        ax1.plot(steps, mean_rewards, label=agent_name, linewidth=2)
        ax1.fill_between(steps, mean_rewards - std_rewards, 
                       mean_rewards + std_rewards, alpha=0.3)
    
    ax1.set_title('Cumulative Rewards Over Time')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Cumulative Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cumulative Regrets
    ax2 = axes[0, 1]
    for agent_name, agent_results in results.items():
        runs_data = agent_results["runs"]
        cumulative_regrets = np.array([r["cumulative_regrets"] for r in runs_data])
        mean_regrets = np.mean(cumulative_regrets, axis=0)
        std_regrets = np.std(cumulative_regrets, axis=0)
        
        steps = np.arange(len(mean_regrets))
        ax2.plot(steps, mean_regrets, label=agent_name, linewidth=2)
        ax2.fill_between(steps, mean_regrets - std_regrets, 
                       mean_regrets + std_regrets, alpha=0.3)
    
    ax2.set_title('Cumulative Regrets Over Time')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Cumulative Regret')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Final Performance Comparison
    ax3 = axes[1, 0]
    agent_names = list(results.keys())
    avg_rewards = [results[name]['avg_reward_mean'] for name in agent_names]
    avg_reward_stds = [results[name]['avg_reward_std'] for name in agent_names]
    
    bars = ax3.bar(range(len(agent_names)), avg_rewards, yerr=avg_reward_stds, 
                  capsize=5, alpha=0.7)
    ax3.set_title('Average Rewards Comparison')
    ax3.set_ylabel('Average Reward')
    ax3.set_xticks(range(len(agent_names)))
    ax3.set_xticklabels(agent_names, rotation=45, ha='right')
    
    # Plot 4: Regret Comparison
    ax4 = axes[1, 1]
    total_regrets = [results[name]['total_regret_mean'] for name in agent_names]
    total_regret_stds = [results[name]['total_regret_std'] for name in agent_names]
    
    bars = ax4.bar(range(len(agent_names)), total_regrets, yerr=total_regret_stds, 
                  capsize=5, alpha=0.7, color='red')
    ax4.set_title('Total Regrets Comparison')
    ax4.set_ylabel('Total Regret')
    ax4.set_xticks(range(len(agent_names)))
    ax4.set_xticklabels(agent_names, rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = "bandit_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plots saved to {plot_path}")
    
    # Show plot
    plt.show()
    
    # Save results
    print("\n6. Saving Results")
    print("-" * 20)
    results_path = "demonstration_results.json"
    trainer.save_results(results_path)
    print(f"✓ Results saved to {results_path}")
    
    print("\n🎉 Demonstration completed successfully!")
    print("\nKey Features Demonstrated:")
    print("• Multiple bandit algorithms (Epsilon-Greedy, UCB, Thompson Sampling, Softmax)")
    print("• Statistical evaluation over multiple runs")
    print("• Comprehensive visualization with confidence intervals")
    print("• Modern Python architecture with type hints and logging")
    print("• Reproducible results with seed management")
    
    print("\nNext Steps:")
    print("• Try different reward distributions: python cli.py train --reward-distribution bernoulli")
    print("• Compare more algorithms: python cli.py compare --agents epsilon_greedy ucb thompson_sampling")
    print("• Use the Streamlit dashboard: streamlit run dashboard.py")
    print("• Run unit tests: python -m pytest tests/")


if __name__ == "__main__":
    main()
