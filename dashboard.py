"""
Streamlit Dashboard for Multi-Armed Bandit Visualization.

This module provides an interactive web interface for running experiments
and visualizing results in real-time.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import sys
import time

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.envs.bandit_env import MultiArmedBanditEnv
from src.agents.bandit_agents import create_agent
from src.trainer import BanditTrainer


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Multi-Armed Bandit Dashboard",
        page_icon="🎰",
        layout="wide"
    )
    
    st.title("🎰 Multi-Armed Bandit Dashboard")
    st.markdown("Interactive visualization and comparison of bandit algorithms")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Environment settings
    st.sidebar.subheader("Environment")
    num_arms = st.sidebar.slider("Number of Arms", 2, 20, 10)
    reward_distribution = st.sidebar.selectbox(
        "Reward Distribution",
        ["gaussian", "bernoulli", "uniform"]
    )
    seed = st.sidebar.number_input("Random Seed", value=42, min_value=0)
    
    # Training settings
    st.sidebar.subheader("Training")
    num_steps = st.sidebar.slider("Number of Steps", 100, 5000, 1000)
    num_runs = st.sidebar.slider("Number of Runs", 1, 20, 5)
    
    # Agent selection
    st.sidebar.subheader("Agents")
    available_agents = [
        "epsilon_greedy", "decaying_epsilon_greedy", "ucb", 
        "thompson_sampling", "softmax", "gradient"
    ]
    selected_agents = st.sidebar.multiselect(
        "Select Agents to Compare",
        available_agents,
        default=["epsilon_greedy", "ucb", "thompson_sampling"]
    )
    
    # Agent parameters
    st.sidebar.subheader("Agent Parameters")
    epsilon = st.sidebar.slider("Epsilon (ε-greedy)", 0.01, 0.5, 0.1, 0.01)
    confidence_level = st.sidebar.slider("Confidence Level (UCB)", 0.5, 5.0, 2.0, 0.1)
    temperature = st.sidebar.slider("Temperature (Softmax)", 0.1, 5.0, 1.0, 0.1)
    learning_rate = st.sidebar.slider("Learning Rate (Gradient)", 0.01, 1.0, 0.1, 0.01)
    
    # Main content
    if st.button("🚀 Run Experiment", type="primary"):
        if not selected_agents:
            st.error("Please select at least one agent!")
            return
        
        # Create environment
        env = MultiArmedBanditEnv(
            num_arms=num_arms,
            reward_distribution=reward_distribution,
            seed=seed
        )
        
        # Create agents
        agents = {}
        agent_params = {
            'epsilon_greedy': {'epsilon': epsilon},
            'decaying_epsilon_greedy': {'initial_epsilon': epsilon},
            'ucb': {'confidence_level': confidence_level},
            'thompson_sampling': {},
            'softmax': {'temperature': temperature},
            'gradient': {'learning_rate': learning_rate}
        }
        
        for agent_name in selected_agents:
            params = agent_params.get(agent_name, {})
            agents[agent_name] = create_agent(
                agent_name, 
                num_arms=num_arms, 
                seed=seed,
                **params
            )
        
        # Create trainer
        trainer = BanditTrainer(seed=seed)
        
        # Run experiment
        with st.spinner("Running experiment..."):
            start_time = time.time()
            
            if num_runs == 1:
                # Single run
                results = {}
                for agent_name, agent in agents.items():
                    result = trainer.train_single_agent(
                        agent, env, num_steps, agent_name
                    )
                    results[agent_name] = result
            else:
                # Multiple runs
                results = trainer.compare_agents(
                    agents, env, num_steps, num_runs
                )
            
            training_time = time.time() - start_time
        
        st.success(f"Experiment completed in {training_time:.2f} seconds!")
        
        # Display results
        display_results(results, num_runs > 1)
        
        # Download results
        if st.button("📥 Download Results"):
            import json
            results_json = trainer._make_serializable(results)
            st.download_button(
                label="Download JSON",
                data=json.dumps(results_json, indent=2),
                file_name=f"bandit_results_{int(time.time())}.json",
                mime="application/json"
            )


def display_results(results, is_multi_run=False):
    """Display experiment results."""
    st.header("📊 Results")
    
    # Summary statistics
    st.subheader("Summary Statistics")
    
    if is_multi_run:
        # Multi-run results
        summary_data = []
        for agent_name, agent_results in results.items():
            summary_data.append({
                'Agent': agent_name,
                'Avg Reward': f"{agent_results['avg_reward_mean']:.3f} ± {agent_results['avg_reward_std']:.3f}",
                'Total Regret': f"{agent_results['total_regret_mean']:.3f} ± {agent_results['total_regret_std']:.3f}",
                'Training Time': f"{agent_results['training_time_mean']:.2f}s"
            })
        
        df = pd.DataFrame(summary_data)
        st.dataframe(df, use_container_width=True)
        
        # Plot results
        plot_multi_run_results(results)
        
    else:
        # Single run results
        summary_data = []
        for agent_name, agent_results in results.items():
            summary_data.append({
                'Agent': agent_name,
                'Avg Reward': f"{agent_results['avg_reward']:.3f}",
                'Total Regret': f"{agent_results['total_regret']:.3f}",
                'Training Time': f"{agent_results['training_time']:.2f}s"
            })
        
        df = pd.DataFrame(summary_data)
        st.dataframe(df, use_container_width=True)
        
        # Plot results
        plot_single_run_results(results)


def plot_multi_run_results(results):
    """Plot multi-run results."""
    st.subheader("📈 Performance Plots")
    
    # Create figure with subplots
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
    
    ax1.set_title('Cumulative Rewards')
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
    
    ax2.set_title('Cumulative Regrets')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Cumulative Regret')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Average Rewards Comparison
    ax3 = axes[1, 0]
    agent_names = list(results.keys())
    avg_rewards = [results[name]['avg_reward_mean'] for name in agent_names]
    avg_reward_stds = [results[name]['avg_reward_std'] for name in agent_names]
    
    bars = ax3.bar(agent_names, avg_rewards, yerr=avg_reward_stds, 
                  capsize=5, alpha=0.7)
    ax3.set_title('Average Rewards Comparison')
    ax3.set_ylabel('Average Reward')
    ax3.tick_params(axis='x', rotation=45)
    
    # Plot 4: Total Regrets Comparison
    ax4 = axes[1, 1]
    total_regrets = [results[name]['total_regret_mean'] for name in agent_names]
    total_regret_stds = [results[name]['total_regret_std'] for name in agent_names]
    
    bars = ax4.bar(agent_names, total_regrets, yerr=total_regret_stds, 
                  capsize=5, alpha=0.7, color='red')
    ax4.set_title('Total Regrets Comparison')
    ax4.set_ylabel('Total Regret')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    st.pyplot(fig)


def plot_single_run_results(results):
    """Plot single run results."""
    st.subheader("📈 Performance Plots")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Multi-Armed Bandit Algorithm Comparison', fontsize=16)
    
    # Plot 1: Cumulative Rewards
    ax1 = axes[0, 0]
    for agent_name, agent_results in results.items():
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
    avg_rewards = [results[name]['avg_reward'] for name in agent_names]
    
    bars = ax3.bar(agent_names, avg_rewards, alpha=0.7)
    ax3.set_title('Average Rewards Comparison')
    ax3.set_ylabel('Average Reward')
    ax3.tick_params(axis='x', rotation=45)
    
    # Plot 4: Total Regrets Comparison
    ax4 = axes[1, 1]
    total_regrets = [results[name]['total_regret'] for name in agent_names]
    
    bars = ax4.bar(agent_names, total_regrets, alpha=0.7, color='red')
    ax4.set_title('Total Regrets Comparison')
    ax4.set_ylabel('Total Regret')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    st.pyplot(fig)


if __name__ == "__main__":
    main()
