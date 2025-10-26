"""
Command Line Interface for Multi-Armed Bandit Training and Evaluation.

This module provides a comprehensive CLI for running experiments,
comparing algorithms, and visualizing results.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List
import logging

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.envs.bandit_env import MultiArmedBanditEnv
from src.agents.bandit_agents import create_agent
from src.trainer import BanditTrainer


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Multi-Armed Bandit Training and Evaluation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train epsilon-greedy agent
  python cli.py train --agent epsilon_greedy --steps 1000
  
  # Compare multiple agents
  python cli.py compare --agents epsilon_greedy ucb thompson_sampling --steps 2000 --runs 5
  
  # Run with custom environment
  python cli.py train --agent epsilon_greedy --num-arms 20 --reward-distribution bernoulli
  
  # Use configuration file
  python cli.py train --config configs/default.yaml
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a single agent')
    train_parser.add_argument('--agent', type=str, default='epsilon_greedy',
                            choices=['epsilon_greedy', 'decaying_epsilon_greedy', 'ucb', 
                                   'thompson_sampling', 'softmax', 'gradient'],
                            help='Agent type to train')
    train_parser.add_argument('--steps', type=int, default=1000,
                            help='Number of training steps')
    train_parser.add_argument('--num-arms', type=int, default=10,
                            help='Number of arms in the bandit')
    train_parser.add_argument('--reward-distribution', type=str, default='gaussian',
                            choices=['gaussian', 'bernoulli', 'uniform'],
                            help='Reward distribution type')
    train_parser.add_argument('--config', type=str, help='Configuration file path')
    train_parser.add_argument('--log-dir', type=str, default='logs',
                            help='Directory for logging results')
    train_parser.add_argument('--seed', type=int, help='Random seed')
    train_parser.add_argument('--save-results', action='store_true',
                            help='Save results to file')
    train_parser.add_argument('--no-plot', action='store_true',
                            help='Disable plotting')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare multiple agents')
    compare_parser.add_argument('--agents', nargs='+', 
                              default=['epsilon_greedy', 'ucb', 'thompson_sampling'],
                              choices=['epsilon_greedy', 'decaying_epsilon_greedy', 'ucb', 
                                     'thompson_sampling', 'softmax', 'gradient'],
                              help='Agents to compare')
    compare_parser.add_argument('--steps', type=int, default=2000,
                              help='Number of training steps')
    compare_parser.add_argument('--runs', type=int, default=5,
                              help='Number of independent runs')
    compare_parser.add_argument('--num-arms', type=int, default=10,
                              help='Number of arms in the bandit')
    compare_parser.add_argument('--reward-distribution', type=str, default='gaussian',
                              choices=['gaussian', 'bernoulli', 'uniform'],
                              help='Reward distribution type')
    compare_parser.add_argument('--config', type=str, help='Configuration file path')
    compare_parser.add_argument('--log-dir', type=str, default='logs',
                              help='Directory for logging results')
    compare_parser.add_argument('--seed', type=int, help='Random seed')
    compare_parser.add_argument('--save-results', action='store_true',
                              help='Save results to file')
    compare_parser.add_argument('--no-plot', action='store_true',
                              help='Disable plotting')
    
    # Agent-specific parameters
    agent_group = parser.add_argument_group('Agent Parameters')
    agent_group.add_argument('--epsilon', type=float, default=0.1,
                           help='Epsilon for epsilon-greedy agents')
    agent_group.add_argument('--confidence-level', type=float, default=2.0,
                           help='Confidence level for UCB agent')
    agent_group.add_argument('--temperature', type=float, default=1.0,
                           help='Temperature for softmax agent')
    agent_group.add_argument('--learning-rate', type=float, default=0.1,
                           help='Learning rate for gradient agent')
    
    return parser


def create_environment(args: argparse.Namespace) -> MultiArmedBanditEnv:
    """Create bandit environment from command line arguments."""
    env_kwargs = {
        'num_arms': args.num_arms,
        'reward_distribution': args.reward_distribution,
        'seed': args.seed,
    }
    
    # Add reward distribution specific parameters
    if args.reward_distribution == 'gaussian':
        env_kwargs['reward_params'] = {'stds': 1.0}
    elif args.reward_distribution == 'bernoulli':
        env_kwargs['reward_params'] = {}
    elif args.reward_distribution == 'uniform':
        env_kwargs['reward_params'] = {}
    
    return MultiArmedBanditEnv(**env_kwargs)


def create_agents(args: argparse.Namespace, num_arms: int) -> Dict[str, Any]:
    """Create agents from command line arguments."""
    agents = {}
    
    # Handle both single agent (train command) and multiple agents (compare command)
    if hasattr(args, 'agents') and args.agents:
        agent_types = args.agents
    elif hasattr(args, 'agent') and args.agent:
        agent_types = [args.agent]
    else:
        raise ValueError("No agents specified")
    
    for agent_type in agent_types:
        agent_kwargs = {'num_arms': num_arms, 'seed': args.seed}
        
        # Add agent-specific parameters
        if agent_type in ['epsilon_greedy', 'decaying_epsilon_greedy']:
            agent_kwargs['epsilon'] = args.epsilon
        elif agent_type == 'ucb':
            agent_kwargs['confidence_level'] = args.confidence_level
        elif agent_type == 'softmax':
            agent_kwargs['temperature'] = args.temperature
        elif agent_type == 'gradient':
            agent_kwargs['learning_rate'] = args.learning_rate
        
        agents[agent_type] = create_agent(agent_type, **agent_kwargs)
    
    return agents


def run_training(args: argparse.Namespace) -> None:
    """Run single agent training."""
    print(f"Training {args.agent} agent...")
    
    # Create environment
    env = create_environment(args)
    
    # Create agent
    agents = create_agents(args, args.num_arms)
    agent = agents[args.agent]
    
    # Create trainer
    trainer = BanditTrainer(log_dir=args.log_dir, seed=args.seed)
    
    # Train agent
    results = trainer.train_single_agent(agent, env, args.steps, args.agent)
    
    # Plot results
    if not args.no_plot:
        trainer.plot_results({args.agent: results})
    
    # Save results
    if args.save_results:
        results_path = Path(args.log_dir) / f"{args.agent}_results.json"
        trainer.results = {args.agent: results}
        trainer.save_results(str(results_path))
    
    print(f"Training completed!")
    print(f"Average reward: {results['avg_reward']:.3f}")
    print(f"Total regret: {results['total_regret']:.3f}")


def run_comparison(args: argparse.Namespace) -> None:
    """Run multi-agent comparison."""
    print(f"Comparing agents: {', '.join(args.agents)}")
    
    # Create environment
    env = create_environment(args)
    
    # Create agents
    agents = create_agents(args, args.num_arms)
    
    # Create trainer
    trainer = BanditTrainer(log_dir=args.log_dir, seed=args.seed)
    
    # Compare agents
    results = trainer.compare_agents(agents, env, args.steps, args.runs)
    
    # Plot results
    if not args.no_plot:
        trainer.plot_results(results)
    
    # Save results
    if args.save_results:
        results_path = Path(args.log_dir) / "comparison_results.json"
        trainer.save_results(str(results_path))
    
    print("Comparison completed!")
    print("\nResults Summary:")
    print("-" * 50)
    for agent_name, agent_results in results.items():
        print(f"{agent_name}:")
        print(f"  Average Reward: {agent_results['avg_reward_mean']:.3f} ± {agent_results['avg_reward_std']:.3f}")
        print(f"  Total Regret: {agent_results['total_regret_mean']:.3f} ± {agent_results['total_regret_std']:.3f}")
        print()


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        if args.command == 'train':
            run_training(args)
        elif args.command == 'compare':
            run_comparison(args)
        else:
            print(f"Unknown command: {args.command}")
            parser.print_help()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        logging.exception("Unexpected error occurred")
        sys.exit(1)


if __name__ == "__main__":
    main()
