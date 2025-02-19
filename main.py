import numpy as np
from Environment.graphs import create_graphs, plot_rl_vs_baseline_actions
from config.experiments import run_experiment1, run_experiment2, run_experiment3


def main():
    # Candidate hyperparameters: actor LR, critic LR, and discount factor gamma
    '''
    environment steps once per “day,” then setting steps_per_episode=60 means 
    each episode simulates a 60-day hedging window. 
    
    For 200 episodes, effectively run 200 separate 60-day simulations
    '''
    candidate_actor_lrs = [1e-4]
    candidate_critic_lrs = [1e-6]
    candidate_gammas = [0.95]
    episodes = 200
    steps_per_episode = 60

    best_reward = -np.inf
    best_params = None

    # Grid search over actor LR, critic LR, and gamma.
    for actor_lr in candidate_actor_lrs:
        for critic_lr in candidate_critic_lrs:
            for gamma in candidate_gammas:
                print(f"\nTraining with actor_lr={actor_lr}, critic_lr={critic_lr}, gamma={gamma}")
                final_reward = run_experiment3(actor_lr, critic_lr, episodes, steps_per_episode, gamma)
                if final_reward > best_reward:
                    best_reward = final_reward
                    best_params = (actor_lr, critic_lr, gamma)
    
    print("\n=== Grid Search Complete ===")
    print(f"Best LR parameters: Actor LR = {best_params[0]}, Critic LR = {best_params[1]}, Gamma = {best_params[2]}")
    print(f"Final test reward with best params: {best_reward}")

    # Also run the cross-correlation comparison experiment:
    # compare_correlation_settings()

    create_graphs()
    print("Graphs created.")
    print("Plotting RL vs. baseline actions...")
    plot_rl_vs_baseline_actions()
    print("Done.")

if __name__ == "__main__":
    main()
