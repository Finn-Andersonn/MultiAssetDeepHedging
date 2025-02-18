import numpy as np
import torch
from Environment.base_env import MultiAssetHedgingEnv
from training.train import train_deep_hedging
from models.policy_network import GaussianPolicyActor
from models.RiskAverseCritic import RiskAverseCritic

def run_experiment(actor_lr, critic_lr, episodes, steps_per_episode, gamma):
    # Create the environment
    env = MultiAssetHedgingEnv(
        d_portfolio=12,
        d_market=6,
        n_hedges=12,
        max_steps=25,
        use_corr_model=True,
        transaction_cost_fn=lambda a, m: 0.001 * np.sum(np.abs(a)),
        seed=42
    )
    # Note: state_dim = d_portfolio + d_market = 12 + 6 = 18
    # and we set action_dim = n_hedges = 12.
    state_dim = 18
    action_dim = 12

    # Create actor and critic
    actor = GaussianPolicyActor(state_dim=state_dim, action_dim=action_dim, hidden_size=32)
    critic = RiskAverseCritic(state_dim=state_dim, hidden_size=32)

    # Train the agent with the given hyperparameters.
    train_deep_hedging(env,
                       actor=actor,
                       critic=critic,
                       lambd=0.5,
                       utility_type="exp",
                       gamma=gamma,
                       episodes=episodes,
                       steps_per_episode=steps_per_episode,
                       batch_size=32,
                       lr_actor=actor_lr,
                       lr_critic=critic_lr)

    # Final test run: run one rollout using the trained actor.
    state = env.reset()
    z_t, m_t = state
    s = torch.tensor(np.concatenate([z_t, m_t]), dtype=torch.float).unsqueeze(0)
    hx, cx = actor.init_hidden(batch_size=1)
    done = False
    total_reward = 0.0
    steps = 0
    while not done and steps < env.max_steps:
        with torch.no_grad():
            mean, _, hx_next, cx_next = actor(s, hx, cx)
            # For evaluation, we use the deterministic mean action.
            action = mean  
        action_np = action[0].numpy()
        next_state, r, done, _ = env.step(action_np)
        total_reward += r
        z_tp1, m_tp1 = next_state
        s = torch.tensor(np.concatenate([z_tp1, m_tp1]), dtype=torch.float).unsqueeze(0)
        hx, cx = hx_next, cx_next
        steps += 1

    return total_reward

def main():
    # Candidate hyperparameters: actor LR, critic LR, and discount factor gamma
    candidate_actor_lrs = [1e-3, 5e-4, 1e-4]
    candidate_critic_lrs = [1e-3, 5e-4, 1e-4]
    candidate_gammas = [0.95, 0.98, 0.99]
    episodes = 30
    steps_per_episode = 20

    best_reward = -np.inf
    best_params = None

    # Grid search over actor LR, critic LR, and gamma.
    for actor_lr in candidate_actor_lrs:
        for critic_lr in candidate_critic_lrs:
            for gamma in candidate_gammas:
                print(f"\nTraining with actor_lr={actor_lr}, critic_lr={critic_lr}, gamma={gamma}")
                final_reward = run_experiment(actor_lr, critic_lr, episodes, steps_per_episode, gamma)
                print(f"Final test reward: {final_reward}")
                if final_reward > best_reward:
                    best_reward = final_reward
                    best_params = (actor_lr, critic_lr, gamma)
    
    print("\n=== Grid Search Complete ===")
    print(f"Best LR parameters: Actor LR = {best_params[0]}, Critic LR = {best_params[1]}, Gamma = {best_params[2]}")
    print(f"Final test reward with best params: {best_reward}")

    # Also run the cross-correlation comparison experiment:
    compare_correlation_settings()

def compare_correlation_settings():
    print("\n\nComparing policies under different correlation settings...")
    # 1) Define correlation matrix #1
    corr_2N_1 = [
        [1.0, 0.2, 0.1, 0.0],
        [0.2, 1.0, 0.05, 0.25],
        [0.1, 0.05, 1.0, 0.3],
        [0.0, 0.25, 0.3, 1.0]
    ]
    # Build environment 1
    env1 = MultiAssetHedgingEnv(
        d_portfolio=12,
        d_market=6,
        n_hedges=12,
        max_steps=15,
        use_corr_model=True,
        transaction_cost_fn=lambda a, m: 0.001 * np.sum(np.abs(a)),
        seed=42
    )
    env1.bates_model.corr_2N = np.array(corr_2N_1, dtype=float)
    env1.bates_model.L = np.linalg.cholesky(env1.bates_model.corr_2N)

    state_dim = 18
    action_dim = 12

    actor1 = GaussianPolicyActor(state_dim=state_dim, action_dim=action_dim, hidden_size=32)
    critic1 = RiskAverseCritic(state_dim=state_dim, hidden_size=64)

    train_deep_hedging(env1,
                       actor=actor1,
                       critic=critic1,
                       lambd=1.0,
                       utility_type="exp",
                       gamma=0.99,
                       episodes=10,
                       steps_per_episode=15,
                       batch_size=16)

    # 2) Define correlation matrix #2 (no cross-correlation)
    corr_2N_2 = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ]
    env2 = MultiAssetHedgingEnv(
        d_portfolio=12,
        d_market=6,
        n_hedges=12,
        max_steps=15,
        use_corr_model=True,
        transaction_cost_fn=lambda a, m: 0.001 * np.sum(np.abs(a)),
        seed=42
    )
    env2.bates_model.corr_2N = np.array(corr_2N_2, dtype=float)
    env2.bates_model.L = np.linalg.cholesky(env2.bates_model.corr_2N)

    actor2 = GaussianPolicyActor(state_dim=state_dim, action_dim=action_dim, hidden_size=32)
    critic2 = RiskAverseCritic(state_dim=state_dim, hidden_size=64)

    train_deep_hedging(env2,
                       actor=actor2,
                       critic=critic2,
                       lambd=1.0,
                       utility_type="exp",
                       gamma=0.99,
                       episodes=10,
                       steps_per_episode=15,
                       batch_size=16)

    # 3) Evaluate both policies on the same initial state from env1
    test_state = env1.reset()
    z_t, m_t = test_state
    s = torch.tensor(np.concatenate([z_t, m_t]), dtype=torch.float).unsqueeze(0)
    hx1, cx1 = actor1.init_hidden(batch_size=1)
    hx2, cx2 = actor2.init_hidden(batch_size=1)

    with torch.no_grad():
        mean1, _, _, _ = actor1(s, hx1, cx1)
        action1 = mean1[0].numpy()
        mean2, _, _, _ = actor2(s, hx2, cx2)
        action2 = mean2[0].numpy()

    print("\nComparison of final policies on the same initial state:")
    print("Action under correlation #1:", action1)
    print("Action under correlation #2:", action2)

if __name__ == "__main__":
    main()
