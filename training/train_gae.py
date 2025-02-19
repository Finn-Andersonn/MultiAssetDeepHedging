import torch
import torch.optim as optim
import numpy as np
from models.policy_network import gaussian_log_prob
from config.utilities import oce_utility
from training.replay_buffer import ReplayBuffer, compute_gae
from config.black_scholes import compute_baseline_pnl, compute_delta_action
import pandas as pd
import torch.nn as nn

def train_deep_hedging_gae(
    env,
    actor,              # GaussianPolicyActor
    critic,             # RiskAverseCritic
    lambd=1.0,          # risk aversion for OCE
    utility_type="exp",
    gamma=0.99,         # discount factor for RL
    lam_gae=0.95,       # lambda for GAE
    episodes=200,       # train for more episodes
    steps_per_episode=60,  # horizon: more steps each episode
    k_epochs=5,         # how many epochs of update we do per rollout
    batch_size=64,      # mini-batch size for updating actor/critic
    lr_actor=3e-4,      # bigger or smaller learning rates as needed
    lr_critic=3e-4,
    max_grad_norm=0.5
):
    """
    This version does on-policy GAE-based advantage estimates.
    For each episode:
      1) Roll out the entire episode (T steps).
      2) Compute GAE advantage for each time step.
      3) Perform multiple epochs of mini-batch gradient updates (actor & critic).

    We also store logs for analysis: step-level logs, episode-level rewards, etc.
    """

    # Create optimizers
    actor_optim = optim.Adam(actor.parameters(), lr=lr_actor)
    critic_optim = optim.Adam(critic.parameters(), lr=lr_critic)

    # Logging containers
    episode_rewards = []
    baseline_pnls = []
    actor_losses = []
    critic_losses = []
    step_logs = []

    # For each training episode:
    for ep in range(episodes):

        # 1) Roll out a single episode, storing transitions in arrays
        states      = []
        actions     = []
        rewards     = []
        dones       = []
        values      = []
        logprobs    = []  # If you want to do something like PPO clipping, for instance

        # Reset environment
        state = env.reset()  # (z_0, m_0)
        z_t, m_t = state

        # Initialize actor hidden states
        hx, cx = actor.init_hidden(batch_size=1)

        # Evaluate value at initial state (for GAE we need V_0 as well)
        s_torch = torch.tensor(np.concatenate([z_t, m_t]), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            v_0 = critic(s_torch).item()
        values.append(v_0)

        episode_return = 0.0

        for step_i in range(steps_per_episode):
            # Actor forward
            s_torch = torch.tensor(np.concatenate([z_t, m_t]), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                mean, logstd, hx_next, cx_next = actor(s_torch, hx, cx)
                std = torch.exp(logstd)
                eps = torch.randn_like(std)
                action_torch = mean + std * eps
                value_t = critic(s_torch)

            # Convert to numpy
            action_np = action_torch[0].numpy()
            value_np = value_t.item()

            # Step environment
            next_state, reward, done, info = env.step(action_np)
            z_tp1, m_tp1 = next_state

            episode_return += reward

            # LOG step
            step_info = {
                "episode": ep,
                "step": step_i,
                "reward": reward,
                "spot_asset0": m_t[0],
                "spot_asset1": m_t[2],
                "spot_asset2": m_t[4],
                "daily_pnl": info.get("daily_pnl", 0.0),
                "cost_val": info.get("cost_val", 0.0)
            }
            for a_idx, val_act in enumerate(action_np):
                step_info[f"action_{a_idx}"] = val_act
            step_logs.append(step_info)

            baseline_action_np = compute_delta_action(m_t, env)

            for i, val_baseline in enumerate(baseline_action_np):
                step_info[f"baseline_action_{i}"] = val_baseline
            step_logs.append(step_info)
            
            # Store transition
            states.append(np.concatenate([z_t, m_t]))
            actions.append(action_np)
            rewards.append(reward)
            dones.append(float(done))
            values.append(value_np)

            # Prepare next iteration
            hx, cx = hx_next, cx_next
            z_t, m_t = z_tp1, m_tp1

            if done:
                # For GAE we still need v_{t+1}, which we have as the last entry in "values".
                break

        episode_rewards.append(episode_return)

        # Compute baseline on the same path
        baseline_pnl = compute_baseline_pnl(env)
        baseline_pnls.append(baseline_pnl)

        # 2) GAE advantage
        # We had T steps, so "rewards" is length T, "values" is length T+1
        # The final value is from the last state => values[-1].
        adv, ret = compute_gae(
            rewards=np.array(rewards, dtype=np.float32),
            values=np.array(values, dtype=np.float32),
            dones=np.array(dones, dtype=np.float32),
            gamma=gamma, lam=lam_gae
        )

        scale_factor = 0.001
        ret *= scale_factor
        adv *= scale_factor

        # Convert everything to Tensors
        states_tensor  = torch.tensor(np.array(states,  dtype=np.float32))
        actions_tensor = torch.tensor(np.array(actions, dtype=np.float32))
        returns_tensor = torch.tensor(ret,   dtype=torch.float32)
        adv_tensor     = torch.tensor(adv,   dtype=torch.float32)
        # For numerical stability, often we normalize advantage:
        adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)

        # 3) Multiple epochs of mini‐batch gradient updates
        T = len(states_tensor)
        num_minibatches = T // batch_size if T > batch_size else 1

        for _ in range(k_epochs):
            # Shuffle indices
            idxs = np.arange(T)
            np.random.shuffle(idxs)

            for start in range(0, T, batch_size):
                end = start + batch_size
                batch_idx = idxs[start:end]

                s_b = states_tensor[batch_idx]
                a_b = actions_tensor[batch_idx]
                ret_b = returns_tensor[batch_idx]
                adv_b = adv_tensor[batch_idx]

                # Critic forward
                # normalize s_b if desired
                s_mean = s_b.mean(dim=0, keepdim=True)
                s_std  = s_b.std(dim=0, keepdim=True) + 1e-8
                s_b_norm = (s_b - s_mean) / s_std

                v_pred = critic(s_b_norm).view(-1)
                critic_loss = (v_pred - ret_b).pow(2).mean()

                critic_optim.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
                critic_optim.step()

                # Actor forward
                # We do a feed to actor for each sample, but let's do a single batch pass
                # We'll create dummy hx/cx for each sample since we do not unroll LSTM over the batch
                hx_b = torch.zeros(len(batch_idx), actor.hidden_size)
                cx_b = torch.zeros(len(batch_idx), actor.hidden_size)

                mean_b, logstd_b, _, _ = actor(s_b, hx_b, cx_b)
                logp_b = gaussian_log_prob(a_b, mean_b, logstd_b)
                actor_loss = - (adv_b.detach() * logp_b).mean()

                actor_optim.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
                actor_optim.step()

                critic_losses.append(critic_loss.item())
                actor_losses.append(actor_loss.item())

        # End of episode logging
        if ep % 20 == 0:
            print(f"Episode {ep} | Return={episode_return:.2f}, Baseline={baseline_pnl:.2f}, Diff={episode_return - baseline_pnl:.2f}")

    # Done with all training
    final_test_reward = episode_rewards[-1]

    print("\nTraining completed.")
    print(f"Final episode reward: {final_test_reward:.4f}")

    # Save logs
    df_rewards = pd.DataFrame({
        "episode": np.arange(len(episode_rewards)),
        "episode_reward": episode_rewards,
        "baseline_pnl": baseline_pnls
    })
    df_rewards.to_csv("/Users/finn/Desktop/UCL Masters/Advanced ML/Project/Code/graphs/episode_rewards.csv", index=False)

    df_losses = pd.DataFrame({
        "actor_loss": actor_losses,
        "critic_loss": critic_losses
    })
    df_losses.to_csv("/Users/finn/Desktop/UCL Masters/Advanced ML/Project/Code/graphs/losses.csv", index=False)

    df_steps = pd.DataFrame(step_logs)
    df_steps.to_csv("/Users/finn/Desktop/UCL Masters/Advanced ML/Project/Code/graphs/step_logs.csv", index=False)

    return final_test_reward
