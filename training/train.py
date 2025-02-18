##############################################################################
# DeepHedgingTrainer.py
# ---------------------------------------------------------------------------
# A fully worked-out training loop that:
#  1) Instantiates our multi-asset environment with cross-asset correlation
#  2) Uses an LSTM-based actor, risk-averse critic
#  3) Repeatedly collects transitions from the environment and updates 
#     (actor, critic) in a minimal manner.
##############################################################################

import torch
import torch.optim as optim
import numpy as np
from models.policy_network import gaussian_log_prob
from config.utilities import oce_utility
from training.replay_buffer import ReplayBuffer
from config.black_scholes import compute_baseline_pnl

def train_deep_hedging(env,
                       actor,        # GaussianPolicyActor
                       critic,       # RiskAverseCritic
                       lambd=1.0,    # risk aversion
                       utility_type="exp", 
                       gamma=0.99,
                       episodes=10,
                       steps_per_episode=10,
                       batch_size=32,
                       lr_actor=1e-4,
                       lr_critic=1e-4):
    """
    We run environment rollouts, store transitions in a replay buffer, 
    then do advantage-based updates in mini-batches.

     advantage_i = [ U(gamma * V(s'_i) + r_i) - V(s_i) ]
     actor_loss = - mean( advantage_i * log_prob(a_i | s_i) )
     critic_loss = MSE( V(s_i), U(gamma * V(s'_i) + r_i) )

    The actor outputs a mean and log_std for a Gaussian; we sample an action.
    """
    # define separate optimizers using the passed learning rates:
    actor_optim  = optim.Adam(actor.parameters(),  lr=lr_actor)
    critic_optim = optim.Adam(critic.parameters(), lr=lr_critic)

    memory = ReplayBuffer(capacity=100000)

    episode_raw_rewards = []

    for ep in range(episodes):
        # reset environment 
        state = env.reset() 
        z_t, m_t = state
        # flatten => shape [1, state_dim]
        s = torch.tensor(np.concatenate([z_t, m_t]), dtype=torch.float).unsqueeze(0)

        # init LSTM hidden states
        hx, cx = actor.init_hidden(batch_size=1)

        raw_rewards = []
        pnl_values = []
        cost_values = []

        done = False
        episode_return = 0.0

        for step_i in range(steps_per_episode):
            with torch.no_grad():
                mean, logstd, hx_next, cx_next = actor(s, hx, cx)
                # sample from Normal(mean, exp(logstd))
                std = torch.exp(logstd)
                eps = torch.randn_like(std)
                action = mean + std * eps  # shape [1, action_dim]
            
            action_np = action[0].numpy()
            next_state, reward, done, info = env.step(action_np)
            episode_return += reward

            raw_rewards.append(reward)
            pnl_values.append(info.get("daily_pnl", 0.0))
            cost_values.append(info.get("cost_val", 0.0))

            # print every 10 steps
            #if step_i % 20 == 0:
                #cost_this_step = info["cost_val"]
                #pnl_this_step = info["daily_pnl"]
                #print(f"Step {step_i} => reward={reward}, cost={cost_this_step}, pnl={pnl_this_step}")
            
            z_tp1, m_tp1 = next_state
            s_next = torch.tensor(np.concatenate([z_tp1, m_tp1]), dtype=torch.float).unsqueeze(0)

            # store transition in replay buffer
            memory.push(s, action, reward, s_next, done, hx, cx, hx_next, cx_next)

            # move to next state
            s = s_next
            hx, cx = hx_next, cx_next

            if done:
                break
        
        #print(f"Episode {ep} finished => return={episode_return} collected {step_i+1} steps")

        if ep % 20 == 0:
            avg_raw_reward = np.mean(raw_rewards)
            std_raw_reward = np.std(raw_rewards)
            avg_pnl = np.mean(pnl_values)
            avg_cost = np.mean(cost_values)
            baseline_pnl = compute_baseline_pnl(env)
            print(f"Episode {ep} finished => return={episode_return:.2f} (steps={step_i+1})")
            print(f"   Avg Raw Reward: {avg_raw_reward:.2f}, Std: {std_raw_reward:.2f}")
            print(f"   Avg Daily PnL: {avg_pnl:.2f}, Avg Cost: {avg_cost:.2f}")
            print(f"   Baseline replication PnL: {baseline_pnl:.2f}")
            print(f"   Difference (Deep - Baseline): {episode_return - baseline_pnl:.2f}")
        
        episode_raw_rewards.append(episode_return)

        print(f"Replay buffer size: {len(memory)}")
        # --- Mini-batch updates ---
        updates = 5
        for i in range(updates):
            if len(memory) < batch_size:
                break

            s_list, a_list, r_list, s_next_list, d_list, hx_list, cx_list, hx_next_list, cx_next_list = memory.sample(batch_size)
            s_batch = torch.cat(s_list, dim=0)
            a_batch = torch.cat(a_list, dim=0)
            r_batch = torch.tensor(r_list, dtype=torch.float)  # shape [batch_size]
            s_next_batch = torch.cat(s_next_list, dim=0)

            s_mean = s_batch.mean(dim=0, keepdim=True)
            s_std  = s_batch.std(dim=0, keepdim=True) + 1e-8
            s_batch_norm = (s_batch - s_mean) / s_std

            s_next_mean = s_next_batch.mean(dim=0, keepdim=True)
            s_next_std  = s_next_batch.std(dim=0, keepdim=True) + 1e-8
            s_next_batch_norm = (s_next_batch - s_next_mean) / s_next_std
            
            # Reward normalization (using batch statistics)
            r_mean = r_batch.mean()
            r_std  = r_batch.std()
            normalized_r = (r_batch - r_mean) / (r_std + 1e-8)

            with torch.no_grad():
                v_next = critic(s_next_batch)  # shape [batch_size]
                target = oce_utility(gamma * v_next + normalized_r, lambd, utility_type)

            v_pred = critic(s_batch_norm)
            critic_loss = ((v_pred - target)**2).mean()
            critic_optim.zero_grad()
            critic_loss.backward()
            # Apply gradient clipping for critic:
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
            critic_optim.step()

            advantage = target - v_pred
            # Use a dummy hidden state for the actor’s forward pass in the update
            hx_dummy = torch.zeros(batch_size, actor.hidden_size)
            cx_dummy = torch.zeros(batch_size, actor.hidden_size)
            mean, logstd, _, _ = actor(s_batch, hx_dummy, cx_dummy)
            logp = gaussian_log_prob(a_batch, mean, logstd)
            actor_loss = - (advantage.detach() * logp).mean()

            actor_optim.zero_grad()
            actor_loss.backward()
            # Apply gradient clipping for actor:
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
            actor_optim.step()

            # Print every 10 updates
            if i % 5 == 0:
                print(f"Update {i}")
                critic_grad_norm = sum(p.grad.norm().item() for p in critic.parameters() if p.grad is not None)
                print(f"   Critic gradient norm: {critic_grad_norm:.2f}")
                actor_grad_norm = sum(p.grad.norm().item() for p in actor.parameters() if p.grad is not None)
                print(f"   Actor gradient norm: {actor_grad_norm:.2f}")
            
                print(f"Critic loss={critic_loss.item()}, Actor loss={actor_loss.item()}")

        memory.clear()

    print("Done with multi-asset deep hedging training using mini-batch, advantage, and log-prob.")
    print("Done with training.")
    final_test_reward = episode_raw_rewards[-1]
    print(f"Final test reward: {final_test_reward:.4f}")
    return final_test_reward
