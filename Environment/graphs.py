import pandas as pd
import matplotlib.pyplot as plt

def create_graphs():
    df_rewards = pd.read_csv("/Users/finn/Desktop/UCL Masters/Advanced ML/Project/Code/graphs/episode_rewards.csv")


    plt.plot(df_rewards["episode"], df_rewards["episode_reward"], label="RL Hedging PnL")
    plt.plot(df_rewards["episode"], df_rewards["baseline_pnl"], label="BS Delta-Hedge PnL")

    plt.xlabel("Episode")
    plt.ylabel("PnL")
    plt.title("RL Hedging vs. Baseline over Episodes")
    plt.legend()
    plt.grid(True)
    plt.savefig("/Users/finn/Desktop/UCL Masters/Advanced ML/Project/Code/graphs/rl_vs_bs.png")
    plt.show()


    df_losses = pd.read_csv("/Users/finn/Desktop/UCL Masters/Advanced ML/Project/Code/graphs/losses.csv")

    # Each row is one update step
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 8), sharex=True)
    ax_actor, ax_critic = axes

    ax_actor.plot(df_losses["actor_loss"], label="Actor Loss", color="blue")
    ax_actor.set_ylabel("Actor Loss")
    ax_actor.set_title("Actor Loss")
    ax_actor.grid(True)
    ax_actor.legend()

    ax_critic.plot(df_losses["critic_loss"], label="Critic Loss", color="orange")
    ax_critic.set_xlabel("Update Step")
    ax_critic.set_ylabel("Critic Loss")
    ax_critic.set_title("Critic Loss")
    ax_critic.grid(True)
    ax_critic.legend()

    plt.suptitle("Actor & Critic Losses (Separate Subplots)")
    plt.tight_layout()
    plt.savefig("/Users/finn/Desktop/UCL Masters/Advanced ML/Project/Code/graphs/losses.png")
    plt.show()

    fig, ax1 = plt.subplots(figsize=(8,5))
    ax2 = ax1.twinx()  # second y-axis

    ax1.plot(df_losses["actor_loss"], label="Actor Loss", color="blue")
    ax1.set_ylabel("Actor Loss")
    ax1.set_xlabel("Update Step")
    ax1.grid(True)

    ax2.plot(df_losses["critic_loss"], label="Critic Loss", color="orange")
    ax2.set_ylabel("Critic Loss")

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.title("Actor & Critic Losses (Dual Y-Axis)")
    plt.show()


    df_steps = pd.read_csv("/Users/finn/Desktop/UCL Masters/Advanced ML/Project/Code/graphs/step_logs.csv")

    # For example, pick one episode, e.g. episode == 0
    episode_idx = 0
    df_epi = df_steps[df_steps["episode"] == episode_idx]

    fig, ax1 = plt.subplots(figsize=(10,6))

    # Plot BTC (Asset 0) on the left y-axis
    color_btc = "tab:blue"
    ax1.set_xlabel("Step (Day)")
    ax1.set_ylabel("BTC Price (Asset 0)", color=color_btc)
    line_btc = ax1.plot(df_epi["step"], df_epi["spot_asset0"], color=color_btc, label="BTC")[0]
    ax1.tick_params(axis='y', labelcolor=color_btc)

    # Create a second y-axis for ETH and LTC
    ax2 = ax1.twinx()  
    color_eth = "tab:orange"
    color_ltc = "tab:green"
    ax2.set_ylabel("ETH & LTC Price (Assets 1 & 2)")

    line_eth = ax2.plot(df_epi["step"], df_epi["spot_asset1"], color=color_eth, label="ETH")[0]
    line_ltc = ax2.plot(df_epi["step"], df_epi["spot_asset2"], color=color_ltc, label="LTC")[0]

    # Combine the legend entries from both axes
    lines = [line_btc, line_eth, line_ltc]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best")

    plt.title(f"Spot Prices with Dual Y-Axis (Episode {episode_idx})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("/Users/finn/Desktop/UCL Masters/Advanced ML/Project/Code/graphs/spot_prices.png")
    plt.show()


    # Comparing all 12 actions
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(12, 8), sharex=True)
    axes = axes.flatten()  # Flatten subplots to iterate easily

    for i in range(12):
        ax = axes[i]
        col_name = f"action_{i}"
        ax.plot(df_epi["step"], df_epi[col_name], label=f"Action {i}")
        ax.set_title(f"Action {i}")
        ax.set_xlabel("Time Step (Day)")
        ax.set_ylabel("Units of Instrument")
        ax.grid(True)

    plt.suptitle(f"Hedge Actions for Episode {episode_idx}")
    plt.tight_layout()
    plt.savefig("/Users/finn/Desktop/UCL Masters/Advanced ML/Project/Code/graphs/hedge_actions.png")
    plt.show()


def plot_rl_vs_baseline_actions(episode_idx=0, step_logs_path="/Users/finn/Desktop/UCL Masters/Advanced ML/Project/Code/graphs/step_logs.csv"):
    df_steps = pd.read_csv(step_logs_path)
    df_epi = df_steps[df_steps["episode"] == episode_idx]

    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(12, 8), sharex=True)
    axes = axes.flatten()  # Flatten subplots to iterate easily

    for i in range(12):
        ax = axes[i]
        col_rl   = f"action_{i}"
        col_base = f"baseline_action_{i}"

        if col_rl not in df_epi.columns or col_base not in df_epi.columns:
            print(f"Missing RL or baseline columns for instrument {i}.")
            continue

        ax.plot(df_epi["step"], df_epi[col_rl], label="RL Action", color="blue")
        ax.plot(df_epi["step"], df_epi[col_base], label="Baseline", color="orange", linestyle="--")
        ax.set_title(f"Instrument {i}")
        ax.set_xlabel("Time Step (Day)")
        ax.set_ylabel("Units of Instrument")
        ax.grid(True)
        ax.legend()

    plt.suptitle(f"RL vs. Baseline Actions - Episode {episode_idx}")
    plt.tight_layout()
    plt.savefig("/Users/finn/Desktop/UCL Masters/Advanced ML/Project/Code/graphs/RL_vs_Baseline Actions.png")
    plt.show()
