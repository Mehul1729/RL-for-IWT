import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import wandb
import os
import random
from collections import deque, namedtuple

# --- NEW: Replay Buffer for DQN ---
# DQN learns "off-policy" from past experiences
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# --- NEW: Limbic Agent (DQN) ---
class LimbicAgentDQN(nn.Module):
    """
    A Deep Q-Network (DQN) agent representing the limbic system.
    This agent learns to predict the "value" of an action (Q-value)
    rather than a probability.
    
    When gamma is low, its value estimate will be dominated
    by the immediate reward (r), making it truly impulsive.
    """
    def __init__(self, input_dims: int, n_actions: int, device):
        super(LimbicAgentDQN, self).__init__()
        self.device = device
        self.n_actions = n_actions
        
        # Q-Network: Maps (state) -> (value of each action)
        self.q_net = nn.Sequential(
            nn.Linear(input_dims, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions) # Outputs Q-value for each deck
        )

    def _format_state(self, last_action: int, last_reward: float) -> torch.Tensor:
        """ Formats the state for the network. """
        if last_action == -1:
            action_one_hot = torch.zeros(self.n_actions, device=self.device)
            reward_tensor = torch.zeros(1, device=self.device)
        else:
            action_tensor = torch.tensor(last_action, device=self.device)
            action_one_hot = F.one_hot(action_tensor, num_classes=self.n_actions).float()
            reward_tensor = torch.tensor([last_reward], dtype=torch.float32, device=self.device)
        
        # Scale reward
        reward_tensor = 2 * ((reward_tensor - (-1250)) / (130 - (-1250))) - 1
        
        return torch.cat([action_one_hot, reward_tensor]).unsqueeze(0)

    def act(self, last_action: int, last_reward: float, epsilon: float) -> int:
        """
        Selects an action using an epsilon-greedy policy.
        With probability epsilon, take a random action (explore).
        With probability 1-epsilon, take the best action (exploit).
        """
        # 1. Exploration
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        
        # 2. Exploitation
        with torch.no_grad():
            state = self._format_state(last_action, last_reward)
            # q_net(state) gives Q-values for all actions
            # .max(1) finds the best action (highest Q-value)
            # [1] gets the index of that action
            action = self.q_net(state).max(1)[1].view(1, 1)
            return action.item()

    def get_q_values(self, state_batch):
        """ Get Q-values for all actions in a batch of states. """
        return self.q_net(state_batch)

    def get_next_max_q(self, next_state_batch):
        """ Get the max Q-value for the next state (for Bellman eq.) """
        # .max(1)[0] gets the *value* of the best action
        return self.q_net(next_state_batch).max(1)[0].detach()

# --- Helper function to format batch of transitions ---
def format_batch(transitions, n_actions, device):
    """ Converts a batch of Transition tuples into Tensors for training. """
    batch = Transition(*zip(*transitions))

    state_tensors = []
    next_state_tensors = []

    # Need to manually format states since they aren't simple tensors
    for i in range(len(batch.state)):
        last_action, last_reward = batch.state[i]
        
        # Format state
        if last_action == -1:
            s_action_one_hot = torch.zeros(n_actions, device=device)
            s_reward_tensor = torch.zeros(1, device=device)
        else:
            s_action_tensor = torch.tensor(last_action, device=device)
            s_action_one_hot = F.one_hot(s_action_tensor, num_classes=n_actions).float()
            s_reward_tensor = torch.tensor([last_reward], dtype=torch.float32, device=device)
        s_reward_tensor = 2 * ((s_reward_tensor - (-1250)) / (130 - (-1250))) - 1
        state_tensors.append(torch.cat([s_action_one_hot, s_reward_tensor]))

        # Format next_state
        next_last_action, next_last_reward = batch.next_state[i]
        if next_last_action == -1: # Should not happen, but good to check
            ns_action_one_hot = torch.zeros(n_actions, device=device)
            ns_reward_tensor = torch.zeros(1, device=device)
        else:
            ns_action_tensor = torch.tensor(next_last_action, device=device)
            ns_action_one_hot = F.one_hot(ns_action_tensor, num_classes=n_actions).float()
            ns_reward_tensor = torch.tensor([next_last_reward], dtype=torch.float32, device=device)
        ns_reward_tensor = 2 * ((ns_reward_tensor - (-1250)) / (130 - (-1250))) - 1
        next_state_tensors.append(torch.cat([ns_action_one_hot, ns_reward_tensor]))

    state_batch = torch.stack(state_tensors)
    action_batch = torch.tensor(batch.action, dtype=torch.int64, device=device).unsqueeze(1)
    reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=device)
    next_state_batch = torch.stack(next_state_tensors)

    return state_batch, action_batch, reward_batch, next_state_batch


# --- Driver Code ---
if __name__ == '__main__':
    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Using device: {device} ---")
    
    # --- Environment Selection ---
    print("Which environment do you want to run?")
    print("1: Classic 4-Deck IGT (Recommended)")
    print("2: Complex 6-Deck IGT (Harder)")
    print("3: Extreme 8-Deck IGT (Very Hard)")
    choice = input("Enter 1, 2, or 3: ")

    if choice == '2':
        from complex_env import IowaEnv as GameEnv
        env_name = "Complex-6-Deck"
        n_actions = 6
    elif choice == '3':
        from complex_env import IowaEnv as GameEnv
        env_name = "Extreme-8-Deck"
        n_actions = 8
    else:
        env_name = "Classic-4-Deck"
        n_actions = 4
        try:
            from classic_env import IowaEnv as GameEnv
        except ImportError:
            print("Could not find classic_env.py. Defaulting to complex_env.py")
            from complex_env import IowaEnv as GameEnv
            env_name = "Complex-6-Deck"
            n_actions = 6
    print(f"\n--- Loading {env_name} Environment ---")

    # --- Hyperparameters ---
    n_episodes = 3000
    learning_rate = 0.001
    gm = float(input("Enter gamma (0.0 to 1.0) for the agent (e.g., 0.1 for impulsive, 0.9 for far-sighted):\t"))
    
    BATCH_SIZE = 128
    EPS_START = 0.9 # Start with 90% random actions
    EPS_END = 0.05  # End with 5% random actions
    EPS_DECAY = n_episodes * 50 # How fast to decay epsilon (over ~5000 steps)
    REPLAY_BUFFER_SIZE = 10000
    
    input_dims = n_actions + 1 # one-hot action + 1 reward
    
    # --- Initialization ---
    env = GameEnv(episode_length=100)
    
    # Create the Q-Network
    limbic_agent = LimbicAgentDQN(
        input_dims=input_dims, 
        n_actions=n_actions, 
        device=device
    ).to(device)
    
    optimizer = optim.Adam(limbic_agent.parameters(), lr=learning_rate)
    memory = ReplayBuffer(REPLAY_BUFFER_SIZE)
    
    # --- Wandb Setup ---
    wandb.init(
        project="igt-limbic-pfc-simulation-dqn",
        config={
            "learning_rate": learning_rate,
            "episodes": n_episodes,
            "gamma": gm,
            "architecture": "DQN_64x64",
            "environment": env_name,
            "batch_size": BATCH_SIZE
        }
    )
    wandb.watch(limbic_agent, log="all", log_freq=300)
    
    print(f"--- Training Limbic Agent (DQN) (Gamma: {gm}) ---")
    episode_rewards = []
    all_losses = []
    final_deck_pulls = {i: 0 for i in range(n_actions)}
    total_steps = 0
    
    for i in range(n_episodes):
        env.reset()
        last_action = -1
        last_reward = 0.0
        
        current_state = (last_action, last_reward)
        
        for t in range(env.episode_length):
            # 1. Select Action (Epsilon-Greedy)
            epsilon = EPS_END + (EPS_START - EPS_END) * \
                      np.exp(-1. * total_steps / EPS_DECAY)
            total_steps += 1
            action = limbic_agent.act(last_action, last_reward, epsilon)
            
            # 2. Take Action in Env
            next_env_state, reward, done, info = env.step(action)
            
            # 3. Store transition in Replay Buffer
            next_state = (next_env_state['last_action'], reward)
            memory.push(current_state, action, next_state, reward)
            
            # 4. Update state
            current_state = next_state
            last_action = next_env_state['last_action']
            last_reward = reward

            # 5. --- Optimize Model ---
            if len(memory) < BATCH_SIZE:
                continue # Wait until buffer is full enough
            
            transitions = memory.sample(BATCH_SIZE)
            state_batch, action_batch, reward_batch, next_state_batch = \
                format_batch(transitions, n_actions, device)

            # --- Q-Learning Calculation ---
            
            # Get Q(s, a) for all actions taken
            q_values = limbic_agent.get_q_values(state_batch)
            # Select the Q-value for the *specific action* we took
            current_q_values = q_values.gather(1, action_batch)

            # Get max Q(s', a') for the next state
            next_max_q = limbic_agent.get_next_max_q(next_state_batch)
            
            # Calculate the "target" Q-value: r + gamma * max(Q(s', a'))
            target_q_values = reward_batch + (gm * next_max_q)

            # Loss = (target - current)^2
            loss = F.smooth_l1_loss(current_q_values, target_q_values.unsqueeze(1))
            
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            all_losses.append(loss.item())

        # --- End of Episode Logging ---
        episode_rewards.append(env.cumulative_reward)
        avg_loss = np.mean(all_losses[-100:]) if len(all_losses) > 0 else 0
        
        log_data = {
            "episode": i,
            "cumulative_reward": env.cumulative_reward,
            "q_loss": avg_loss, # Log average loss
            "epsilon": epsilon
        }
        
        # Log Q-values for a test state (e.g., "first turn")
        with torch.no_grad():
            test_state = limbic_agent._format_state(-1, 0.0)
            q_vals = limbic_agent.get_q_values(test_state).squeeze()
            q_log = {f"Q_val_deck_{chr(65+j)}": qv.item() for j, qv in enumerate(q_vals)}
            log_data.update(q_log)
            
        wandb.log(log_data)
        
        if i >= n_episodes - 100:
            for deck, count in env._deck_pull_counts.items():
                final_deck_pulls[deck] += count

        if (i + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {i+1}/{n_episodes}, Avg Reward (last 100): {avg_reward:.2f}, Epsilon: {epsilon:.3f}")
            wandb.log({"average_reward_100_ep": avg_reward, "episode": i})
            
    total_average_reward = np.mean(episode_rewards)
    print(f"\nOverall Average Reward after {n_episodes} episodes: {total_average_reward:.2f} for gamma = {gm}")
    
    print("\n--- Average Deck Pulls (Last 100 Episodes) ---")
    for deck, count in final_deck_pulls.items():
        print(f"Deck {chr(65 + deck)}: {count / 100.0:.2f} pulls/episode")

    print("\n--- Training Complete ---")

    # --- Save Model ---
    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"limbic_dqn_gamma_{gm}_env_{choice}.pth")
    torch.save(limbic_agent.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # --- Plotting ---
    print("\nGenerating training plots...")

    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    window_size = 50
    if n_episodes > window_size:
        avg_rewards = moving_average(episode_rewards, window_size)
    else:
        avg_rewards = episode_rewards
        window_size = 1

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    if n_episodes > window_size:
        ax1.plot(range(window_size - 1, n_episodes), avg_rewards)
    else:
        ax1.plot(avg_rewards)
    ax1.set_title(f"Smoothed Cumulative Reward (Window={window_size}) - Gamma={gm}")
    ax1.set_ylabel("Average Reward")
    ax1.grid(True)

    # Plot 2: Smoothed Q-Loss
    if len(all_losses) > window_size:
        avg_losses = moving_average(all_losses, window_size)
        ax2.plot(range(window_size - 1, len(all_losses)), avg_losses)
    else:
        ax2.plot(all_losses)
    ax2.set_title("Smoothed Q-Loss per Step")
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Avg Q-Loss")
    ax2.grid(True)
    
    plt.tight_layout()
    plot_filename = f"training_plots_dqn_gamma_{gm}.png"
    plt.savefig(plot_filename)
    print(f"Saved training plots to {plot_filename}")

    wandb.finish()
