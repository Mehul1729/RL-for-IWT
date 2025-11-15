import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
import numpy as np
import matplotlib.pyplot as plt # Import matplotlib for plotting
import wandb # Import Weights & Biases
import os # Import os for creating directories

# --- Step 2: Limbic System (RL Agent) ---
class LimbicAgent(nn.Module):
    """
    A Reinforcement Learning agent representing the limbic system.
    This agent uses a Policy Gradient method (REINFORCE) to learn. Its decision-making
    is driven by immediate rewards, making it prone to impulsive choices that seem
    good in the short term but may be detrimental in the long run.
    Attributes:
        policy_net (nn.Sequential): The neural network that maps states to action logits.
        optimizer (torch.optim.Adam): The optimizer for training the network.
        rewards (list): A buffer to store rewards received during an episode.
        saved_log_probs (list): A buffer to store the log probability of actions taken.
    """
    
    
    
    # Initializing the Policy NN:
    
    def __init__(self, gm, input_dims: int, n_actions: int, device, learning_rate: float = 0.005):
        """
        Initializes the LimbicAgent.
        Args:
            input_dims (int): The dimensionality of the state representation.
            n_actions (int): The number of possible actions (i.e., decks).
            device (torch.device): The device (CPU or GPU) to run on.
            learning_rate (float): The learning rate for the optimizer.
        """
        super(LimbicAgent, self).__init__()
        self.gm = gm
        self.device = device # Store the device
        
        # Define the policy network (a simple Multi-Layer Perceptron)
        self.policy_net = nn.Sequential(
            nn.Linear(input_dims, 4),
            nn.ReLU(),
            nn.Linear(4, n_actions)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        # Buffers for the REINFORCE algorithm
        self.rewards = [] # list of all the rewards in the episode 
        self.saved_log_probs = [] # This will store tensors now on the device
        self.episode_action_probs = [] # For wandb logging





# this function creates the format of input state to the policy NN :

    def _format_state(self, last_action: int, last_reward: float, n_actions: int) -> torch.Tensor:
        """
        Formats the input for the policy network as specified:
        one-hot encoded last action + previous reward.
        All tensors are created on self.device (GPU if available).
        """
        # For the first turn, there is no previous action or reward.
        if last_action == -1:
            # Create tensors directly on the correct device
            action_one_hot = torch.zeros(n_actions, device=self.device)
            reward_tensor = torch.zeros(1, device=self.device)
            
        else:
            # Create tensors directly on the correct device
            action_tensor = torch.tensor(last_action, device=self.device)
            action_one_hot = F.one_hot(action_tensor, num_classes=n_actions).float()
            reward_tensor = torch.tensor([last_reward], dtype=torch.float32, device=self.device)
        
        # Min-Max scaling of the rewards :
        reward_tensor = 2 * ((reward_tensor - (-1250)) / (130 - (-1250))) - 1
        
        # Concatenate to create the final state vector (will be on device)
        return torch.cat([action_one_hot, reward_tensor]).unsqueeze(0)




# Action fucntion:

    def act(self, last_action: int, last_reward: float, n_actions: int) -> int:
        """
        Selects an action based on the current policy.
        """
        # 1. Format the state for the network
        # state is already on the GPU thanks to _format_state
        state = self._format_state(last_action, last_reward, n_actions)
        
        # 2. Forward pass to get action logits
        # policy_net is on GPU, state is on GPU. This works.
        action_logits = self.policy_net(state)
        
        # 3. Create a probability distribution and sample an action
        action_distribution = Categorical(logits=action_logits)
        action = action_distribution.sample()
        
        # 4. Save log_prob and action_probs for update/logging
        self.saved_log_probs.append(action_distribution.log_prob(action))
        self.episode_action_probs.append(action_distribution.probs) # Log probs for wandb
        
        # .item() moves the scalar value from GPU back to CPU
        return action.item()



# for updating the weights of the policy net :
# this updation code will run for each episode:
# LOSS FUNCTION CALCULATION:
    def update(self, gm):
        """
        Updates the policy network using the REINFORCE algorithm.
        Now ALWAYS normalizes returns to stabilize training.
        Returns loss, mean log prob, and avg action probs for logging.
        """
        if not self.saved_log_probs:
            return 0.0, 0.0, torch.zeros(self.policy_net[-1].out_features) # No actions

        policy_loss = []
        discounted_returns = deque()
        R = 0 # total long term reward 

        # for the given trial, we clauclate the discounted retrun:
        for r in reversed(self.rewards): # for all the rewrds in a given episode 
            R = r + gm * R # bellman equation trick 
            discounted_returns.appendleft(R)
        
        
        # Create the 'returns' tensor directly on the GPU
        returns = torch.tensor(list(discounted_returns), dtype=torch.float32, device=self.device)
        
        # --- *** THE CORRECT FIX *** ---
        # ALWAYS normalize the returns (z-score) to stabilize gradients.
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        else:
            # Handle edge case of 1-step episode
            returns = torch.tensor([0.0], dtype=torch.float32, device=self.device) 
            
        # 3. Calculate the policy loss (all ops on GPU)
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R) # collecting policy loss for each action taken
            
        # --- Data for Logging ---
        # Stack all log_probs and probs tensors (which are on GPU)
        log_probs_tensor = torch.stack(self.saved_log_probs)
        action_probs_tensor = torch.stack(self.episode_action_probs)
        
        mean_log_prob = log_probs_tensor.mean().item()
        avg_episode_probs = action_probs_tensor.mean(dim=0).squeeze().detach() # Avg across episode
            
        # 4. Perform the optimization step
        self.optimizer.zero_grad()
        loss = torch.cat(policy_loss).sum() # loss is a tensor on GPU
        loss.backward() # Gradients are computed on GPU
        self.optimizer.step()
        
        # 5. Clear the episode's data
        self.rewards = []
        self.saved_log_probs = [] # Clear list of GPU tensors
        self.episode_action_probs = []

        return loss.item(), mean_log_prob, avg_episode_probs # Return logs


# --- ** NEW: Function to encapsulate a single training run ** ---
def run_experiment(gm, env_name, n_actions, GameEnv, device, n_episodes, learning_rate, run_name, group_name, run_counter, total_runs):
    """
    Runs a single complete training experiment for a given gamma.
    Initializes, trains, logs to wandb, and saves plots/model.
    """
    print(f"\n--- STARTING RUN {run_counter}/{total_runs} (Gamma: {gm}, ID: {run_name}) ---")
    
    # State is one-hot action + last reward
    input_dims = n_actions + 1
    
    # Initialization
    env = GameEnv(episode_length=100) # 100 trials per episdoe 
    
    # --- Pass device and gamma to agent and move agent to device ---
    limbic_agent = LimbicAgent(
        input_dims=input_dims, 
        n_actions=n_actions, 
        gm=gm, 
        device=device,
        learning_rate=learning_rate
    ).to(device)
    
    # --- Initialize Wandb for this specific run ---
    wandb.init(
        project="RL-IWT",
        config={
            "learning_rate": learning_rate,
            "episodes": n_episodes,
            "gamma": gm,
            "architecture": "SimpleMLP_32", # Kept from original
            "environment": env_name,
            "n_actions": n_actions
        },
        group=group_name,   # --- ** NEW ** --- Group runs by env and gamma
        name=run_name,      # --- ** NEW ** --- Give each run a unique name
        reinit=True         # --- ** NEW ** --- Crucial for running in a loop
    )
    # Watch the model to log gradients
    wandb.watch(limbic_agent, log="all", log_freq=300) # Log gradients every 300 eps
    
    print(f"--- Training Limbic Agent (Gamma: {gm}, Run: {run_name}) ---")
    episode_rewards = []
    policy_losses = [] # List to store losses
    final_deck_pulls = {i: 0 for i in range(n_actions)}
    
    for i in range(n_episodes):
        env_state = env.reset()
        last_action = env_state['last_action']
        last_reward = 0.0
        
        # Run one full episode
        for t in range(env.episode_length):
            # Agent chooses an action
            action = limbic_agent.act(last_action, last_reward, n_actions)
            # Environment responds
            next_env_state, reward, done, info = env.step(action)
            # Store the immediate reward for the agent's update
            limbic_agent.rewards.append(reward)
            # Update state variables for the next turn
            last_action = action
            last_reward = reward
        
        # At the end of the episode, update the agent's policy
        loss, mean_log_prob, avg_probs = limbic_agent.update(gm) # Pass correct gamma
        
        policy_losses.append(loss) # Store the loss
        episode_rewards.append(env.cumulative_reward)
        
        # --- Log to Wandb ---
        log_data = {
            "episode": i,
            "policy_loss": loss,
            "cumulative_reward": env.cumulative_reward,
            "mean_log_prob_episode": mean_log_prob
        }
        # Add average action probabilities to the log
        prob_log = {f"avg_prob_deck_{chr(65+j)}": p.item() for j, p in enumerate(avg_probs)}
        log_data.update(prob_log)
        
        wandb.log(log_data)
        
        
        # Store deck pulls from the *final* episodes to see learned policy
        if i >= n_episodes - 100:
            for deck, count in env._deck_pull_counts.items():
                final_deck_pulls[deck] += count

        if (i + 1) % 300 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Run {run_counter}, Ep {i+1}/{n_episodes}, Avg Reward (100): {avg_reward:.2f}")
            # Log the 100-episode average reward
            wandb.log({"average_reward_100_ep": avg_reward, "episode": i})
            
    total_average_reward = np.mean(episode_rewards)
    print(f"\nRun {run_counter} Overall Avg Reward: {total_average_reward:.2f} (gamma = {gm})")
    
    print("\n--- Average Deck Pulls (Last 100 Episodes) ---")
    for deck, count in final_deck_pulls.items():
        print(f"Deck {chr(65 + deck)}: {count / 100.0:.2f} pulls/episode")

    # --- Save the model ---
    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)
    # --- ** NEW ** --- Make save path unique for this run
    save_path = os.path.join(save_dir, f"limbic_agent_env_{env_name}_gamma_{gm}_{run_name}.pth")
    torch.save(limbic_agent.state_dict(), save_path)
    print(f"Model saved to {save_path}")


    # --- PLOTTING SECTION ---
    print("\nGenerating training plots...")

    def moving_average(data, window_size):
        """Computes the moving average of a list."""
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    # Calculate moving average for rewards
    window_size = 50
    # Handle cases where n_episodes < window_size
    if n_episodes > window_size:
        avg_rewards = moving_average(episode_rewards, window_size)
    else:
        avg_rewards = episode_rewards # Not enough data to smooth
        window_size = 1


    # Create the plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot 1: Smoothed Cumulative Reward
    if n_episodes > window_size:
        ax1.plot(range(window_size - 1, n_episodes), avg_rewards)
    else:
        ax1.plot(avg_rewards)
    # --- ** NEW ** --- Make plot title unique
    ax1.set_title(f"Smoothed Reward (Window={window_size}) - Gamma={gm}, {run_name}")
    ax1.set_ylabel("Average Reward")
    ax1.grid(True)

    # Plot 2: Raw Policy Loss
    ax2.plot(policy_losses)
    ax2.set_title("Raw Policy Loss per Episode")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Loss Value")
    ax2.grid(True)
    
    plt.tight_layout()
    # --- ** NEW ** --- Make plot filename unique
    plot_filename = f"training_plots_env_{env_name}_gamma_{gm}_{run_name}.png"
    plt.savefig(plot_filename)
    print(f"Saved training plots to {plot_filename}")
    
    # --- ** NEW ** --- Close the plot to free memory
    plt.close(fig)

    wandb.finish() # Finish the wandb run
    print(f"--- Finished Run {run_counter}/{total_runs} (Gamma: {gm}) ---")



# --- Driver Code ---
if __name__ == '__main__':
    # --- ** NEW: Set up device (GPU or CPU) ** ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Using device: {device} ---")
    
    # --- Environment Selection (Done ONCE) ---
    print("Which environment do you want to run for all 40 experiments?")
    print("1: Classic 4-Deck IGT (Recommended)")
    print("2: Complex 6-Deck IGT (Harder)")
    print("3: Extreme 8-Deck IGT (Very Hard)")
    choice = input("Enter 1, 2, or 3: ")

    if choice == '2':
        from complex_env import IowaEnv as GameEnv
        env_name = "Complex-6-Deck"
        temp_env = GameEnv()
        n_actions = len(temp_env.action_space)
        del temp_env
        print(f"\n--- Loading Complex {n_actions}-Deck Environment ---")
    elif choice == '3':
        from complex_env import IowaEnv as GameEnv
        env_name = "Extreme-8-Deck"
        temp_env = GameEnv()
        n_actions = len(temp_env.action_space)
        del temp_env
        print(f"\n--- Loading Extreme {n_actions}-Deck Environment ---")
    else:
        # Default to 4-deck
        env_name = "Classic-4-Deck"
        try:
            from classic_env import IowaEnv as GameEnv
            n_actions = 4 # Assuming env.py has 4 decks
            print("\n--- Loading Classic 4-Deck Environment ---")
        except ImportError:
            print("Could not find env.py. Defaulting to env_complex.py")
            from complex_env import IowaEnv as GameEnv
            n_actions = len(GameEnv().action_space)
            print(f"\n--- Loading {n_actions}-Deck Environment from env_complex.py ---")


    # Hyperparameters
    n_episodes = 3000
    learning_rate = 0.005
    # --- ** NEW ** --- Remove interactive gamma input
    
    # --- ** NEW: Define the full experiment sequence ** ---
    runs_per_gamma = 20
    gamma_settings = [0.9, 0.3]
    total_runs = len(gamma_settings) * runs_per_gamma
    
    print(f"\n--- Preparing to run {total_runs} total experiments ---")
    print(f"--- {runs_per_gamma} runs for Gamma={gamma_settings[0]} ---")
    print(f"--- {runs_per_gamma} runs for Gamma={gamma_settings[1]} ---")
    
    current_run_number = 0
    # Loop over each gamma setting
    for gm_value in gamma_settings:
        # Run the experiment 20 times for this gamma
        for i in range(runs_per_gamma):
            current_run_number += 1
            
            # --- ** NEW ** --- Define unique names for logging
            run_name = f"run_{i+1}" # e.g., "run_1", "run_2", ...
            group_name = f"Env-{env_name}-Gamma-{gm_value}" # e.g., "Env-Classic-4-Deck-Gamma-0.9"
            
            # Pass all necessary config to the experiment function
            run_experiment(
                gm=gm_value,
                env_name=env_name,
                n_actions=n_actions,
                GameEnv=GameEnv,
                device=device,
                n_episodes=n_episodes,
                learning_rate=learning_rate,
                run_name=run_name,
                group_name=group_name,
                run_counter=current_run_number,
                total_runs=total_runs
            )

    print("\n--- All 40 experiments are complete. ---")