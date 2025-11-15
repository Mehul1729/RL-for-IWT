import numpy as np
from collections import deque

class IowaEnv:
    """
    This class simulates an EXTREME IGT with 8 decks.
    It includes the 4 classic decks plus four "trap" decks (E, F, G, H)
    to make the learning problem significantly harder.

    
    - Decks A, B: High reward, net loss (Classic Bad)
    - Decks C, D: Low reward, net gain (Classic Good)
    - Deck E: Medium reward, net zero (Deceptive Trap)
    - Deck F: Very low reward, net gain (Hidden Gem)
    - Deck G: Highest reward, net zero (The "Siren" Trap)
    - Deck H: Variable, net *slight* loss (The "Grin" Trap)
    """
    def __init__(self, episode_length: int = 100):
        """
        Initializes the Extreme Complex Iowa Gambling Task environment.
            episode_length : The number of trials in one episode.
        """
        self.episode_length = episode_length
        
        self._deck_properties = {
            
            # --- DECKS---
            # BAD DECKS: High immediate reward, net loss.
            # Deck A: EV per 10: (10 * 100) - (5 * 250) = 1000 - 1250 = -250
            0: {'reward': [80, 100, 120], 'penalty_amount': [-200, -250, -300], 'penalty_prob': 0.5},
            
            # Deck B: EV per 10: (10 * 100) - (1 * 1250) = 1000 - 1250 = -250
            1: {'reward': [90, 100, 110], 'penalty_amount': [-1200, -1250, -1300], 'penalty_prob': 0.1},
            
            
            
            # GOOD DECKS: Lower immediate reward, net gain.
            # Deck C: EV per 10: (10 * 50) - (5 * 50) = 500 - 250 = +250
            2: {'reward': [40, 50, 60], 'penalty_amount': [-40, -50, -60], 'penalty_prob': 0.5},
    
            # Deck D: EV per 10: (10 * 50) - (1 * 250) = 500 - 250 = +250
            3: {'reward': [45, 50, 55], 'penalty_amount': [-225, -250, -275], 'penalty_prob': 0.1},
            
            
            
            # --- COMPLEX/TRAP DECKS (NOW STOCHASTIC) ---
            # DECEPTIVE TRAP: Looks better than C/D, but is worse.
            # Deck E: EV per 10: (10 * 75) - (2.5 * 300) = 750 - 750 = 0 (Net Zero)
            4: {'reward': [70, 75, 80], 'penalty_amount': [-275, -300, -325], 'penalty_prob': 0.25},
            
            # HIDDEN GEM: Looks terrible, but is secretly good.
            # Deck F: EV per 10: (10 * 25) - (1 * 10) = 250 - 10 = +240 (Net Good)
            5: {'reward': [20, 25, 30], 'penalty_amount': [-5, -10, -15], 'penalty_prob': 0.1},
            
            # --- TRAP DECKS ---
            #  Highest immediate reward, but net zero. Very attractive trap.
            # Deck G: EV per 10: (10 * 125) - (2.5 * 500) = 1250 - 1250 = 0 (Net Zero)
            6: {'reward': [120, 125, 130], 'penalty_amount': [-450, -500, -550], 'penalty_prob': 0.25},
            
            # Variable outcomes that average to a *slight* loss. Very hard to detect.
            # Deck H: EV per 10: (10 * 60) - (2 * 325) = 600 - 650 = -50 (Net Slight Loss)
            7: {'reward': [10, 50, 60, 70, 110], 'penalty_amount': [-300, -325, -350], 'penalty_prob': 0.2}
        }
        
        self.action_space = list(self._deck_properties.keys()) # [0, 1, 2, 3, 4, 5, 6, 7]
        self.reset()
        
    def _calculate_reward(self, action: int) -> int:
        """
        Calculates the reward for a given action based on the deck's
        stochastic properties.
        """
        deck = self._deck_properties[action]
        self._deck_pull_counts[action] += 1
        
        # Selecting a random reward from the deck's reward list:
        reward = np.random.choice(deck['reward'])
        
        # Checking if a penalty occurs based on the deck's penalty probability:
        if np.random.rand() < deck['penalty_prob']:
            # Select a random penalty from the deck's penalty list:
            reward += np.random.choice(deck['penalty_amount'])
            
        return reward
    
    def reset(self):
        """Resets the environment to its initial state for a new episode."""
        self._current_trial = 0
        self.cumulative_reward = 0
        self._last_action = -1 
        self._action_history = deque(maxlen=10)
        self._deck_pull_counts = {i: 0 for i in self.action_space}
        return self._get_state()

    def _get_state(self):
        """Constructs the state dictionary from the environment's current properties."""
        return {
            'trial': self._current_trial,
            'cumulative_reward': self.cumulative_reward,
            'last_action': self._last_action,
            'history': list(self._action_history)
        }
    
    def step(self, action: int):
        """Executes one time step in the environment."""
        if action not in self.action_space:
            raise ValueError(f"Invalid action {action}. Action must be in {self.action_space}.")
        
        reward = self._calculate_reward(action)
        self.cumulative_reward += reward
        self._last_action = action
        self._action_history.append(action)
        self._current_trial += 1
        
        done = self._current_trial >= self.episode_length
        next_state = self._get_state()
        info = {'deck_pulls': self._deck_pull_counts.copy()}
        
        return next_state, reward, done, info
    
    def render(self):
        """Prints a summary of the current environment state."""
        print(f"--- Trial: {self._current_trial}/{self.episode_length} ---")
        print(f"Last Action: {'None' if self._last_action == -1 else f'Deck {chr(65 + self._last_action)}'}")
        print(f"Cumulative Reward: {self.cumulative_reward}")
        print(f"Deck Pulls: {self._deck_pulls}")
        print("-" * 25)

