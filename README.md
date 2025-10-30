# Policy Learning for Iowa Gambling Problem

**A simple Neural Network based Agent is trained to learn the best policy for a complex set of decks of the Iowa Gambling Task. Following are the related plots and results:**

---

### Agent chooses deck "D" out of all other decks to be the best
![Agent Deck Choice](https://github.com/Mehul1729/RL-for-IWT/blob/807550f9f0f42e057a4edea20a388bb8de1b379c/Policy_Learning/Plots/policy%20learning%20decks%20.png)

---

### Agent becoming confident with its prediction as evidenced by an increasing log probabilities it assigns to each choice
![Agent Confidence](https://github.com/Mehul1729/RL-for-IWT/blob/807550f9f0f42e057a4edea20a388bb8de1b379c/Policy_Learning/Plots/model%20confidence%20building.png)

---

### REINFORCE based loss function settling to near zero after oscillating for extreme negatives and positives
![Loss Function Trend](https://github.com/Mehul1729/RL-for-IWT/blob/807550f9f0f42e057a4edea20a388bb8de1b379c/Policy_Learning/Plots/policy%20losse.png)

---

**Loss Function used --> Cross Entropy Style loss for Discounted Returns and learned log probabilities of action from each trial :**  


![eqn](https://github.com/Mehul1729/RL-for-IWT/blob/4257898f7b82063d5727a644464e7ca844494aef/Policy_Learning/Plots/eqn%20of%20lss%20.png)
