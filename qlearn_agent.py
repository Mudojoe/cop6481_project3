import numpy as np

class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1,
                 #discount_rate=0.95,
                 discount_rate=0.05,
                 exploration_rate=1.0,
                 exploration_decay=0.99, exploration_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.exploration_min = exploration_min

        # Initialize Q-table with zeros
        self.q_table = np.zeros((state_size, action_size))

    def choose_best_action(self, state, test=False):
        if test:
            # When testing, always choose the best action (exploitation)
            return np.argmax(self.q_table[state])
        else:
            # During training, use an epsilon-greedy strategy for exploration-exploitation
            if np.random.rand() < self.exploration_rate:
                #return np.random.randint(self.action_size)  # Explore
                return np.random.randint(0, self.action_size)
            else:
                return np.argmax(self.q_table[state])  # Exploit


    def update_q_table(self, state, action, reward, next_state, done):
        # Update Q-table using the Q-learning formula, using encoded states
        # The states are now single integers, so they can index the Q-table directly
        #print(f"Updating Q-table with next_state: {next_state}, Type: {type(next_state)}")  # Debug print

        if not done:
            max_future_q = np.max(self.q_table[next_state])  # Use encoded next_state directly
            current_q = self.q_table[state, action]
            new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (
                        reward + self.discount_rate * max_future_q)
            self.q_table[state, action] = new_q

        self.exploration_rate = max(self.exploration_min, self.exploration_rate * self.exploration_decay)

    def inspect_q_table_for_testing(self) :
        inspect = self.q_table
        is_all_zeros = np.all((self.q_table == 0))
        print(f"Is the Q-table all zeros? {is_all_zeros}")
