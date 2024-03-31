import pandas as pd

class StockTrading_Environment:
    def __init__(self, data_file, initial_balance=10000, num_bins=10):
        self.initial_data_file = data_file
        self.df = pd.read_csv(data_file)
        self.initial_balance = initial_balance
        self.num_bins = num_bins  # Number of bins for discretization
        #  state and action spaces : 0: hold, 1: buy, 2: sell
        self.n_actions = 3
        #self.state_size = (self.df.shape[1] - 1) * num_bins  # Adjusted for binned features
        self.state_size = 12000000
        self.discretize_features()  # Discretize features
        self.reset_the_model()

    def discretize_features(self):
        # Identify continuous feature columns to discretize, excluding 'Target'
        continuous_features = ['Close', 'Daily_Change', 'Daily_Return', 'MA_5', 'MA_10', 'Volume_Change', 'Volatility']

        # Discretize each feature
        for feature in continuous_features:
            self.df[feature + '_binned'], bins = pd.cut(self.df[feature], self.num_bins, retbins=True, labels=False)
            # Storing bins is optional, useful if you need to interpret the bins later

    def reset_the_model(self, test_mode=False, test_data_file=None):
        # Test Mode ? Yes - Load test data
        # else,  reload  initial training data
        if test_mode and test_data_file:
            self.df = pd.read_csv(test_data_file)
        else:
            self.df = pd.read_csv(self.initial_data_file)
        self.discretize_features()

        # Reset environment state
        self.current_step = 0
        self.total_balance = self.initial_balance
        self.holdings = 0
        self.done = False
        return self.get_the_current_state()

    def get_the_current_state(self):
        encoded_state = 0
        multiplier = 1
        for feature in ['Close_binned', 'Daily_Change_binned', 'Daily_Return_binned', 'MA_5_binned', 'MA_10_binned',
                        'Volume_Change_binned', 'Volatility_binned']:
            encoded_state += self.df.loc[self.current_step, feature] * multiplier
            multiplier *= self.num_bins
        return int(encoded_state)

    def simulate_actions(self, action):
        current_price = self.df.loc[self.current_step, 'Close']
        reward = 0

        if action == 1:
            # buy one unit of stock
            self.total_balance -= current_price
            self.holdings += 1
        elif action == 2:
            # sell one unit of stock
            self.total_balance += current_price
            self.holdings -= 1
            if self.current_step > 0:
                reward = max(0, current_price - self.df.loc[self.current_step - 1, 'Close'])  # Profit from last buy

        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            self.done = True

        # New  state and reward
        new_state = self.get_the_current_state()
        return new_state, reward, self.done
