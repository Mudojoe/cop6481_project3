from qlearn_trade_bins import StockTrading_Environment
from qlearn_agent import QLearningAgent
from chart_aapl_stock import evaluate_buy_and_hold

def evaluate_the_stock_model(env, agent, episodes=100):
    total_rewards = []
    for episode in range(episodes):
        # Reset the environment for evaluation with the test data

        state = env.reset_the_model(True, 'AAPL_6M_Test_Data_Reformatted.csv')
        total_reward = 0
        done = False
        while not done:
            action = agent.choose_best_action(state, test=True)  # Select action based on the learned Q-values
            next_state, reward, done = env.simulate_actions(action)
            total_reward += reward
            state = next_state

        total_rewards.append(total_reward)

    average_reward = sum(total_rewards) / len(total_rewards)
    average_reward_percentage = average_reward * 100  # Convert to percentage
    print(f"Average Reward over {episodes} evaluation episodes: {average_reward_percentage:.2f}%")


def main():
    # Initialization
    training_data_file = 'AAPL_3Y_Historical_Data_Reformatted.csv'
    env = StockTrading_Environment(training_data_file)
    agent = QLearningAgent(state_size=env.state_size, action_size=env.n_actions)
    number_of_episodes = 5000  # Training number_of_episodes

    for episode in range(number_of_episodes):
        state = env.reset_the_model()  # Reset the environment for a new episode
        total_reward = 0
        done = False

        while not done:
            action = agent.choose_best_action(state)  # Agent selects an action based on current state
            next_state, reward, done = env.simulate_actions(action)  # Execute the action in the environment

            # Update the Q-table with the new knowledge
            agent.update_q_table(state, action, reward, next_state, done)

            state = next_state  # Transition to the next state
            total_reward += reward

        if episode % 500 == 0:  # Log progress every 100 number_of_episodes
            print(f"Episode: {episode}, Total Reward: {total_reward}, Exploration Rate: {agent.exploration_rate}")


    # After training is complete, evaluate the model on test data
    test_data_file = 'AAPL_6M_Test_Data_Reformatted.csv'
    evaluate_the_stock_model(env, agent, episodes=100)
    buy_and_hold_percent_diff = evaluate_buy_and_hold()
    print(f"Buy-and-Hold Percent Difference: {buy_and_hold_percent_diff:.2f}%")


if __name__ == "__main__":
    main()
