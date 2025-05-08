import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from chatbot_env import ChatbotEnv

# Ensure the directory for saving the model exists
if not os.path.exists("models"):
    os.makedirs("models")

# Create the environment wrapped in DummyVecEnv and Monitor
env = DummyVecEnv([lambda: Monitor(ChatbotEnv(), "models/chatbot_training")])

# Initialize the PPO agent with a simple MLP policy
model = PPO("MlpPolicy", env, verbose=1, device="cpu")

# Train the model
model.learn(total_timesteps=1000)

# Save the trained model
model.save("models/chatbot_rl_agent")

# Optionally: Evaluate the model after training
# Here, you could run the model on a few episodes to see how it performs:
obs = env.reset()
for _ in range(10):
    action, _ = model.predict(obs)
    obs, rewards, dones, infos = env.step(action)
    print(f"Response: {infos[0]['response']}, Sentiment Score: {infos[0]['sentiment_score']}, Reward: {rewards[0]}")
    if dones[0]:
        obs = env.reset()
