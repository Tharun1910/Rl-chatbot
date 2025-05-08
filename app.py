import streamlit as st
from nlp_utils import generate_response, sentiment_analyzer
from stable_baselines3 import PPO
from chatbot_env import ChatbotEnv
import numpy as np
import sys
if 'torch.classes' in sys.modules:
    del sys.modules['torch.classes']


st.title(" Personalized Chatbot using RL")
st.write("This is a personalized chatbot that uses reinforcement learning to generate responses.")

model = PPO.load("models/chatbot_rl_agent")

env = ChatbotEnv()

user_input = st.text_input("You:", "")

if user_input:
    state, _ = env.reset()
    env.current_query = user_input
    env.candidates = generate_response(user_input, num_candidates=1)
    env.cached_embedding = env.observation_space.sample()  # dummy input

    action, _ = model.predict(state, deterministic=True)
    response = env.candidates[action]

    st.write(f"🤖 Bot: {response}")
    st.write(f"📝 Sentiment: {sentiment_analyzer.polarity_scores(response)}")
