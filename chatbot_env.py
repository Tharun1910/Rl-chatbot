import gymnasium as gym
from gymnasium import spaces
import numpy as np
from nlp_utils import get_query_embedding, generate_response, sentiment_analyzer

class ChatbotEnv(gym.Env):
    def __init__(self):
        super(ChatbotEnv, self).__init__()
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(384,), dtype=np.float32)
        self.action_space = spaces.Discrete(1)  # only 1 candidate now
        self.current_query = ""
        self.cached_embedding = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_query = "Hello"
        self.cached_embedding = get_query_embedding(self.current_query)
        self.candidates = generate_response(self.current_query, num_candidates=1)
        return self.cached_embedding, {}

    def step(self, action):
        chosen_response = self.candidates[action]
        sentiment_score = sentiment_analyzer.polarity_scores(chosen_response)["compound"]
        reward = 2 if sentiment_score >= 0.5 else (1 if sentiment_score >= 0 else -1)
        terminated = True
        truncated = False
        info = {"response": chosen_response, "sentiment_score": sentiment_score}
        return self.cached_embedding, reward, terminated, truncated, info
