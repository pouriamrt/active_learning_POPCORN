import streamlit as st
import pandas as pd
import random
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from gymnasium import spaces

# Sample dataset: Replace with real research paper abstracts
dataset = [
    {"title": "Deep Learning in Healthcare", "abstract": "This paper explores the application of deep learning models in diagnosing diseases."},
    {"title": "Neural Networks for Natural Language Processing", "abstract": "A study on how neural networks improve NLP tasks like sentiment analysis and translation."},
    {"title": "Quantum Computing and Cryptography", "abstract": "Quantum mechanics' impact on modern encryption techniques is discussed in this research."},
    {"title": "Reinforcement Learning for Robotics", "abstract": "How reinforcement learning enables robots to make autonomous decisions."},
    {"title": "Graph Neural Networks in Drug Discovery", "abstract": "This paper highlights the role of GNNs in identifying potential drug candidates."}
]

# Define custom reinforcement learning environment
class PaperRecommenderEnv(gym.Env):
    def __init__(self):
        super(PaperRecommenderEnv, self).__init__()
        self.num_papers = len(dataset)
        self.observation_space = spaces.Discrete(self.num_papers)  # Each paper as a discrete state
        self.action_space = spaces.Discrete(2)  # Include (1) or Exclude (0)
        self.current_paper = 0
        self.user_feedback = []

    def step(self, action):
        reward = 1 if action == 1 else -1  # Reward for include, penalty for exclude
        self.user_feedback.append((self.current_paper, action))
        self.current_paper = (self.current_paper + 1) % self.num_papers
        done = len(self.user_feedback) >= 10  # Episode ends after 10 interactions
        return self.current_paper, reward, done, False, {}

    def reset(self, seed=None, options=None):
        self.current_paper = 0
        self.user_feedback = []
        return self.current_paper, {}

# Train RL Model
def train_rl_model():
    env = PaperRecommenderEnv()
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=5000)  # Train for 5000 steps
    return model

# Load or train model
@st.cache_resource
def load_model():
    return train_rl_model()

model = load_model()

def recommend_paper(env):
    action, _ = model.predict(env.current_paper)
    return dataset[env.current_paper], action

# Streamlit UI
st.title("Reinforcement Learning Paper Recommender")
env = PaperRecommenderEnv()

if "current_paper" not in st.session_state:
    st.session_state.current_paper, _ = env.reset()

paper, predicted_action = recommend_paper(env)
st.subheader(paper['title'])
st.write(paper['abstract'])

col1, col2 = st.columns(2)
with col1:
    if st.button("Include Paper"):
        obs, reward, done, _, _ = env.step(1)
        st.session_state.current_paper = obs
        st.rerun()

with col2:
    if st.button("Exclude Paper"):
        obs, reward, done, _, _ = env.step(0)
        st.session_state.current_paper = obs
        st.rerun()

st.write("Feedback collected:", env.user_feedback)