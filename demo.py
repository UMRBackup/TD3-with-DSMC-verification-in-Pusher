import gymnasium as gym
import numpy as np
import torch
import time
from models import TD3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def demo_agent(model_path, episodes=5, timesteps=500, render_speed=0.02, window_size=(1920, 1080)):

    # Create environment
    env = gym.make(
        "Pusher-v5",
        render_mode="human",
        max_episode_steps=timesteps,
        width=window_size[0],
        height=window_size[1]
    )
    
    # Get environment parameters
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    # Initialize policy
    policy = TD3(state_dim, action_dim, max_action)
    
    try:
        # Load trained model
        policy.load(model_path)
        print(f"model loaded: {model_path}")
        
        total_reward = 0
        successful_episodes = 0
        
        # Run multiple demo episodes
        for episode in range(episodes):
            
            # Reset environment
            state = env.reset()[0]
            episode_reward = 0
            episode_steps = 0
            
            # Run episode
            for step in range(timesteps):
                # Select action
                action = policy.select_action(state)
                
                # Take action
                next_state, reward, done, truncated, info = env.step(action)
                done = done or truncated
                
                episode_reward += reward
                episode_steps += 1
                state = next_state
                
                # Control render speed
                time.sleep(render_speed)
            
            # Episode statistics
            total_reward += episode_reward
            
            # Pause between episodes
            time.sleep(1)            
    finally:
        env.close()

def quick_demo():

    model_paths = {
        "TD3": "./training_model_com/model_best",
        "TD3_with_DSMC": "./training_model_ver/model_best"
    }
    
    # List available models
    print("Models:")
    for i, (name, path) in enumerate(model_paths.items(), 1):
        print(f"{i}. {name}: {path}")
    
    selected_model = list(model_paths.values())[0]
    selected_name = list(model_paths.keys())[0]
    
    print(f"\n Model: {selected_name}")
    
    # Start demo
    demo_agent(
        model_path=selected_model,
        episodes=5,
        timesteps=500,
        render_speed=0.01
    )

if __name__ == "__main__":

    quick_demo()
