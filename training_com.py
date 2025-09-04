import gymnasium as gym
import numpy as np
import torch
from datetime import datetime
import csv
import time
from models import TD3, Actor, Critic, device, BATCH_SIZE, EXPLORATION_NOISE

print(f"Using device: {device}")

SAVE = "./training_model_com/model"
LOAD = "./training_model_com/model_final"

def calculate_reward(next_state, base_reward):
    # get coordinates
    fingertip_z = next_state[16]
    object_pos = next_state[17:19]
    goal_pos = next_state[20:22]

    if fingertip_z > 0.1:
        height_penalty = -0.4 * fingertip_z
    else:
        height_penalty = 0.0

    # calculate distance from object to goal
    dist_squared = np.sum(np.square(object_pos - goal_pos))
    distance_penalty = -1 * dist_squared
    # combine rewards
    combined_reward = base_reward + distance_penalty + height_penalty
    
    return combined_reward

def train_agent(save_path=SAVE, load_path=LOAD, start_episode=0, load=False):

    max_episodes = 5000
    max_timesteps = 500
    total_steps = 0
    warmup_steps = 10000

    env = gym.make("Pusher-v5", render_mode=None, max_episode_steps=max_timesteps)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    policy = TD3(state_dim, action_dim, max_action)
    best_reward = -float('inf')

    if load and load_path:
        try:
            policy.load(load_path)
            print(f"\nModel loaded: {load_path}")
        except Exception as e:
            print(f"Error when loading model: {e}")
            return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_time = time.time()

    training_stats_path = f"{save_path}_training_stats_{timestamp}.csv"
    episode_stats_path = f"{save_path}_episode_stats_{timestamp}.csv"

    with open(training_stats_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Episode", "Episode Reward", "Episode Time",
            "Best Reward", "Replay Buffer Size"
        ])

    with open(episode_stats_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Episode', 'Total Steps', 'Training Time',
            'Best Reward', 'Replay Buffer Size'
        ])

    print("\nStart training...")
    if load == True:
        print("Continue with existing model...")
    print(f"State Space Dimension: {state_dim}")
    print(f"Action Space Dimension: {action_dim}")
    print(f"Max Action: {max_action}")
    print(f"Max Episodes: {max_episodes}")
    print(f"Max Timesteps: {max_timesteps}")
        
    try:
        for episode in range(start_episode, max_episodes):
            epidode_strat_time = time.time()
            state = env.reset()[0]
            episode_reward = 0
            
            for t in range(max_timesteps):
                if total_steps < warmup_steps and load == False:
                    action = env.action_space.sample()
                else:
                    action = policy.select_action(state)
                    action = action + np.random.normal(0, EXPLORATION_NOISE, size=action_dim)
                    action = np.clip(action, -max_action, max_action)
                
                # get next action
                next_state, reward, done, truncated, _ = env.step(action)
                done = done or truncated

                combined_reward = calculate_reward(next_state, reward)

                # store to ReplayBuffer
                policy.replay_buffer.push(state, action, combined_reward, next_state, float(done))
                
                state = next_state
                episode_reward += combined_reward
                total_steps += 1

                # train the agent
                if len(policy.replay_buffer) > BATCH_SIZE:
                    policy.train(BATCH_SIZE)

                if done:
                    break

            episode_time = time.time() - epidode_strat_time

            episode_data = [
                episode + 1, 
                round(episode_reward, 2), 
                round(episode_time, 2),       
                round(best_reward, 2), 
                len(policy.replay_buffer)
            ]

            with open(training_stats_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(episode_data)
            
            if (episode + 1) % 10 == 0:
                print(f"Episode: {episode}")
                print(f"Reward: {episode_reward:.2f}")
                print(f"Exploration: {EXPLORATION_NOISE:.3f}")
                print(f"Replay Buffer Size: {len(policy.replay_buffer)}")
            
            # save the best model
            if episode_reward > best_reward:
                best_reward = episode_reward
                try:
                    policy.save(f"{save_path}_best")
                    print(f"\nBest model saved to {save_path}_best")
                    print(f"Best Reward: {best_reward:.2f}")
                except Exception as e:
                    print(f"Error when saving best model: {e}")
            
            # save checkpoint
            if (episode + 1) % 100 == 0:
                try:
                    checkpoint_path = f"{save_path}_episode_{episode+1}"
                    policy.save(checkpoint_path)
                    print(f"\nCheckpoint saved: {checkpoint_path}")
                except Exception as e:
                    print(f"Error when saving checkpoint: {e}")

                try:
                    training_time = round(time.time() - start_time, 2)
                    with open(episode_stats_path, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        episode_stats = [
                            episode + 1, 
                            total_steps, 
                            training_time,
                            best_reward, 
                            len(policy.replay_buffer)
                        ]
                        writer.writerow(episode_stats)
                except Exception as e:
                    print(f"Error when saving episode status: {e}")
    
    except KeyboardInterrupt:
        print("\nKeyboard interruption. Saving...")
    
    finally:
        # save final model
        try:
            final_path = f"{save_path}_final"
            policy.save(final_path)
            print(f"\nTraining finished. Final model saved to: {final_path}")
            print(f"Best Reward: {best_reward:.2f}")
            print(f"Training time: {time.time() - start_time:.2f} seconds")
        except Exception as e:
            print(f"Error when saving final model: {e}")
        
        env.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
        train_agent()