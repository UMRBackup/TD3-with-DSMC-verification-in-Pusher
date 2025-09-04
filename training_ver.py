import gymnasium as gym
import numpy as np
import torch
import time
import os
import csv
from datetime import datetime
from models import TD3, Actor, Critic, device, BATCH_SIZE, EXPLORATION_NOISE
import math
import random

print(f"Using device: {device}")

SAVE = "./training_model_ver/model"
LOAD = "./training_model_ver/model_final"

# observation
JOINT_LIMITS = {
    'shoulder_pan': {
        'pos_idx': 0,
        'vel_idx': 7,
        'pos_range': (-2.2854, 2.2854),
        'vel_range': (-3.0, 3.0),
        'name': 'r_shoulder_pan_joint'
    },
    'shoulder_lift': {
        'pos_idx': 1,
        'vel_idx': 8,
        'pos_range': (-1.7140, 1.7140),
        'vel_range': (-3.0, 3.0),
        'name': 'r_shoulder_lift_joint'
    },
    'upper_arm_roll': {
        'pos_idx': 2,
        'vel_idx': 9,
        'pos_range': (-3.9000, 3.9000),
        'vel_range': (-4.0, 4.0),
        'name': 'r_upper_arm_roll_joint'
    },
    'elbow_flex': {
        'pos_idx': 3,
        'vel_idx': 10,
        'pos_range': (-2.2854, 2.2854),
        'vel_range': (-4.0, 4.0),
        'name': 'r_elbow_flex_joint'
    },
    'forearm_roll': {
        'pos_idx': 4,
        'vel_idx': 11,
        'pos_range': (-3.9000, 3.9000),
        'vel_range': (-4.0, 4.0),
        'name': 'r_forearm_roll_joint'
    },
    'wrist_flex': {
        'pos_idx': 5,
        'vel_idx': 12,
        'pos_range': (-2.2854, 2.2854),
        'vel_range': (-4.0, 4.0),
        'name': 'r_wrist_flex_joint'
    },
    'wrist_roll': {
        'pos_idx': 6,
        'vel_idx': 13,
        'pos_range': (-3.9000, 3.9000),
        'vel_range': (-4.0, 4.0),
        'name': 'r_wrist_roll_joint'
    }
}

# penalties
VIOLATION_PENALTIES = {
    'velocity': {
        'severity_thresholds': [0.1, 0.3, 0.5],
        'penalties': [-0.2, -0.5, -1]
    },
    'fingertip': {
        'severity_thresholds': [0.1, 0.3, 0.5],
        'penalties': [-0.1, -0.3, -0.8]
    }
}

# fingertip restraints
FINGERTIP_LIMITS = {
    'x_range': (-1, 1),   
    'y_range': (-0.7, 1), 
    'z_range': (-0.5, 0.7), 
    'back_penalty': -1
}

# DSMC
DSMC_CONFIG = {
    'check_interval': 1000, 
    'check_episodes': 2000,
    'grid_size': 0.1,
    'confidence_k': 0.08,    # confidence interval
    'error_epsilon': 0.12,
}

# validator
class PusherStateValidator:
    def __init__(self):
        self.joint_limits = JOINT_LIMITS
        self.violations = []
        self.total_checks = 0
        self.violation_counts = {joint: 0 for joint in JOINT_LIMITS.keys()}
        self.violation_history = []
        self.fingertip_violations = []
    
    def check_fingertip_position(self, env):
        while hasattr(env, 'env'):
            env = env.env
        
        fingertip_pos = env.get_body_com("tips_arm")
        violations = []
        
        if fingertip_pos[1] < FINGERTIP_LIMITS['y_range'][0]:
            violations.append({
                'type': 'fingertip_position',
                'axis': 'y',
                'value': fingertip_pos[1],
                'limit': FINGERTIP_LIMITS['y_range'],
                'timestamp': self.total_checks,
                'penalty': FINGERTIP_LIMITS['back_penalty'],
                'description': 'Fingertip moving backward'
            })
        
        for axis, pos, limits, desc in zip(
            ['x', 'z'], 
            [fingertip_pos[0], fingertip_pos[2]], 
            [FINGERTIP_LIMITS['x_range'], FINGERTIP_LIMITS['z_range']],
            ['x out of range', 'z out of range']
        ):
            if pos < limits[0] or pos > limits[1]:
                violations.append({
                    'type': 'fingertip_position',
                    'axis': axis,
                    'value': pos,
                    'limit': limits,
                    'timestamp': self.total_checks,
                    'penalty': VIOLATION_PENALTIES['fingertip']['penalties'][-1],
                    'description': desc
                })
        
        if violations:
            self.fingertip_violations.extend(violations)
            return False, violations
        return True, []
    
    def check_state(self, state, env):
        self.total_checks += 1
        current_violations = []
        
        # speed limit
        for joint, limits in self.joint_limits.items():
            vel = state[limits['vel_idx']]
            vel_min, vel_max = limits['vel_range']
            if vel < vel_min or vel > vel_max:
                violation = {
                    'joint': joint,
                    'type': 'velocity',
                    'value': vel,
                    'limit': limits['vel_range'],
                    'timestamp': self.total_checks
                }
                current_violations.append(violation)
                self.violation_counts[joint] += 1

        is_safe_fingertip, fingertip_violations = self.check_fingertip_position(env)
        if not is_safe_fingertip:
            current_violations.extend(fingertip_violations)
        
        if current_violations:
            self.violations.extend(current_violations)
            return False, current_violations
        return True, []
    
    def calculate_violation_penalty(self, violation):
        v_type = violation['type']
        value = violation['value']
        limit_min, limit_max = violation['limit']
        limit_range = limit_max - limit_min

        if value < limit_min:
            severity = abs(limit_min - value) / limit_range
        else:
            severity = abs(value - limit_max) / limit_range
            
        # give penalty based on severity
        thresholds = VIOLATION_PENALTIES[v_type]['severity_thresholds']
        penalties = VIOLATION_PENALTIES[v_type]['penalties']
        
        for threshold, penalty in zip(thresholds, penalties):
            if severity <= threshold:
                return penalty
        return penalties[-1]
    
    def get_total_penalty(self, violations):
        total_penalty = 0
        for violation in violations:
            if violation['type'] == 'fingertip_position':
                penalty = violation['penalty']
            else:
                penalty = self.calculate_violation_penalty(violation)
            
            total_penalty += penalty
            self.violation_history.append({
                'timestamp': violation['timestamp'],
                'type': violation['type'],
                'value': violation['value'],
                'limit': violation['limit'],
                'penalty': penalty
            })
        return total_penalty

    def get_statistics(self):
        return {
            'total_checks': self.total_checks,
            'total_violations': len(self.violations),
            'violation_counts': self.violation_counts,
            'violation_rate': len(self.violations) / max(1, self.total_checks)
        }

class FixedWidthConfidenceInterval:
    def __init__(self, epsilon=0.1, delta=0.05, k_threshold=0.05):
        self.epsilon = epsilon
        self.delta = delta
        self.k_threshold = k_threshold
        self.confidence_level = 1 - delta
        
        # Nx ≥ 1/(2ε²)ln(2/δ)
        self.min_samples = self.calculate_min_samples()
        
    def calculate_min_samples(self):
        return math.ceil((1 / (2 * self.epsilon**2)) * math.log(2 / self.delta))
    
    def evaluate_confidence(self, violation_episodes, total_episodes):

        if total_episodes < self.min_samples:
            return {
                'has_confidence': False,
                'empirical_error': None,
                'confidence_bound': None,
                'samples_needed': self.min_samples - total_episodes,
                'current_samples': total_episodes,
                'min_required_samples': self.min_samples,
                'confidence_level': self.confidence_level
            }
        
        empirical_error = violation_episodes / total_episodes
        
        confidence_bound = empirical_error + self.epsilon
        
        has_confidence = empirical_error <= self.k_threshold
        
        return {
            'has_confidence': has_confidence,
            'empirical_error': empirical_error,
            'confidence_bound': confidence_bound,
            'samples_needed': 0,
            'current_samples': total_episodes,
            'min_required_samples': self.min_samples,
            'epsilon': self.epsilon,
            'delta': self.delta,
            'k_threshold': self.k_threshold,
            'confidence_level': self.confidence_level
        }

class DSMCValidator:
    def __init__(self, env, policy, validator):
        self.env = env
        self.policy = policy
        self.validator = validator
        self.state_statistics = {}
        
        self.confidence_calculator = FixedWidthConfidenceInterval(
            epsilon=DSMC_CONFIG['error_epsilon'], 
            delta=0.05,
            k_threshold=DSMC_CONFIG['confidence_k']
        )
        
    def get_grid_position(self, pos):
        grid_size = DSMC_CONFIG['grid_size']
        return (
            int(pos[0] / grid_size),
            int(pos[1] / grid_size)
        )
    
    def get_state_key(self, obj_pos, target_pos):
        obj_grid = self.get_grid_position(obj_pos)
        target_grid = self.get_grid_position(target_pos)
        return f"obj_{obj_grid[0]}_{obj_grid[1]}_target_{target_grid[0]}_{target_grid[1]}"
    
    def run_verification(self):
        print("\nDSMC verification started...")
        print(f"Verification Episodes: {DSMC_CONFIG['check_episodes']}")
        print(f"Required samples per state: {self.confidence_calculator.min_samples}")
        print(f"Error bound ε: {self.confidence_calculator.epsilon}")
        print(f"Confidence level: {self.confidence_calculator.confidence_level*100:.1f}%")
        
        for episode in range(DSMC_CONFIG['check_episodes']):
            state = self.env.reset()[0]
            
            real_env = self.env
            while hasattr(real_env, 'env'):
                real_env = real_env.env
            obj_pos = real_env.get_body_com("object")[:2]
            goal_pos = real_env.get_body_com("goal")[:2]
            state_key = self.get_state_key(obj_pos, goal_pos)
            
            if state_key not in self.state_statistics:
                self.state_statistics[state_key] = {
                    'total_episodes': 0,
                    'violation_episodes': 0,
                    'violations': [],
                    'rewards': []
                }
            
            done = False
            episode_violations = []
            episode_reward = 0
            
            while not done:
                action = self.policy.select_action(state)
                next_state, reward, done, truncated, _ = self.env.step(action)
                done = done or truncated
                
                is_safe, violations = self.validator.check_state(state, self.env)
                if not is_safe:
                    episode_violations.extend(violations)
                
                state = next_state
                episode_reward += reward
            
            stats = self.state_statistics[state_key]
            stats['total_episodes'] += 1
            stats['rewards'].append(episode_reward)
            if episode_violations:
                stats['violation_episodes'] += 1
                stats['violations'].extend(episode_violations)
            
            if (episode + 1) % 50 == 0:
                print(f"Verification progress: {episode + 1}/{DSMC_CONFIG['check_episodes']}")
        
        return self.calculate_confidence()
    
    def calculate_confidence(self):
        results = {}
        
        for state_key, stats in self.state_statistics.items():
            if stats['total_episodes'] == 0:
                continue
            
            # Fixed-width confidence intervals
            confidence_info = self.confidence_calculator.evaluate_confidence(
                stats['violation_episodes'], 
                stats['total_episodes']
            )
            
            avg_reward = sum(stats['rewards']) / len(stats['rewards'])
            
            results[state_key] = {
                'total_episodes': stats['total_episodes'],
                'violation_episodes': stats['violation_episodes'],
                'violation_rate': stats['violation_episodes'] / stats['total_episodes'],
                'avg_reward': avg_reward,
                'empirical_error': confidence_info['empirical_error'],
                'confidence_bound': confidence_info['confidence_bound'],
                'has_confidence': confidence_info['has_confidence'],
                'samples_needed': confidence_info['samples_needed'],
                'min_required_samples': confidence_info['min_required_samples'],
                'confidence_level': confidence_info.get('confidence_level', 0.95)
            }
        
        return results

class DSMCEvaluationStage:
    def __init__(self, env, policy, validator, confidence_calculator):
        self.env = env
        self.policy = policy
        self.validator = validator
        self.confidence_calculator = confidence_calculator
        self.target_states = {}  # store weakpoint
        self.evaluation_history = []
        
    def set_target_states(self, confidence_results):
        self.target_states = {}
        
        for state_key, result in confidence_results.items():
            # undersampled or not confident state
            if result['samples_needed'] > 0 or not result['has_confidence']:
                self.target_states[state_key] = {
                    'current_samples': result['total_episodes'],
                    'violation_episodes': result['violation_episodes'],
                    'samples_needed': max(result['samples_needed'], 
                                        result['min_required_samples'] - result['total_episodes']),
                    'target_episodes': result['min_required_samples'],
                    'violation_rate': result.get('violation_rate', 0),
                    'priority': self.calculate_priority(result)
                }
        
        print(f"\nEvaluation Stage: {len(self.target_states)} states need further training")
        return len(self.target_states) > 0
    
    def calculate_priority(self, result):
        priority = 0
        
        # weight for undersampling
        if result['samples_needed'] > 0:
            priority += result['samples_needed'] * 0.5
        
        # weight for violation
        if result.get('violation_rate', 0) > 0:
            priority += result['violation_rate'] * 100
        
        # extra weight for no confidence
        if not result['has_confidence']:
            priority += 50
            
        return priority
    
    def should_reset_to_target_state(self, current_obj_pos, current_goal_pos):
        current_state_key = self.get_state_key(current_obj_pos, current_goal_pos)
        return current_state_key in self.target_states
    
    def get_state_key(self, obj_pos, target_pos):
        grid_size = DSMC_CONFIG['grid_size']
        obj_grid = (int(obj_pos[0] / grid_size), int(obj_pos[1] / grid_size))
        target_grid = (int(target_pos[0] / grid_size), int(target_pos[1] / grid_size))
        return f"obj_{obj_grid[0]}_{obj_grid[1]}_target_{target_grid[0]}_{target_grid[1]}"
    
    def force_reset_to_target_state(self):
        if not self.target_states:
            return None
            
        sorted_states = sorted(self.target_states.items(), 
                             key=lambda x: x[1]['priority'], reverse=True)
        target_state_key = sorted_states[0][0]
        
        # get coordinate from state key
        try:
            parts = target_state_key.split('_')
            obj_x = int(parts[1]) * DSMC_CONFIG['grid_size']
            obj_y = int(parts[2]) * DSMC_CONFIG['grid_size']
            target_x = int(parts[4]) * DSMC_CONFIG['grid_size']
            target_y = int(parts[5]) * DSMC_CONFIG['grid_size']
            
            return self.try_set_environment_state(obj_x, obj_y, target_x, target_y)
        except:
            return None
    
    def try_set_environment_state(self, obj_x, obj_y, target_x, target_y):
        try:
            real_env = self.env
            while hasattr(real_env, 'env'):
                real_env = real_env.env
            
            real_env.set_state(real_env.init_qpos, real_env.init_qvel)
            
            if hasattr(real_env, 'model'):
                obj_body_id = real_env.model.body_name2id("object")
                real_env.model.body_pos[obj_body_id][:2] = [obj_x, obj_y]
                
                goal_body_id = real_env.model.body_name2id("goal")
                real_env.model.body_pos[goal_body_id][:2] = [target_x, target_y]
                
                real_env.forward()
                return True
        except Exception as e:
            print(f"Failed to set environment state: {e}")
            return False
        
        return False
    
    def run_evaluation_stage(self, max_evaluation_episodes=2000):
        print(f"\n=== DSMC Evaluation Stage Started ===")
        print(f"Target states to evaluate: {len(self.target_states)}")
        print(f"Max evaluation episodes: {max_evaluation_episodes}")
        
        evaluation_stats = {state_key: {'episodes': 0, 'violations': 0} 
                          for state_key in self.target_states.keys()}
        
        episodes_completed = 0
        target_states_completed = 0
        
        for episode in range(max_evaluation_episodes):
            state = self.env.reset()[0]

            real_env = self.env
            while hasattr(real_env, 'env'):
                real_env = real_env.env
            
            obj_pos = real_env.get_body_com("object")[:2]
            goal_pos = real_env.get_body_com("goal")[:2]
            current_state_key = self.get_state_key(obj_pos, goal_pos)
            
            # reset or skip if not in target states
            if current_state_key not in self.target_states:
                if not self.force_reset_to_target_state():
                    continue
                
                obj_pos = real_env.get_body_com("object")[:2]
                goal_pos = real_env.get_body_com("goal")[:2]
                current_state_key = self.get_state_key(obj_pos, goal_pos)
            
            if current_state_key not in self.target_states:
                continue
            
            done = False
            episode_violations = []
            episode_reward = 0
            
            while not done:
                action = self.policy.select_action(state)
                next_state, reward, done, truncated, _ = self.env.step(action)
                done = done or truncated
                
                is_safe, violations = self.validator.check_state(state, self.env)
                if not is_safe:
                    episode_violations.extend(violations)
                
                state = next_state
                episode_reward += reward

            evaluation_stats[current_state_key]['episodes'] += 1
            if episode_violations:
                evaluation_stats[current_state_key]['violations'] += 1
            
            episodes_completed += 1
            
            # check confidence
            current_stats = evaluation_stats[current_state_key]
            total_episodes = (self.target_states[current_state_key]['current_samples'] + 
                            current_stats['episodes'])
            total_violations = (self.target_states[current_state_key]['violation_episodes'] + 
                              current_stats['violations'])
            
            confidence_info = self.confidence_calculator.evaluate_confidence(
                total_violations, total_episodes
            )
            
            # remove flag if confident
            if (confidence_info['samples_needed'] == 0 and 
                confidence_info['has_confidence']):
                print(f"State {current_state_key} achieved confidence!")
                del self.target_states[current_state_key]
                target_states_completed += 1
            
            if episodes_completed % 100 == 0:
                print(f"Evaluation progress: {episodes_completed}/{max_evaluation_episodes}")
                print(f"Remaining target states: {len(self.target_states)}")
                print(f"Completed target states: {target_states_completed}")
            
            if len(self.target_states) == 0:
                print("All target states achieved confidence!")
                break
        
        print(f"\n=== Evaluation Stage Completed ===")
        print(f"Episodes completed: {episodes_completed}")
        print(f"Target states completed: {target_states_completed}")
        print(f"Remaining target states: {len(self.target_states)}")
        
        return evaluation_stats

def calculate_reward(next_state, base_reward):
    fingertip_z = next_state[16]
    object_pos = next_state[17:19]
    goal_pos = next_state[20:22]

    if fingertip_z > 0.1:
        height_penalty = -0.4 * fingertip_z
    else:
        height_penalty = 0.0
    
    # calculate distance
    dist_squared = np.sum(np.square(object_pos - goal_pos))
    distance_penalty = -1 * dist_squared

    combined_reward = base_reward + distance_penalty + height_penalty
    
    return combined_reward

def save_training_data(data, filename, headers, message=True):
    try:
        file_exists = os.path.isfile(filename)

        with open(filename, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(headers)
            writer.writerows(data)
        if message:
            print(f"Data saved to: {filename}")
    except Exception as e:
        print(f"Error when saving data: {e}")

def train_agent(save_path=SAVE, load_path=LOAD, start_episode=0, load=False):

    max_episodes = 5000
    max_timesteps = 500
    total_steps = 0
    warmup_steps = 10000

    env = gym.make("Pusher-v5", render_mode=None, max_episode_steps= max_timesteps)
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

    validator = PusherStateValidator()
        
    print("\nStart training...")
    if load == True:
        print("Continue with existing model...")
    print(f"State Space Dimension: {state_dim}")
    print(f"Action Space Dimension: {action_dim}")
    print(f"Max Action: {max_action}")
    print(f"Max Episodes: {max_episodes}")
    print(f"Max Timesteps: {max_timesteps}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_time = time.time()
    
    # log path
    training_stats_path = f"{save_path}_training_stats_{timestamp}.csv"
    violation_stats_path = f"{save_path}_violation_stats_{timestamp}.csv"
    dsmc_results_path = f"{save_path}_dsmc_results_{timestamp}.csv"
    episode_stats_path = f"{save_path}_episode_stats_{timestamp}.csv"
    training_time_path = f"{save_path}_training_time_{timestamp}.txt"
    
    try:
        for episode in range(start_episode, max_episodes):
            episode_start_time = time.time()
            state = env.reset()[0]
            episode_reward = 0
            episode_violations = []
            episode_fingertip_violations = 0
            episode_velocity_violations = 0
            
            for t in range(max_timesteps):
                
                is_safe, violations = validator.check_state(state, env)

                if total_steps < warmup_steps and load == False:
                    action = env.action_space.sample()
                elif not is_safe:
                    episode_violations.extend(violations)
                    
                    # adjust action based on penalty
                    action = policy.select_action(state)
                    violation_penalty = validator.get_total_penalty(violations)
                    action *= max(0.1, 1.0 + violation_penalty/10)
                else:
                    action = policy.select_action(state)
                
                action = action + np.random.normal(0, EXPLORATION_NOISE, size=action_dim)
                action = np.clip(action, -max_action, max_action)

                next_state, reward, done, truncated, _ = env.step(action)
                done = done or truncated

                if not is_safe:
                    violation_penalty = validator.get_total_penalty(violations)
                    reward += violation_penalty
                
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
            
            episode_time = time.time() - episode_start_time
            stats = validator.get_statistics()
            
            for v in episode_violations:
                if v['type'] == 'fingertip_position':
                    episode_fingertip_violations += 1
                elif v['type'] == 'velocity':
                    episode_velocity_violations += 1

            episode_data = [[
                episode + 1, 
                round(episode_reward, 2), 
                round(episode_time, 2),       
                len(episode_violations), 
                episode_fingertip_violations, 
                episode_velocity_violations, 
                round(best_reward, 2), 
                len(policy.replay_buffer)
            ]]
            
            episode_headers = ['Episode', 'Episode Reward', 
                                'Episode Time', 'Total Violations', 
                                'Fingertip Violations', 'Velocity Violations', 
                                'Best Reward', 'Replay Buffer Size']
            save_training_data(episode_data, training_stats_path, episode_headers, message=False)
            
            if episode % 10 == 0:
                print(f"Episode: {episode}")
                print(f"Reward: {episode_reward:.2f}")
                print(f"Violations: {len(episode_violations)}")
                print(f"Exploration Noise: {EXPLORATION_NOISE:.3f}")
                print(f"Replay Buffer Size: {len(policy.replay_buffer)}")
 
            if (episode + 1) % 100 == 0:

                try:
                    checkpoint_path = f"{save_path}_checkpoint_{episode + 1}"
                    policy.save(checkpoint_path)
                    print(f"\nCheckpoint saved: {checkpoint_path}")
                except Exception as e:
                    print(f"Error when saving checkpoint: {e}")

                violation_data = []
                for joint, count in validator.violation_counts.items():
                    violation_data.append([
                        episode + 1,
                        joint,
                        count,
                        count / validator.total_checks
                    ])
                
                violation_headers = ['Episode', 'Joint', 'Violation Count', 'Violation Rate']
                save_training_data(violation_data, violation_stats_path, violation_headers)
                
                training_time = round(time.time() - start_time, 2)
                episode_stats = [[    
                        episode + 1,
                        total_steps, 
                        training_time,
                        best_reward, 
                        len(policy.replay_buffer), 
                        stats['violation_rate']
                ]]

                episode_headers = ['Episode', 'Total Steps', 'Training Time', 'Best Reward', 'Replay Buffer Size', 'Violation Rate']
                save_training_data(episode_stats, episode_stats_path, episode_headers)
            
            if episode_reward > best_reward:
                best_reward = episode_reward
                try:
                    policy.save(f"{save_path}_best")
                    print(f"\nBest model saved to {save_path}_best")
                    print(f"Best Reward: {best_reward:.2f}")
                except Exception as e:
                    print(f"Error when saving best model: {e}")
                
            # DSMC
            if (episode + 1) % DSMC_CONFIG['check_interval'] == 0:
                print(f"\nDSMC verification started at episode {episode + 1}...")
                
                # 创建验证器
                dsmc_validator = DSMCValidator(env, policy, validator)
                confidence_results = dsmc_validator.run_verification()
                
                # 保存验证结果
                dsmc_data = []
                for state_key, result in confidence_results.items():
                    dsmc_data.append([
                        episode + 1,
                        state_key,
                        result['total_episodes'],
                        result['violation_episodes'],
                        result['violation_rate'],
                        result['empirical_error'] if result['empirical_error'] is not None else 'N/A',
                        result['confidence_bound'] if result['confidence_bound'] is not None else 'N/A',
                        result['has_confidence'],
                        result['samples_needed'],
                        result['min_required_samples'],
                        result['avg_reward'],
                        result['confidence_level']
                    ])

                dsmc_headers = [
                    'Episode', 'Starting State', 'Total Episodes', 'Violation Episodes',
                    'Violation Rate', 'Empirical Error', 'Confidence Bound', 
                    'Has Confidence', 'Samples Needed', 'Min Required Samples',
                    'Average Reward', 'Confidence Level'
                ]
                save_training_data(dsmc_data, dsmc_results_path, dsmc_headers)

                print("\nFixed-width Confidence Intervals Results:")
                total_states = len(confidence_results)
                confident_states = sum(1 for r in confidence_results.values() if r['has_confidence'])
                insufficient_samples = sum(1 for r in confidence_results.values() if r['samples_needed'] > 0)
                
                print(f"Total starting states: {total_states}")
                print(f"States with sufficient samples: {total_states - insufficient_samples}")
                print(f"States meeting confidence criteria: {confident_states}")
                print(f"Overall confidence rate: {confident_states/max(1, total_states-insufficient_samples):.2%}")
                
                print("\nDetailed Confidence Analysis:")
                for state_key, result in confidence_results.items():
                    if result['samples_needed'] > 0:
                        print(f"\nState {state_key}: INSUFFICIENT SAMPLES")
                        print(f"  Current: {result['total_episodes']}, Required: {result['min_required_samples']}")
                    else:
                        status = "CONFIDENT" if result['has_confidence'] else "NOT CONFIDENT"
                        print(f"\nState {state_key}: {status}")
                        print(f"  Empirical error: {result['empirical_error']:.4f}")
                        print(f"  Confidence bound: {result['confidence_bound']:.4f}")
                        print(f"  Threshold k: {dsmc_validator.confidence_calculator.k_threshold}")
                        print(f"  Violation rate: {result['violation_rate']:.2%}")

                # check if ES is needed
                evaluation_stage = DSMCEvaluationStage(
                    env, policy, validator, dsmc_validator.confidence_calculator
                )
                
                if evaluation_stage.set_target_states(confidence_results):
                    print(f"\nStarting DSMC Evaluation Stage...")
                    
                    evaluation_stats = evaluation_stage.run_evaluation_stage(
                        max_evaluation_episodes=1000
                    )
                    
                    eval_data = []
                    for state_key, stats in evaluation_stats.items():
                        eval_data.append([
                            episode + 1,
                            state_key,
                            stats['episodes'],
                            stats['violations'],
                            stats['violations'] / max(1, stats['episodes'])
                        ])
                    
                    eval_headers = ['Episode', 'State Key', 'Eval Episodes', 
                                    'Eval Violations', 'Eval Violation Rate']
                    eval_path = f"{save_path}_evaluation_{timestamp}.csv"
                    save_training_data(eval_data, eval_path, eval_headers)
                    
                    # reassessment
                    print("\nRe-running verification after evaluation stage...")
                    final_confidence_results = dsmc_validator.run_verification()
                    
                    final_dsmc_data = []
                    for state_key, result in final_confidence_results.items():
                        final_dsmc_data.append([
                            episode + 1,
                            state_key,
                            result['total_episodes'],
                            result['violation_episodes'],
                            result['violation_rate'],
                            result['empirical_error'] if result['empirical_error'] is not None else 'N/A',
                            result['confidence_bound'] if result['confidence_bound'] is not None else 'N/A',
                            result['has_confidence'],
                            result['samples_needed'],
                            result['min_required_samples'],
                            result['avg_reward'],
                            result['confidence_level']
                        ])
                    
                    final_dsmc_path = f"{save_path}_final_dsmc_{timestamp}.csv"
                    save_training_data(final_dsmc_data, final_dsmc_path, dsmc_headers)
                    
                    print(f"\nFinal confidence results after evaluation:")
                    total_states = len(final_confidence_results)
                    final_confident_states = sum(1 for r in final_confidence_results.values() 
                                                if r['has_confidence'])
                    print(f"Final confidence rate: {final_confident_states/max(1, total_states):.2%}")
                
                else:
                    print("All states already meet confidence criteria!")

    except KeyboardInterrupt:
        print("\nKeyboard Interrupt. Saving...")

    finally:
        final_stats = {
            'training': training_stats_path,
            'violation': violation_stats_path,
            'dsmc': dsmc_results_path,
            'episode': episode_stats_path
        }
        
        print("\nTraining Data Saving Location:")
        for data_type, path in final_stats.items():
            print(f"{data_type}: {path}")
                
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