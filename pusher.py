import gymnasium as gym
import numpy as np
import torch
import time
import csv
from datetime import datetime
from models import TD3, Actor, Critic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

FINGERTIP_LIMITS = {
    'x_range': (-1, 1),   
    'y_range': (-0.7, 1), 
    'z_range': (-0.5, 0.7), 
    'back_penalty': -1
}

class DemoValidator:
    def __init__(self):
        self.joint_limits = JOINT_LIMITS
        self.violations = []
        self.total_checks = 0
        self.violation_counts = {joint: 0 for joint in JOINT_LIMITS.keys()}
        self.fingertip_violations = []
        self.episode_violations = []
        
    def check_fingertip_position(self, env):
        #check violations of fingertip
        real_env = env
        while hasattr(real_env, 'env'):
            real_env = real_env.env
        
        fingertip_pos = real_env.get_body_com("tips_arm")
        violations = []
        
        # check y axle limit (back direction)
        if fingertip_pos[1] < FINGERTIP_LIMITS['y_range'][0]:
            violations.append({
                'type': 'fingertip_position',
                'axis': 'y',
                'value': fingertip_pos[1],
                'limit': FINGERTIP_LIMITS['y_range'],
                'timestamp': self.total_checks,
                'description': 'Fingertip moving backward'
            })
        
        # check x and z axles
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
                    'description': desc
                })
        
        if violations:
            self.fingertip_violations.extend(violations)
            return False, violations
        return True, []
    
    def check_state(self, state, env):
        # check joint violations
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
                    'timestamp': self.total_checks,
                    'step': self.total_checks,
                    'description': f'{joint} joint exceeded speed limit'
                }
                current_violations.append(violation)
                self.violation_counts[joint] += 1
        
        # check fingertip
        is_safe_fingertip, fingertip_violations = self.check_fingertip_position(env)
        if not is_safe_fingertip:
            current_violations.extend(fingertip_violations)
        
        if current_violations:
            self.violations.extend(current_violations)
            self.episode_violations.extend(current_violations)
            return False, current_violations
        return True, []
    
    def get_episode_summary(self, episode_num):
        velocity_violations = sum(1 for v in self.episode_violations if v['type'] == 'velocity')
        fingertip_violations = sum(1 for v in self.episode_violations if v['type'] == 'fingertip_position')
        
        summary = {
            'episode': episode_num,
            'total_violations': len(self.episode_violations),
            'velocity_violations': velocity_violations,
            'fingertip_violations': fingertip_violations,
            'violations': self.episode_violations.copy()
        }
        
        # clear current log
        self.episode_violations = []
        return summary
    
    def get_statistics(self):
        return {
            'total_checks': self.total_checks,
            'total_violations': len(self.violations),
            'violation_counts': self.violation_counts,
            'violation_rate': len(self.violations) / max(1, self.total_checks)
        }

def save_violation_records(violation_data, filename):
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            if violation_data:
                headers = ['Episode', 'Step', 'Violation_Type', 'Joint/Axis', 
                          'Value', 'Limit_Min', 'Limit_Max', 'Description', 'Timestamp']
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(violation_data)
        print(f"Violation log saved to: {filename}")
    except Exception as e:
        print(f"Violation log save failed: {e}")

def save_episode_summary(summary_data, filename):
    """保存回合摘要到CSV文件"""
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            headers = ['Episode', 'Total_Violations', 'Velocity_Violations', 
                      'Fingertip_Violations', 'Episode_Reward', 'Episode_Steps']
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(summary_data)
        print(f"Episode summary saved to: {filename}")
    except Exception as e:
        print(f"Episode summary save failed: {e}")

def demo_agent(model_path, episodes=10, timesteps=500):
    env = gym.make("Pusher-v5", render_mode="human", max_episode_steps=timesteps)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    policy = TD3(state_dim, action_dim, max_action)
    validator = DemoValidator()

    all_violations = []
    episode_summaries = []
    
    try:
        policy.load(model_path)
        print(f"\nModel loaded: {model_path}")
        
        total_reward = 0
        
        for episode in range(episodes):
            state = env.reset()[0]
            episode_reward = 0
            episode_steps = 0
            
            print(f"\n回合 {episode + 1} 开始...")
            
            for t in range(timesteps):
                is_safe, violations = validator.check_state(state, env)
                
                if not is_safe:
                    print(f"  step {t+1}: {len(violations)} violations detected")
                    for v in violations:
                        print(f"    - {v['description']}: {v['value']:.3f}")

                action = policy.select_action(state)

                next_state, reward, done, truncated, _ = env.step(action)
                done = done or truncated
                
                episode_reward += reward
                episode_steps += 1
                state = next_state
                
                # slow render
                time.sleep(0.01)
                
                if done:
                    break
            
            episode_summary = validator.get_episode_summary(episode + 1)
            episode_summaries.append([
                episode + 1,
                episode_summary['total_violations'],
                episode_summary['velocity_violations'],
                episode_summary['fingertip_violations'],
                round(episode_reward, 2),
                episode_steps
            ])
            
            for violation in episode_summary['violations']:
                limit_min = violation['limit'][0] if isinstance(violation['limit'], (list, tuple)) else 'N/A'
                limit_max = violation['limit'][1] if isinstance(violation['limit'], (list, tuple)) else 'N/A'
                
                all_violations.append([
                    episode + 1,
                    violation['timestamp'],
                    violation['type'],
                    violation.get('joint', violation.get('axis', 'N/A')),
                    round(violation['value'], 6),
                    limit_min,
                    limit_max,
                    violation['description'],
                    violation['timestamp']
                ])
            
            print(f"Episode {episode + 1} ended")
            print(f"Reward: {episode_reward:.2f}")
            print(f"Violations: {episode_summary['total_violations']}")
            print(f"  - Speed violations: {episode_summary['velocity_violations']}")
            print(f"  - Workspace violations: {episode_summary['fingertip_violations']}")
            
            total_reward += episode_reward
        
        avg_reward = total_reward / episodes
        total_stats = validator.get_statistics()
        
        print(f"\nSummary")
        print(f"Average reward: {avg_reward:.2f}")
        print(f"total checks: {total_stats['total_checks']}")
        print(f"total violations: {total_stats['total_violations']}")
        print(f"violation rate: {total_stats['violation_rate']:.4f}")
        print(f"joint violation counts:")
        for joint, count in total_stats['violation_counts'].items():
            print(f"  - {joint}: {count}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        violation_filename = f"demo_violations_{timestamp}.csv"
        save_violation_records(all_violations, violation_filename)
        
        # 保存回合摘要
        summary_filename = f"demo_summary_{timestamp}.csv"
        save_episode_summary(episode_summaries, summary_filename)
        
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        env.close()

if __name__ == "__main__":

    model_path = "./training_model_ver/model_best"
    
    demo_agent(
        model_path=model_path,
        episodes=100,
        timesteps=500
    )