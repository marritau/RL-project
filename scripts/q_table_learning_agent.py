import random
import pickle
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageGrab
import matplotlib.pyplot as plt
from pacman_rldp.env import PacmanEnv, build_env_config
from pacman_rldp.utils import load_yaml
from pacman_rldp.third_party.bk.game import Actions
from pacman_rldp.third_party.bk.util import Counter, manhattanDistance

# Захват кадра с canvas (аналогично q_obs_learning_agent_copy)
def capture_human_frame():
    from pacman_rldp.third_party.bk import graphicsUtils
    canvas = graphicsUtils._canvas
    if canvas is None:
        return None
    try:
        canvas.update()
        x = canvas.winfo_rootx()
        y = canvas.winfo_rooty()
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        bbox = (x, y, x + w, y + h)
        img = ImageGrab.grab(bbox)
        return img
    except Exception:
        return None

# Сохранение GIF
def save_gif(frames, output_path, frame_time):
    if not frames:
        return
    duration_ms = max(20, int(frame_time * 1000))
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        optimize=False,
        duration=duration_ms,
        loop=0,
    )

# Признаковая дискретизация (как в q_obs_learning_agent_copy)
def feature_key(state, action):
    pacman_pos = state.getPacmanPosition()
    food = state.getFood()
    walls = state.getWalls()
    ghosts = state.getGhostPositions()
    capsules = state.getCapsules()
    dx, dy = Actions.directionToVector(action)
    next_x, next_y = int(pacman_pos[0] + dx), int(pacman_pos[1] + dy)

    # hit-wall
    hit_wall = (
        next_x < 0 or next_x >= walls.width or next_y < 0 or next_y >= walls.height or walls[next_x][next_y]
    )

    # scared
    ghost_timers = getattr(state, "getGhostTimers", None)
    scared = 0
    if ghost_timers:
        timers = state.getGhostTimers()
        if any(t > 0 for t in timers):
            scared = 1

    # eats-food
    eats_food = int(not hit_wall and food[next_x][next_y])

    # closest-food
    def closest_food(pos, food, walls):
        fringe = [(pos[0], pos[1], 0)]
        expanded = set()
        while fringe:
            pos_x, pos_y, dist = fringe.pop(0)
            if (pos_x, pos_y) in expanded:
                continue
            expanded.add((pos_x, pos_y))
            if food[pos_x][pos_y]:
                return dist
            for dx2, dy2 in [(0,1),(0,-1),(1,0),(-1,0)]:
                nx = pos_x + dx2
                ny = pos_y + dy2
                if 0 <= nx < walls.width and 0 <= ny < walls.height and not walls[nx][ny]:
                    fringe.append((nx, ny, dist + 1))
        return None
    closest_food_dist = None if hit_wall else closest_food((next_x, next_y), food, walls)
    # food-nearby
    food_count = 0
    for dx2 in [-1, 0, 1]:
        for dy2 in [-1, 0, 1]:
            x = next_x + dx2
            y = next_y + dy2
            if 0 <= x < walls.width and 0 <= y < walls.height:
                if food[x][y]:
                    food_count += 1
    food_nearby = food_count
    # ghost-distance, danger
    min_ghost_dist = None
    danger = 0
    if ghosts and not hit_wall:
        min_ghost_dist = min(manhattanDistance((next_x, next_y), g) for g in ghosts)
        if min_ghost_dist <= 1:
            danger = 1
    # capsule-distance
    capsule_dist = None
    if capsules and not hit_wall:
        capsule_dist = min(manhattanDistance((next_x, next_y), c) for c in capsules)
    # stop
    stop = int(action == "Stop")
    # Собираем кортеж признаков (можно добавить/убрать по необходимости)
    return (
        int(hit_wall),
        scared,
        eats_food,
        closest_food_dist if closest_food_dist is not None else -1,
        food_nearby,
        min_ghost_dist if min_ghost_dist is not None else -1,
        danger,
        capsule_dist if capsule_dist is not None else -1,
        stop
    )

class QTableAgent:
    def __init__(self, alpha=0.1, gamma=0.95, epsilon=0.1):
        self.q_table = dict()  # (state_key, action_id) -> value
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_map = ["North", "South", "East", "West", "Stop"]

    def get_q_value(self, state, action_id):
        action = self.action_map[action_id]
        key = feature_key(state, action)
        return self.q_table.get(key, 0.0)

    def choose_action(self, state, legal_actions):
        if not legal_actions:
            return 4
        if random.random() < self.epsilon:
            return random.choice(legal_actions)
        q_values = {a: self.get_q_value(state, a) for a in legal_actions}
        max_q = max(q_values.values())
        best_actions = [a for a, q in q_values.items() if q == max_q]
        return random.choice(best_actions)

    def learn(self, state, action, reward, next_state, done, legal_next):
        action_str = self.action_map[action]
        key = feature_key(state, action_str)
        q = self.q_table.get(key, 0.0)
        if done or not legal_next:
            target = reward
        else:
            next_qs = [self.get_q_value(next_state, a) for a in legal_next]
            target = reward + self.gamma * max(next_qs)
        self.q_table[key] = q + self.alpha * (target - q)

    def save(self, path="q_table.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.q_table, f)

    def load(self, path="q_table.pkl"):
        with open(path, "rb") as f:
            self.q_table = pickle.load(f)

def train(env, agent, episodes):
    rewards_history = []
    for i in range(episodes):
        obs, info = env.reset()
        state = env.runtime_state
        total_reward = 0
        done = False
        while not done:
            legal_actions = info.get("legal_action_ids", [])
            action = agent.choose_action(state, legal_actions)
            _, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = env.runtime_state
            legal_next = info.get("legal_action_ids", [])
            agent.learn(state, action, reward, next_state, done, legal_next)
            state = next_state
            total_reward += reward
        rewards_history.append(total_reward)
        if (i + 1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(f"Episode {i+1}/{episodes}, Avg Reward: {avg_reward:.2f}")
    # Сохраняем график обучения
    Path("results/important").mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    window = 200
    avg_rewards = [np.mean(rewards_history[max(0, i-window):i+1]) for i in range(len(rewards_history))]
    plt.plot(avg_rewards, label="Avg total reward (window=200)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Q-Table Pacman Training Total Reward (Moving Average)")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/important/q_table_learn.png")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Q-Table Agent for Pacman")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes to train")
    parser.add_argument("--train", action="store_true", help="Run in training mode")
    parser.add_argument("--eval", action="store_true", help="Run in evaluation mode")
    parser.add_argument("--model", type=str, default="q_table.pkl", help="Path to save/load model")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for the environment")
    parser.add_argument("--record", action="store_true", help="Record frames for GIF during evaluation")
    parser.add_argument("--render", action="store_true", help="Render Pacman window during evaluation")
    args = parser.parse_args()

    config_dict = load_yaml(args.config)
    env_config = build_env_config(config_dict)
    if args.seed is not None:
        env_config.seed = args.seed
    # Включить визуализацию, если render или record
    if args.render or args.record:
        env_config.render_mode = "human"
    env = PacmanEnv(env_config)
    agent = QTableAgent()

    if args.train:
        train(env, agent, args.episodes)
        print(f"Q-table size: {len(agent.q_table)}")
        agent.save(args.model)
        print(f"Q-table saved to {args.model}")

    if args.eval:
        agent.load(args.model)
        agent.epsilon = 0.0
        wins = 0
        total_score = 0
        gif_frames = []
        for i in range(args.episodes):
            obs, info = env.reset(seed=(env_config.seed if env_config.seed is not None else 0) + i)
            state = env.runtime_state
            done = False
            while not done:
                legal_actions = info.get("legal_action_ids", [])
                action = agent.choose_action(state, legal_actions)
                _, _, terminated, truncated, info = env.step(action)
                state = env.runtime_state
                done = terminated or truncated
                if args.record:
                    frame = capture_human_frame()
                    if frame: gif_frames.append(frame)
            if info.get("is_win", False):
                wins += 1
            total_score += info.get("score", 0)
            if (i+1) % 100 == 0:
                print(f"Episode {i+1}: win={info.get('is_win', False)}, score={info.get('score', 0)}")
        print(f"Win rate: {wins / args.episodes:.3f}, Avg score: {total_score / args.episodes:.2f}")
        print("Evaluation finished.")
        if args.record and gif_frames:
            Path("results/important").mkdir(parents=True, exist_ok=True)
            output_path = Path("results/important/q_table_learn.gif")
            frame_time = getattr(env_config, "frame_time", 0.1)
            save_gif(gif_frames, output_path, frame_time)
            print(f"GIF saved to {output_path}")
    env.close()

if __name__ == "__main__":
    main()
