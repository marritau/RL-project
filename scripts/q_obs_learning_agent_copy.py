import random
import pickle
import argparse
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from pacman_rldp.env import PacmanEnv, build_env_config
from pacman_rldp.utils import load_yaml
from pacman_rldp.third_party.bk.util import Counter, manhattanDistance
from pacman_rldp.third_party.bk.game import Actions
from pacman_rldp.third_party.bk import graphicsUtils
from PIL import Image, ImageGrab


class SimpleFeatureExtractor:
    """Извлекает расширенные признаки для Approximate Q-learning."""
    
    def get_features(self, state, action):
        features = Counter()
        features["bias"] = 1.0
        
        pacman_pos = state.getPacmanPosition()
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()
        capsules = state.getCapsules()
        
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(pacman_pos[0] + dx), int(pacman_pos[1] + dy)

        # Проверка стены
        if next_x < 0 or next_x >= walls.width or next_y < 0 or next_y >= walls.height or walls[next_x][next_y]:
            features["hit-wall"] = 1.0
            return features

        next_pos = (next_x, next_y)

        # Признак: съел капсулу и может есть призраков
        ghost_timers = getattr(state, "getGhostTimers", None)
        if ghost_timers:
            timers = state.getGhostTimers()
            if any(t > 0 for t in timers):
                features["scared"] = 1.0

        # ------------------------------------------------
        # 1. Съедим ли еду
        # ------------------------------------------------
        if food[next_x][next_y]:
            features["eats-food"] = 1.0

        # ------------------------------------------------
        # 2. Расстояние до ближайшей еды
        # ------------------------------------------------
        dist = self.closest_food(next_pos, food, walls)
        if dist is not None:
            features["closest-food"] = float(dist) / (walls.width + walls.height)

        # ------------------------------------------------
        # 3. Сколько еды рядом (локальная плотность)
        # ------------------------------------------------
        food_count = 0
        for dx2 in [-1, 0, 1]:
            for dy2 in [-1, 0, 1]:
                x = next_x + dx2
                y = next_y + dy2
                if 0 <= x < walls.width and 0 <= y < walls.height:
                    if food[x][y]:
                        food_count += 1
        features["food-nearby"] = food_count / 9.0

        # ------------------------------------------------
        # 4. Расстояние до ближайшего призрака
        # ------------------------------------------------
        if ghosts:
            min_ghost_dist = min(manhattanDistance(next_pos, g) for g in ghosts)
            features["ghost-distance"] = float(min_ghost_dist) / (walls.width + walls.height)

            if min_ghost_dist <= 1:
                features["danger"] = 1.0

        # ------------------------------------------------
        # 5. Расстояние до капсулы
        # ------------------------------------------------
        if capsules:
            capsule_dist = min(manhattanDistance(next_pos, c) for c in capsules)
            features["capsule-distance"] = float(capsule_dist) / (walls.width + walls.height)

        # ------------------------------------------------
        # 6. Stop действие
        # ------------------------------------------------
        if action == "Stop":
            features["stop"] = 1.0

        return features


    def closest_food(self, pos, food, walls):
        fringe = [(pos[0], pos[1], 0)]
        expanded = set()

        while fringe:
            pos_x, pos_y, dist = fringe.pop(0)

            if (pos_x, pos_y) in expanded:
                continue

            expanded.add((pos_x, pos_y))

            if food[pos_x][pos_y]:
                return dist

            for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                nx = pos_x + dx
                ny = pos_y + dy

                if 0 <= nx < walls.width and 0 <= ny < walls.height and not walls[nx][ny]:
                    fringe.append((nx, ny, dist + 1))

        return None


class QLearningAgent:
    """Approximate Q-learning Agent."""
    def __init__(self, action_space_size, alpha=0.005, gamma=0.95, epsilon=0.05):
        self.weights = Counter()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.feature_extractor = SimpleFeatureExtractor()

    def get_q_value(self, state, action_id):
        action = ["North", "South", "East", "West", "Stop"][action_id]
        features = self.feature_extractor.get_features(state, action)
        return features * self.weights

    def choose_action(self, state, legal_actions):
        if not legal_actions:
            return 4
        if random.random() < self.epsilon:
            return random.choice(legal_actions)
        
        q_values = {a: self.get_q_value(state, a) for a in legal_actions}
        max_q = max(q_values.values())
        best_actions = [a for a, q in q_values.items() if q == max_q]
        return random.choice(best_actions)

    def learn(self, state, action, reward, next_state, done):
        q_value = self.get_q_value(state, action)
        # Reward shaping
        action_str = ["North", "South", "East", "West", "Stop"][action]
        features = self.feature_extractor.get_features(state, action_str)
        if features.get("stop", 0.0) == 1.0:
            reward -= 5
        if features.get("hit-wall", 0.0) == 1.0:
            reward -= 10
        if done:
            target = reward
        else:
            # V(s') = max_a Q(s', a) для легальных действий
            legal_next = next_state.getLegalPacmanActions()
            if not legal_next:
                next_max_q = 0
            else:
                action_map = ["North", "South", "East", "West", "Stop"]
                q_values = [self.get_q_value(next_state, action_map.index(a)) for a in legal_next]
                next_max_q = max(q_values)
            target = reward + self.gamma * next_max_q
            
        diff = target - q_value
        for feature, value in features.items():
            self.weights[feature] += self.alpha * diff * value

    def save(self, path=None):
        if path is None:
            path = "q_obs_weights_copy.pkl"
        with open(path, "wb") as f:
            pickle.dump(dict(self.weights), f)

    def load(self, path):
        with open(path, "rb") as f:
            w_dict = pickle.load(f)
            self.weights = Counter(w_dict)


def capture_human_frame() -> Image.Image | None:
    """Capture one frame from the active Tk canvas (через postscript, без рамок)."""
    canvas = graphicsUtils._canvas
    if canvas is None:
        return None
    try:
        import PIL.ImageGrab
        canvas.update()
        x = canvas.winfo_rootx()
        y = canvas.winfo_rooty()
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        bbox = (x, y, x + w, y + h)
        img = PIL.ImageGrab.grab(bbox)
        return img
    except Exception:
        return None


def save_gif(frames: list[Image.Image], output_path: Path, frame_time: float) -> None:
    """Save accumulated frames into an animated GIF."""
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


def train(env, agent, episodes):
    print(f"Starting training for {episodes} episodes...")
    rewards_history = []
    scores_history = []
    
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
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        
        rewards_history.append(total_reward)
        scores_history.append(info.get("score", 0))
        if (i + 1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(f"Episode {i+1}/{episodes}, Avg Reward: {avg_reward:.2f}, Weights: {dict(agent.weights)}")
    # Сохраняем график обучения по total_reward
    Path("results/important").mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    window = 200
    avg_rewards = [np.mean(rewards_history[max(0, i-window):i+1]) for i in range(len(rewards_history))]
    plt.plot(avg_rewards, label="Avg total reward (window=200)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Q-Learning Pacman Training Total Reward (Moving Average)")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/important/q_learn.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Q-Learning Agent for Pacman")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes to train")
    parser.add_argument("--train", action="store_true", help="Run in training mode")
    parser.add_argument("--eval", action="store_true", help="Run in evaluation mode")
    parser.add_argument("--model", type=str, default="q_weights.pkl", help="Path to save/load model")
    parser.add_argument("--render", action="store_true", help="Render during evaluation")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for the environment")
    parser.add_argument("--record", action="store_true", help="Record frames for GIF")
    parser.add_argument("--record-dir", type=str, default="frames", help="Directory to save frames")
    args = parser.parse_args()

    config_dict = load_yaml(args.config)
    env_config = build_env_config(config_dict)
    
    # Override seed if provided via CLI
    if args.seed is not None:
        env_config.seed = args.seed
        
    # Force human render mode if requested
    if args.render or args.record:
        env_config.render_mode = "human"
        
    env = PacmanEnv(env_config)

    agent = QLearningAgent(action_space_size=5)

    if args.train:
        train(env, agent, args.episodes)
        Path("results").mkdir(exist_ok=True) # Ensure results directory exists
        agent.save("q_obs_weights_copy.pkl")
        print(f"Model saved to q_obs_weights_copy.pkl")

    if args.eval:
        if not Path(args.model).exists():
            print(f"Model file {args.model} not found!")
            return
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
                action = agent.choose_action(state, info.get("legal_action_ids", []))
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
            output_path = Path("results/important/q_learn.gif")
            save_gif(gif_frames, output_path, env_config.frame_time)
            print(f"GIF saved to {output_path}")

    env.close()


if __name__ == "__main__":
    main()
