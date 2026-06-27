import numpy as np
import random
import ast
from .base import BaseAgent

# class SarsaAgent(BaseAgent):
#     def __init__(self, alpha=0.1, gamma=0.99, epsilon=0.1, action_size=5):
#         self.alpha = alpha      
#         self.gamma = gamma      
#         self.epsilon = epsilon  
#         self.action_size = action_size
#         self.q_table = {}      

#     def _get_state_key(self, observation: dict[str, np.ndarray]) -> tuple:
#         pac_pos = tuple(observation["pacman_position"].astype(int))
#         ghost_pos = tuple(map(tuple, observation["ghost_positions"].astype(int)))
#         food_indices = np.argwhere(observation["food"] == 1)
#         food_key = tuple(map(tuple, food_indices))
        
#         return (pac_pos, ghost_pos, food_key)

#     def _get_q_values(self, state_key):
#         if state_key not in self.q_table:
#             self.q_table[state_key] = np.zeros(self.action_size)
#         return self.q_table[state_key]

#     def select_action(self, observation, info) -> int:
#         state_key = self._get_state_key(observation)
#         legal_actions = info.get("legal_action_ids", list(range(self.action_size)))

#         if random.random() < self.epsilon:
#             return random.choice(legal_actions)
        
#         q_values = self._get_q_values(state_key)
#         masked_q = np.full(self.action_size, -np.inf)
#         for a in legal_actions:
#             masked_q[a] = q_values[a]
        
#         return int(np.argmax(masked_q))

#     def update(self, state, action, reward, next_state, next_action, terminated):
#         """SARSA: Q(s,a) = Q(s,a) + alpha * [R + gamma*Q(s',a') - Q(s,a)]"""
#         s_key = self._get_state_key(state)
#         n_s_key = self._get_state_key(next_state)
        
#         current_q = self._get_q_values(s_key)[action]
#         next_q = self._get_q_values(n_s_key)[next_action] if not terminated else 0
        
#         new_q = current_q + self.alpha * (reward + self.gamma * next_q - current_q)
#         self.q_table[s_key][action] = new_q

import numpy as np
import random
import yaml
from pathlib import Path
from .base import BaseAgent

# class SarsaAgent(BaseAgent):
#     def __init__(self, alpha=0.1, gamma=0.99, epsilon=0.1, action_size=5):
#         self.alpha = alpha
#         self.gamma = gamma
#         self.epsilon = epsilon
#         self.action_size = action_size
#         self.q_table = {}  # {(state_key): [q_values]}


#     def _get_state_key(self, observation: dict[str, np.ndarray]) -> tuple:
#         pac_pos = observation["pacman_position"]
#         ghost_pos = observation["ghost_positions"]
#         food_grid = observation["food"]

#         food_indices = np.argwhere(food_grid == 1)
#         if len(food_indices) > 0:
#             distances = np.abs(food_indices - pac_pos).sum(axis=1)
#             closest_food = tuple(food_indices[np.argmin(distances)] - pac_pos)
#         else:
#             closest_food = (0, 0)

#         ghost_dirs = []
#         for g_pos in ghost_pos:
#             dist = np.abs(g_pos - pac_pos).sum()
#             if dist < 3: 
#                 ghost_dirs.append(tuple(g_pos - pac_pos))
#             else:
#                 ghost_dirs.append(None) 

#         return (tuple(pac_pos), closest_food, tuple(ghost_dirs))

#     def _get_q_values(self, state_key):
#         if state_key not in self.q_table:
#             self.q_table[state_key] = np.zeros(self.action_size)
#         return self.q_table[state_key]

#     def select_action(self, observation, info) -> int:
#         state_key = self._get_state_key(observation)
#         legal_actions = info.get("legal_action_ids", list(range(self.action_size)))

#         if random.random() < self.epsilon:
#             return random.choice(legal_actions)
        
#         q_values = self._get_q_values(state_key)
#         masked_q = np.full(self.action_size, -np.inf)
#         for a in legal_actions:
#             masked_q[a] = q_values[a]
        
#         max_q = np.max(masked_q)
#         best_actions = np.where(masked_q == max_q)[0]
#         return int(random.choice(best_actions))

#     # def update(self, state, action, reward, next_state, next_action, terminated)
#     def update(self, state, action, reward, next_state, next_action, terminated, next_info=None):
#         """Обновление Q-таблицы с защитой от numpy-массивов."""
#         s_key = self._get_state_key(state)
#         n_s_key = self._get_state_key(next_state)

#         reward_val = float(np.asarray(reward).item())
#         action_idx = int(np.asarray(action).item())

#         q_values = self._get_q_values(s_key)
#         current_q = float(q_values[action_idx])

#         if terminated or next_action is None:
#             next_q = 0.0
#         else:
#             next_act_idx = int(np.asarray(next_action).item())
#             next_q = float(self._get_q_values(n_s_key)[next_act_idx])
            
#         new_q = current_q + self.alpha * (reward_val + self.gamma * next_q - current_q)

#         q_values[action_idx] = new_q

#     def save_policy(self, filepath: str | Path):
#         """Сохраняет Q-таблицу в YAML файл."""
#         serializable_table = {str(k): v.tolist() for k, v in self.q_table.items()}

#         data = {
#             "hyperparameters": {
#                 "alpha": self.alpha,
#                 "gamma": self.gamma,
#                 "action_size": self.action_size
#             },
#             "q_table": serializable_table
#         }

#         with open(filepath, 'w') as f:
#             yaml.dump(data, f, default_flow_style=False)
#         print(f"Policy saved to {filepath}")

#     def load_policy(self, filepath: str | Path):
#         """Загружает Q-таблицу из YAML файла."""
#         with open(filepath, 'r') as f:
#             data = yaml.safe_load(f)

#         hp = data["hyperparameters"]
#         self.alpha = hp["alpha"]
#         self.gamma = hp["gamma"]
#         self.action_size = hp["action_size"]

#         self.q_table = {eval(k): np.array(v) for k, v in data["q_table"].items()}
#         print(f"Policy loaded from {filepath}. States in table: {len(self.q_table)}")


# class SarsaAgent(BaseAgent):
#     def __init__(self, alpha=0.1, gamma=0.99, epsilon=0.1, action_size=5):
#         self.alpha = alpha
#         self.gamma = gamma
#         self.epsilon = epsilon
#         self.action_size = action_size
#         self.q_table = {}
#         # Отдельный генератор случайных чисел для воспроизводимости
#         self._rng = random.Random()

#     # def _get_state_key(self, observation: dict[str, np.ndarray]) -> tuple:
#     #     pac_pos = tuple(observation["pacman_position"].astype(int))
#     #     pac_arr = observation["pacman_position"]

#     #     ghost_features = []
#     #     for i, present in enumerate(observation["ghost_present"]):
#     #         if not present:
#     #             continue
#     #         g = observation["ghost_positions"][i]
#     #         diff = g - pac_arr
#     #         dist = float(np.abs(diff).sum())
#     #         direction = (int(np.sign(diff[0])), int(np.sign(diff[1])))
#     #         bucket = 0 if dist <= 2 else (1 if dist <= 6 else 2)
#     #         scared = int(observation["ghost_timers"][i] > 0)
#     #         ghost_features.append((direction, bucket, scared))

#     #     ghost_features.sort(key=lambda x: (x[2], x[1]))
#     #     ghost_key = tuple(ghost_features)

#     #     food_indices = np.argwhere(observation["food"] == 1)
#     #     if len(food_indices) > 0:
#     #         diffs = food_indices - pac_arr.astype(int)
#     #         distances = np.abs(diffs).sum(axis=1)
#     #         closest = diffs[np.argmin(distances)]
#     #         food_dir = (int(np.sign(closest[0])), int(np.sign(closest[1])))
#     #         dist_val = int(distances.min())
#     #         food_bucket = 0 if dist_val <= 2 else (1 if dist_val <= 5 else 2)
#     #     else:
#     #         food_dir = (0, 0)
#     #         food_bucket = 3

#     #     food_count = int(np.sum(observation["food"]))
#     #     food_count_bucket = 0 if food_count <= 3 else (1 if food_count <= 10 else 2)

#     #     return (pac_pos, ghost_key, food_dir, food_bucket, food_count_bucket)

#     def _get_state_key(self, observation: dict[str, np.ndarray]) -> tuple:
#         pac_pos = tuple(observation["pacman_position"].astype(int))

#         # Призраки — оставляем свою логику, она богаче чем в observation spec
#         pac_arr = observation["pacman_position"]
#         ghost_features = []
#         for i, present in enumerate(observation["ghost_present"]):
#             if not present:
#                 continue
#             g = observation["ghost_positions"][i]
#             diff = g - pac_arr
#             dist = float(np.abs(diff).sum())
#             direction = (int(np.sign(diff[0])), int(np.sign(diff[1])))
#             bucket = 0 if dist <= 2 else (1 if dist <= 6 else 2)
#             scared = int(observation["ghost_timers"][i] > 0)
#             ghost_features.append((direction, bucket, scared))
#         ghost_features.sort(key=lambda x: (x[2], x[1]))

#         if "nearest_food_bucket" in observation:
#             # Используем готовые признаки из BITMASK_DISTANCE_BUCKETS
#             food_bucket = int(observation["nearest_food_bucket"][0])
#             food_dir    = int(observation["nearest_food_direction"][0])
#             # food_bitmask кодирует точное состояние еды — используем как есть
#             food_state  = int(observation["food_bitmask"])
#             return (pac_pos, tuple(ghost_features), food_dir, food_bucket, food_state)
#         else:
#             # Fallback для RAW observation — старая логика
#             food_indices = np.argwhere(observation["food"] == 1)
#             if len(food_indices) > 0:
#                 diffs = food_indices - pac_arr.astype(int)
#                 distances = np.abs(diffs).sum(axis=1)
#                 closest = diffs[np.argmin(distances)]
#                 food_dir = (int(np.sign(closest[0])), int(np.sign(closest[1])))
#                 dist_val = int(distances.min())
#                 food_bucket = 0 if dist_val <= 2 else (1 if dist_val <= 5 else 2)
#             else:
#                 food_dir, food_bucket = (0, 0), 3
#             food_count = int(np.sum(observation["food"]))
#             food_count_bucket = 0 if food_count <= 3 else (1 if food_count <= 10 else 2)
#             return (pac_pos, tuple(ghost_features), food_dir, food_bucket, food_count_bucket)

#     def _get_q_values(self, state_key):
#         if state_key not in self.q_table:
#             self.q_table[state_key] = np.zeros(self.action_size)
#         return self.q_table[state_key]
 
#     def select_action(self, observation, info) -> int:
#         state_key = self._get_state_key(observation)
#         legal_actions = info.get("legal_action_ids", list(range(self.action_size)))
 
#         if self._rng.random() < self.epsilon:
#             return self._rng.choice(legal_actions)
 
#         q_values = self._get_q_values(state_key)
#         masked_q = np.full(self.action_size, -np.inf)
#         for a in legal_actions:
#             masked_q[a] = q_values[a]
 
#         max_q = np.max(masked_q)
#         best_actions = np.where(masked_q == max_q)[0].tolist()
#         return int(self._rng.choice(best_actions))
 
#     def update(self, state, action, reward, next_state, next_action, terminated):
#         s_key = self._get_state_key(state)
#         n_s_key = self._get_state_key(next_state)
 
#         reward_val = float(np.asarray(reward).item())
#         action_idx = int(np.asarray(action).item())
 
#         q_values = self._get_q_values(s_key)
#         current_q = float(q_values[action_idx])
 
#         if terminated or next_action is None:
#             next_q = 0.0
#         else:
#             next_act_idx = int(np.asarray(next_action).item())
#             next_q = float(self._get_q_values(n_s_key)[next_act_idx])
 
#         q_values[action_idx] = (1.0 - self.alpha) * current_q + self.alpha * (reward_val + self.gamma * next_q - current_q)
 
#     def seed(self, seed: int) -> None:
#         self._rng.seed(seed)
 
#     def save_policy(self, filepath: str | Path):
#         serializable_table = {str(k): v.tolist() for k, v in self.q_table.items()}
#         data = {
#             "hyperparameters": {
#                 "alpha": self.alpha,
#                 "gamma": self.gamma,
#                 "action_size": self.action_size,
#             },
#             "q_table": serializable_table,
#         }
#         with open(filepath, "w") as f:
#             yaml.dump(data, f, default_flow_style=False)
#         print(f"Policy saved to {filepath}")
 
#     def load_policy(self, filepath: str | Path):
#         with open(filepath, "r") as f:
#             data = yaml.safe_load(f)
 
#         hp = data["hyperparameters"]
#         self.alpha = hp["alpha"]
#         self.gamma = hp["gamma"]
#         self.action_size = hp["action_size"]
 
#         # ast.literal_eval вместо eval для безопасной десериализации ключей
#         self.q_table = {ast.literal_eval(k): np.array(v) for k, v in data["q_table"].items()}
#         print(f"Policy loaded from {filepath}. States in table: {len(self.q_table)}")
 
import ast
import random
from pathlib import Path

import numpy as np
import yaml

from .base import BaseAgent


class SarsaAgent(BaseAgent):
    def __init__(self, alpha=0.1, gamma=0.99, epsilon=0.1, action_size=5):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_size = action_size
        self.q_table = {}
        self._rng = random.Random()

    def _get_ghost_features(self, observation: dict) -> tuple:
        pac_arr = observation["pacman_position"]
        ghost_features = []
        for i, present in enumerate(observation["ghost_present"]):
            if not present:
                continue
            g = observation["ghost_positions"][i]
            diff = g - pac_arr
            dist = float(np.abs(diff).sum())
            direction = (int(np.sign(diff[0])), int(np.sign(diff[1])))
            bucket = 0 if dist <= 2 else (1 if dist <= 6 else 2)
            scared = int(observation["ghost_timers"][i] > 0)
            ghost_features.append((direction, bucket, scared))
        ghost_features.sort(key=lambda x: (x[2], x[1]))
        return tuple(ghost_features)

    def _get_state_key(self, observation: dict[str, np.ndarray]) -> tuple:
        pac_pos = tuple(observation["pacman_position"].astype(int))
        ghost_key = self._get_ghost_features(observation)

        if "nearest_food_bucket" in observation:
            # Ветка BITMASK_DISTANCE_BUCKETS:
            # готовые признаки из observation_spec — не пересчитываем вручную
            food_dir    = int(observation["nearest_food_direction"][0])
            food_bucket = int(observation["nearest_food_bucket"][0])
            # food_bitmask не используем напрямую — 2^50 уникальных значений
            # считаем количество единичных битов как число оставшейся еды
            food_count  = bin(int(observation["food_bitmask"])).count("1")
            food_count_bucket = 0 if food_count <= 3 else (1 if food_count <= 10 else 2)
            return (pac_pos, ghost_key, food_dir, food_bucket, food_count_bucket)

        # Ветка RAW observation
        pac_arr = observation["pacman_position"]
        food_indices = np.argwhere(observation["food"] == 1)
        if len(food_indices) > 0:
            diffs = food_indices - pac_arr.astype(int)
            distances = np.abs(diffs).sum(axis=1)
            closest = diffs[np.argmin(distances)]
            food_dir = (int(np.sign(closest[0])), int(np.sign(closest[1])))
            dist_val = int(distances.min())
            food_bucket = 0 if dist_val <= 2 else (1 if dist_val <= 5 else 2)
        else:
            food_dir, food_bucket = (0, 0), 3
        food_count = int(np.sum(observation["food"]))
        food_count_bucket = 0 if food_count <= 3 else (1 if food_count <= 10 else 2)
        return (pac_pos, ghost_key, food_dir, food_bucket, food_count_bucket)

    def _get_q_values(self, state_key: tuple) -> np.ndarray:
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        return self.q_table[state_key]

    def select_action(self, observation: dict, info: dict) -> int:
        state_key = self._get_state_key(observation)
        legal_actions = info.get("legal_action_ids", list(range(self.action_size)))

        if self._rng.random() < self.epsilon:
            return self._rng.choice(legal_actions)

        q_values = self._get_q_values(state_key)
        masked_q = np.full(self.action_size, -np.inf)
        for a in legal_actions:
            masked_q[a] = q_values[a]

        max_q = np.max(masked_q)
        best_actions = np.where(masked_q == max_q)[0].tolist()
        return int(self._rng.choice(best_actions))

    def update(self, state, action, reward, next_state, next_action, terminated):
        s_key  = self._get_state_key(state)
        ns_key = self._get_state_key(next_state)

        reward_val = float(np.asarray(reward).item())
        action_idx = int(np.asarray(action).item())
        current_q  = float(self._get_q_values(s_key)[action_idx])

        if terminated or next_action is None:
            next_q = 0.0
        else:
            next_act_idx = int(np.asarray(next_action).item())
            next_q = float(self._get_q_values(ns_key)[next_act_idx])

        # Каноническое обновление SARSA: Q += alpha * (r + gamma*Q' - Q)
        self._get_q_values(s_key)[action_idx] = current_q + self.alpha * (
            reward_val + self.gamma * next_q - current_q
        )

    def seed(self, seed: int) -> None:
        self._rng.seed(seed)

    def save_policy(self, filepath: str | Path) -> None:
        serializable_table = {str(k): v.tolist() for k, v in self.q_table.items()}
        data = {
            "hyperparameters": {
                "alpha": self.alpha,
                "gamma": self.gamma,
                "action_size": self.action_size,
            },
            "q_table": serializable_table,
        }
        with open(filepath, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
        print(f"Policy saved to {filepath}")

    def load_policy(self, filepath: str | Path) -> None:
        with open(filepath, "r") as f:
            data = yaml.safe_load(f)

        hp = data["hyperparameters"]
        self.alpha = hp["alpha"]
        self.gamma = hp["gamma"]
        self.action_size = hp["action_size"]

        self.q_table = {
            ast.literal_eval(k): np.array(v)
            for k, v in data["q_table"].items()
        }
        print(f"Policy loaded from {filepath}. States in table: {len(self.q_table)}")