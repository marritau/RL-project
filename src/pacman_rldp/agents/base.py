from abc import ABC, abstractmethod
import numpy as np
from typing import Any

class BaseAgent(ABC):
    @abstractmethod
    def select_action(self, observation: dict[str, np.ndarray], info: dict[str, Any]) -> int:
        pass

    def update(self, state, action, reward, next_state, next_action, terminated, next_info):
        pass