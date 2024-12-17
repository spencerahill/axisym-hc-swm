from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from .model_state import ModelState


@dataclass
class ThetaEConfig:
    """Configuration for the θₑ profile."""

    theta_00: float = 330.0
    y_0: float = 0.0
    y_one: float = 9439e3
    delta_y: float = 50
    theta_e_type: str = "sin2"


class ThetaEProfile(ABC):
    """Abstract base class for θₑ profiles."""

    def __init__(self, config: ThetaEConfig):
        self.config = config

    @abstractmethod
    def __call__(self, state: ModelState) -> np.ndarray:
        """Calculate θₑ for given model state."""
        pass


class SS09Profile(ThetaEProfile):
    """θₑ profile using (y/y₁)² form from the original paper"""

    def __call__(self, state: ModelState) -> np.ndarray:
        return np.where(
            np.abs(state.y) < self.config.y_one,
            self.config.theta_00
            - self.config.delta_y * (state.y / self.config.y_one) ** 2,
            self.config.theta_00 - self.config.delta_y,
        )


class Sin2Profile(ThetaEProfile):
    """θₑ profile using sin²(πy/2y₁) form"""

    def __call__(self, state: ModelState) -> np.ndarray:
        return np.where(
            np.abs(state.y - self.config.y_0) < self.config.y_one,
            self.config.theta_00
            - self.config.delta_y
            * (
                np.sin(0.5 * np.pi * (state.y - self.config.y_0) / self.config.y_one)
                ** 2
            ),
            self.config.theta_00 - self.config.delta_y,
        )
