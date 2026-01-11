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

    # Seasonal cycle parameters (disabled by default)
    y_0_seasonal_amp: float = 0.0        # m - ITCZ migration amplitude (0 = no seasonal cycle)
    seasonal_period_days: float = 360.0  # days - seasonal period
    seasonal_phase_days: float = 0.0     # days - phase offset


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


class SB08Profile(ThetaEProfile):
    """θₑ profile using the formula from Schneider and Bordoni (2008)"""

    def __call__(self, state: ModelState) -> np.ndarray:
        # Compute time-varying y_0 if seasonal amplitude > 0
        if self.config.y_0_seasonal_amp > 0:
            # Convert periods from days to seconds
            period_seconds = self.config.seasonal_period_days * 86400
            phase_seconds = self.config.seasonal_phase_days * 86400

            # Compute seasonal phase
            phase = 2 * np.pi * (state.t - phase_seconds) / period_seconds

            # Time-varying y_0
            y_0_t = self.config.y_0 + self.config.y_0_seasonal_amp * np.sin(phase)
        else:
            # No seasonal cycle - use constant y_0
            y_0_t = self.config.y_0

        # Original SB08 formula with time-varying y_0
        term1 = np.sin(np.pi * state.y / (2 * self.config.y_one)) ** 2
        term2 = (
            2
            * np.sin(np.pi * y_0_t / (2 * self.config.y_one))
            * np.sin(np.pi * state.y / (2 * self.config.y_one))
        )

        return self.config.theta_00 - self.config.delta_y * (term1 - term2)
