from pydantic import BaseModel, Field
from typing import Optional
import yaml
import os

class HandoffConfig(BaseModel):
    hysteresis_db: float = Field(default=3.0, description="Hysteresis in dB")
    dwell_steps: int = Field(default=5, description="Minimum dwell time in steps")

class LinkConfig(BaseModel):
    carrier_freq_hz: float = Field(default=14e9, description="Default carrier frequency in Hz")
    bandwidth_hz: float = Field(default=36e6, description="Default bandwidth in Hz")
    polarization: str = Field(default="vertical", description="Default polarization (vertical/horizontal)")
    snr_threshold_db: float = Field(default=10.0, description="SNR threshold for packet loss")

class SimulationConfig(BaseModel):
    handoff: HandoffConfig = Field(default_factory=HandoffConfig)
    link: LinkConfig = Field(default_factory=LinkConfig)
    dt_s: int = Field(default=60, description="Time step in seconds")
    n_steps: int = Field(default=60, description="Number of simulation steps")

class RainConfig(BaseModel):
    tau_c: float = Field(default=300.0, description="Rain coherence time in seconds")

class AppConfig(BaseModel):
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    rain: RainConfig = Field(default_factory=RainConfig)

def load_config(config_path: Optional[str] = None) -> AppConfig:
    if config_path is None:
        config_path = os.environ.get("SATLINKSIM_CONFIG", "config.yaml")
    
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
            if data:
                return AppConfig.model_validate(data)
    
    return AppConfig()

# Global config instance
config = load_config()
