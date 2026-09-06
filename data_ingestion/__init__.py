"""This package exposes core telemetry streaming modules, enabling real-time 
data collection (Producers) and persistent storage (Consumers) within the 
CARLA autonomous driving ecosystem.
"""

from .carla_streaming import CarlaDataStreaming
from .telemetry_consumer import TelemetryConsumer

__all__ = ["CarlaDataStreaming", "TelemetryConsumer"]