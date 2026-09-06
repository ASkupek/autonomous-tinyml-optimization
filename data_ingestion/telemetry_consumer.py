"""Kafka Telemetry Consumer Engine.

This module subscribes to Apache Kafka telemetry topics, filters and formats
the incoming vehicle state streams, and buffers them for efficient batch 
writing into local CSV datasets.
"""

import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd
from confluent_kafka import Consumer, KafkaError

from config import GLOBAL_CONFIG, GlobalConfig

# Documentation:
# https://docs.confluent.io/platform/current/clients/confluent-kafka-python/html/index.html#pythonclient-producer
# https://github.com/confluentinc/librdkafka/blob/master/CONFIGURATION.md

# TODO(v0.2.0): flash_buffer need to delete existing csv if there

class TelemetryConsumer:
    """Consumes telemetry data streams from Apache Kafka and persists them to disk.

    This class runs an event loop that polls driving features from a Kafka topic,
    filters relevant columns based on project configuration, and appends them
    to a localized CSV dataset using a memory buffer to optimize I/O performance.
    """

    def __init__(self, config: GlobalConfig = GLOBAL_CONFIG) -> None:
        """Initializes the telemetry consumer with buffering and storage profiles.

        Args:
            config (GlobalConfig): Object containing network, topic, and storage configurations.
                Defaults to GLOBAL_CONFIG.
        """

        self.config: GlobalConfig = config
        self.csv_path: str = self.config.pipeline.csv_path
        metadata = ["timestamp", "vehicle_id"]
        self.received_data: List[str] = metadata + self.config.pipeline.input_features + self.config.pipeline.target_outputs
        self.kafka_consumer: Optional[Consumer] = None
        self.kafka_buffer: List[Dict[str, Any]] = []
        self.kafka_buffer_size: int = self.config.kafka.kafka_buffer_size
        self.total_saved: int = 0
        self.is_running: bool = False

    
    def kafka_connection(self) -> None:
        """Establishes a connection to the Kafka broker and subscribes to telemetry topics.

        Raises:
            Exception: If subscription or configuration mapping fails.
        """
        print(f"[Kafka Consumer] Connecting to broker at {self.config.kafka.kafka_server}...")
        try:
            confluent_config = {
                'bootstrap.servers': self.config.kafka.kafka_server,
                'group.id': self.config.kafka.kafka_group_id,
                'auto.offset.reset': self.config.kafka.kafka_offset_reset,
                'enable.auto.commit': self.config.kafka.kafka_auto_commit,
                'auto.commit.interval.ms': self.config.kafka.kafka_commit_interval_ms,
                'session.timeout.ms': self.config.kafka.kafka_session_timeout_ms,
            }

            self.kafka_consumer = Consumer(confluent_config)
            self.kafka_consumer.subscribe([self.config.kafka.kafka_topic])
            print(f"[Kafka Consumer] Successfully subscribed to topic '{self.config.kafka.kafka_topic}'.")

        except Exception as error:
            print(f"[Kafka Consumer Error] Connection/Subscription failed: {error}")
            raise error
    
    def kafka_start_listening(self) -> None:
        """ Enters an infinite polling loop to extract, decode, and buffer incoming telemetry.
            Continuously polls the Kafka topic for new messages. When the internal memory
            buffer reaches `kafka_buffer_size`, it automatically triggers `flush_buffer()`.
            Handles KeyboardInterrupt gracefully to clean up connections.
        """

        if not self.kafka_consumer:
            self.kafka_connection()
        
        self.is_running = True

        try:
            while self.is_running:

                msg = self.kafka_consumer.poll(timeout=1.0)

                if msg is None:
                    continue
                
                if msg.error():
                    if msg.error().code() == KafkaError._TIMED_OUT:
                        continue
                    else:
                        print(f"[Kafka Consumer Error] Stream error: {msg.error()}")
                        continue

                try:
                    raw_payload = msg.value()
                    if not raw_payload:
                        continue
                    data = json.loads(raw_payload.decode('utf-8'))
                except Exception as parse_error:
                    print(f"[Kafka Consumer Error] Payload parsing failed: {parse_error}")
                    continue

                filtered_row = {col: data[col] for col in self.received_data if col in data}

                if len(filtered_row) == len(self.received_data):
                    self.kafka_buffer.append(filtered_row)
                    
                if len(self.kafka_buffer) >= self.kafka_buffer_size:
                    self.flush_buffer()
        except KeyboardInterrupt:
            print("\n[Kafka Consumer] User interruption detected (Ctrl+C). Stopping loop...")
        except Exception as error:
            print(f"[Kafka Consumer Error] Unexpected error during event loop: {error}")
        finally:
            self.stop_listening()
    
    def flush_buffer(self) -> None:
        """Flushes buffered records directly onto storage media via Pandas engine."""

        if not self.kafka_buffer:
            return

        buffer_len = len(self.kafka_buffer)
        try:
            data_frame = pd.DataFrame(self.kafka_buffer)
            file_exists = os.path.isfile(self.csv_path)

            data_frame.to_csv(self.csv_path, mode='a', index=False, header=not file_exists)

            self.total_saved += buffer_len
            self.kafka_buffer.clear()
            print(f"[Disk I/O] Flushed {buffer_len} records to '{self.csv_path}' (Total saved: {self.total_saved}).")

        except (PermissionError, FileNotFoundError, OSError) as disk_error:
            print(f"[Disk I/O Error] Failed writing to file '{self.csv_path}': {disk_error}")
        except Exception as error:
            print(f"[Disk I/O Error] Unexpected failure during buffer flush: {error}")
    
    def stop_listening(self) -> None:
        """Halts the polling process, safely empties residual buffers, and releases sockets."""

        print("[Kafka Consumer] Stopping consumer and cleaning up resources...")
        self.is_running = False
        if self.kafka_buffer:
            print(f"[Kafka Consumer] Flushing remaining {len(self.kafka_buffer)} records before shutdown...")
            self.flush_buffer()
        if self.kafka_consumer:
            self.kafka_consumer.close()
            print("[Kafka Consumer] Consumer connection closed cleanly.")
