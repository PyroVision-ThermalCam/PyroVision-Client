"""
PyroVision Network Client
ESP32 Thermal Camera communication library for Python applications

Copyright (C) 2026
This file is part of PyroVision.

PyroVision is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

PyroVision is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with PyroVision. If not, see <https://www.gnu.org/licenses/>.
"""

import json
import time
import asyncio
import requests
import websockets
import logging

from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from websockets.client import WebSocketClientProtocol

@dataclass
class TelemetryData:
    """
    Telemetry data structure containing device status information
    
    Attributes:
        uptime_s: Device uptime in seconds
        sensor_temp_c: Lepton sensor temperature in degrees Celsius
        core_temp_c: ESP32 core temperature in degrees Celsius
        supply_voltage_v: Supply voltage in volts
        wifi_rssi_dbm: WiFi signal strength in dBm
        sdcard_present: SD card presence status
        sdcard_free_mb: Available SD card space in megabytes
    """
    uptime_s: int
    sensor_temp_c: float
    core_temp_c: float
    supply_voltage_v: float
    wifi_rssi_dbm: int
    sdcard_present: bool
    sdcard_free_mb: int

class PyroVisionClient:
    """
    PyroVision ESP32 Thermal Camera Client

    Provides REST API and WebSocket communication with the thermal camera device.
    Supports image capture, telemetry monitoring, LED control, and firmware updates.

    Example:
        >>> client = PyroVisionClient("192.168.4.1", api_key="secret")
        >>> telemetry = client.get_telemetry()
        >>> print(f"Sensor temp: {telemetry.sensor_temp_c}°C")
    """

    def __init__(self, host: str, port: int = 80, api_key: Optional[str] = None):
        """
        Initialize the PyroVision client

        Args:
            host: Device IP address or hostname
            port: HTTP port (default: 80)
            api_key: Optional API key for authentication

        Raises:
            ValueError: If host is empty or invalid
        """

        if(not host):
            raise ValueError("Host cannot be empty")

        self._logger = logging.getLogger("PyroVisionClient")
        self._base_url = f"http://{host}:{port}/api/v1"
        self._ws_url = f"ws://{host}:{port}/ws"
        self._api_key = api_key
        self._ws_connection: Optional[WebSocketClientProtocol] = None

    def _get_headers(self) -> Dict[str, str]:
        """
        Get HTTP headers with optional API key

        Returns:
            Dictionary containing HTTP headers with Content-Type and optional X-API-Key
        """

        headers = {"Content-Type": "application/json"}
        if(self._api_key):
            headers["X-API-Key"] = self._api_key

        return headers

    def set_time(self, epoch: Optional[int] = None, timezone: str = "Europe/Berlin") -> Dict[str, Any]:
        """
        Set device time and timezone

        Synchronizes the device's internal RTC with the provided timestamp.
        If no timestamp is provided, uses the current system time.

        Args:
            epoch: Unix timestamp in seconds (if None, uses current time)
            timezone: IANA timezone string (default: Europe/Berlin)

        Returns:
            Response dictionary with confirmation

        Raises:
            requests.HTTPError: If the request fails

        Example:
            >>> client.set_time(timezone="America/New_York")
            {'status': 'ok', 'message': 'Time synchronized'}
        """

        if(epoch is None):
            epoch = int(time.time())

        payload = {
            "epoch": epoch,
            "timezone": timezone
        }

        response = requests.post(
            f"{self._base_url}/time",
            json = payload,
            headers = self._get_headers()
        )
        response.raise_for_status()

        return response.json()

    def get_image(self,
                  format: str = "jpeg",
                  palette: str = "iron",
                  scale: str = "linear",
                  save_path: Optional[Path] = None) -> bytes:
        """
        Retrieve a single thermal image from the camera

        Captures and returns a thermal image with the specified format and colorization.
        Optionally saves the image to disk.

        Args:
            format: Image format - "jpeg", "png", or "raw" (14-bit grayscale)
            palette: Color palette - "iron", "gray", or "rainbow"
            scale: Scaling method - "linear" or "histogram" (auto-contrast)
            save_path: Optional path to save the image file

        Returns:
            Image binary data as bytes

        Raises:
            requests.HTTPError: If the request fails
            ValueError: If format, palette, or scale are invalid

        Example:
            >>> image_data = client.get_image(format="jpeg", palette="iron")
            >>> with open("thermal.jpg", "wb") as f:
            ...     f.write(image_data)
        """

        params = {
            "format": format,
            "palette": palette,
            "scale": scale
        }

        response = requests.get(
            f"{self._base_url}/image",
            params = params,
            headers = self._get_headers()
        )
        response.raise_for_status()

        image_data = response.content

        if(save_path):
            save_path.write_bytes(image_data)

        return image_data

    def get_telemetry(self) -> TelemetryData:
        """
        Retrieve current telemetry data from the device

        Queries the device for current status information including temperatures,
        voltage, WiFi signal strength, and SD card status.

        Returns:
            TelemetryData object containing all status information

        Raises:
            requests.HTTPError: If the request fails
            KeyError: If response data is malformed

        Example:
            >>> telemetry = client.get_telemetry()
            >>> print(f"Sensor: {telemetry.sensor_temp_c}°C")
            >>> print(f"Uptime: {telemetry.uptime_s}s")
        """

        response = requests.get(
            f"{self._base_url}/telemetry",
            headers = self._get_headers()
        )
        response.raise_for_status()
        data = response.json()

        return TelemetryData(
            uptime_s = data["uptime_s"],
            sensor_temp_c = data["sensor_temp_c"],
            core_temp_c = data["core_temp_c"],
            supply_voltage_v = data["supply_voltage_v"],
            wifi_rssi_dbm = data["wifi_rssi_dbm"],
            sdcard_present = data["sdcard"]["present"],
            sdcard_free_mb = data["sdcard"]["free_mb"]
        )

    @property
    def connection(self) -> WebSocketClientProtocol:
        """
        Get the current WebSocket connection

        Returns:
            Active WebSocketClientProtocol instance or None if not connected
        """

        return self._ws_connection

    async def connect(self):
        """
        Establish WebSocket connection to the device

        Opens a persistent WebSocket connection for streaming data.
        Automatically called by streaming methods.

        Raises:
            websockets.exceptions.WebSocketException: If connection fails
        """

        if (self._ws_connection is None) or self._ws_connection.closed:
            self._ws_connection = await websockets.connect(self._ws_url, ping_interval = 20, ping_timeout = 60)
            self._logger.info("WebSocket connection established")

    async def disconnect(self):
        """
        Close WebSocket connection

        Gracefully closes the WebSocket connection if open.
        """

        if(self._ws_connection and (not self._ws_connection.closed)):
            await self._ws_connection.close()
            self._logger.info("WebSocket connection closed")

    async def send_command(self, cmd: str, data: Dict[str, Any] = None):
        """
        Send command via WebSocket

        Sends a JSON command message through the WebSocket connection.
        Automatically establishes connection if not already connected.

        Args:
            cmd: Command name
            data: Command data dictionary (optional)

        Raises:
            websockets.exceptions.WebSocketException: If send fails
        """

        if(not(self._ws_connection)):
            await self.connect()

        message = {
            "cmd": cmd,
            "data": data or {}
        }

        await self._ws_connection.send(json.dumps(message))

    async def start_image_stream(self,
                                  format: str = "jpeg",
                                  fps: int = 8,
                                  callback: Optional[Callable[[bytes], None]] = None):
        """
        Start live thermal image streaming via WebSocket

        Initiates continuous image streaming at the specified frame rate.
        Images are delivered as binary frames through the WebSocket connection.

        Args:
            format: Image format - "jpeg" or "png"
            fps: Target frames per second (1-9 Hz for Lepton 3.5)
            callback: Optional callback function called for each frame with image bytes

        Raises:
            websockets.exceptions.WebSocketException: If streaming fails

        Example:
            >>> def on_frame(data: bytes):
            ...     with open("frame.jpg", "wb") as f:
            ...         f.write(data)
            >>> await client.start_image_stream(fps = 8, callback = on_frame)
        """

        await self.connect()

        self._logger.info(f"Starting image stream: format = {format}, fps = {fps}")

        await self.send_command("start", {
            "fps": fps
        })

        try:
            async for message in self._ws_connection:
                if isinstance(message, bytes):
                    # Binary frame (image data)
                    if(callback):
                        callback(message)
                else:
                    # JSON message
                    data = json.loads(message)

        except websockets.exceptions.ConnectionClosed:
            pass

    async def stop_image_stream(self):
        """
        Stop live image streaming

        Sends command to stop the current image stream.

        Raises:
            websockets.exceptions.WebSocketException: If send fails
        """

        await self.send_command("stop", {})

    async def subscribe_telemetry(self,
                                   interval_ms: int = 1000,
                                   callback: Optional[Callable[[Dict[str, Any]], None]] = None):
        """
        Subscribe to live telemetry stream via WebSocket

        Receives periodic telemetry updates at the specified interval.

        Args:
            interval_ms: Update interval in milliseconds (minimum: 100ms)
            callback: Optional callback function called with telemetry data dictionary

        Raises:
            websockets.exceptions.WebSocketException: If streaming fails

        Example:
            >>> def on_telemetry(data: dict):
            ...     print(f"Temp: {data['sensor_temp_c']}°C")
            >>> await client.subscribe_telemetry(interval_ms = 1000, callback = on_telemetry)
        """

        await self.connect()

        # Subscribe
        await self.send_command("subscribe", {
            "interval": interval_ms
        })

        try:
            async for message in self._ws_connection:
                if isinstance(message, str):
                    data = json.loads(message)

                    if data.get("cmd") == "telemetry":
                        if callback:
                            callback(data.get("data", {}))

        except websockets.exceptions.ConnectionClosed:
            pass