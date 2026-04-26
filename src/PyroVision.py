"""
PyroVision communication library.

Copyright (C) 2026
This file is part of the PyroVision project.

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
import requests
import websockets
import logging

from enum import Enum
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from websockets.client import WebSocketClientProtocol

class Camera(Enum):
    THERMAL = 0
    RGB = 1

@dataclass
class TelemetryData:
    """
    Telemetry data structure containing device status information

    Attributes:
        UptimeS: Device uptime in seconds
        BatteryVoltageMv: Battery voltage in millivolts
        BatteryPercentage: Battery level as percentage (0-100)
        BatteryCharging: Battery charging status
        WifiRssiDbm: WiFi signal strength in dBm
        LeptonFpaC: Lepton FPA temperature in Celsius
        LeptonAuxC: Lepton AUX temperature in Celsius
        DeviceTemperatureC: Internal device temperature in Celsius
    """
    UptimeS: int
    BatteryVoltageMv: int
    BatteryPercentage: int
    BatteryCharging: bool
    WifiRssiDbm: int
    LeptonFpaC: float
    LeptonAuxC: float
    DeviceTemperatureC: float

@dataclass
class MemoryData:
    """
    Memory data structure containing storage and coredump usage information

    Attributes:
        SdcardPresent: SD card present flag
        MemoryFreeMb: Free memory in megabytes
        MemoryTotalMb: Total memory in megabytes
        MemoryUsedMb: Used memory in megabytes
        CoredumpFreeMb: Free coredump storage in megabytes
        CoredumpTotalMb: Total coredump storage in megabytes
        CoredumpUsedMb: Used coredump storage in megabytes
    """
    SdcardPresent: bool
    MemoryFreeMb: Optional[float] = None
    MemoryTotalMb: Optional[float] = None
    MemoryUsedMb: Optional[float] = None
    CoredumpFreeMb: Optional[float] = None
    CoredumpTotalMb: Optional[float] = None
    CoredumpUsedMb: Optional[float] = None

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

    def __init__(self, Host: str, Port: int = 80, ApiKey: Optional[str] = None):
        """
        Initialize the PyroVision client

        Args:
            Host: Device IP address or hostname
            Port: HTTP port (default: 80)
            ApiKey: Optional API key for authentication

        Raises:
            ValueError: If Host is empty or invalid
        """

        if(not Host):
            raise ValueError("Host cannot be empty")

        self._Logger = logging.getLogger("PyroVisionClient")
        self._BaseUrl = f"http://{Host}:{Port}/api/v1"
        self._WsUrl = f"ws://{Host}:{Port}/ws"
        self._ApiKey = ApiKey
        self._WsConnection: Optional[WebSocketClientProtocol] = None

    def _GetHeaders(self) -> Dict[str, str]:
        """
        Get HTTP headers with optional API key

        Returns:
            Dictionary containing HTTP headers with Content-Type and optional X-API-Key
        """

        Headers = {"Content-Type": "application/json"}
        if(self._ApiKey):
            Headers["X-API-Key"] = self._ApiKey

        return Headers

    def SetTime(self, Epoch: Optional[int] = None, Timezone: str = "Europe/Berlin") -> Dict[str, Any]:
        """
        Set device time and timezone

        Synchronizes the device's internal RTC with the provided timestamp.
        If no timestamp is provided, uses the current system time.

        Args:
            Epoch: Unix timestamp in seconds (if None, uses current time)
            Timezone: IANA timezone string (default: Europe/Berlin)

        Returns:
            Response dictionary with confirmation

        Raises:
            requests.HTTPError: If the request fails

        Example:
            >>> client.SetTime(Timezone="America/New_York")
            {'status': 'ok', 'message': 'Time synchronized'}
        """

        if(Epoch is None):
            Epoch = int(time.time())

        Payload = {
            "epoch": Epoch,
            "timezone": Timezone
        }

        Response = requests.post(
            f"{self._BaseUrl}/time",
            json = Payload,
            headers = self._GetHeaders()
        )
        Response.raise_for_status()

        return Response.json()

    def SetPalette(self, Palette: int) -> Dict[str, Any]:
        """
        Set the color palette for thermal images

        Updates the device's current color palette used for image generation.
        Palette options depend on the camera capabilities.

        Args:
            Palette: Palette index (e.g., 0 for grayscale, 1 for ironbow)

        Returns:
            Response dictionary with confirmation

        Raises:
            requests.HTTPError: If the request fails

        Example:
            >>> client.SetPalette(1)
            {'status': 'ok', 'message': 'Palette updated'}
        """

        Payload = {
            "palette": Palette
        }

        Response = requests.post(
            f"{self._BaseUrl}/settings",
            json = Payload,
            headers = self._GetHeaders()
        )
        Response.raise_for_status()

        return Response.json()

    def SetFormat(self, Format: int) -> Dict[str, Any]:
        """
        Set the image format for thermal images

        Updates the device's current image format used for streaming and capture.
        Format options depend on the camera capabilities.

        Args:
            Format: Format index (e.g., 0 for JPEG, 1 for PNG)

        Returns:
            Response dictionary with confirmation

        Raises:
            requests.HTTPError: If the request fails

        Example:
            >>> client.SetFormat(1)
            {'status': 'ok', 'message': 'Format updated'}
        """

        Payload = {
            "format": Format
        }

        Response = requests.post(
            f"{self._BaseUrl}/settings",
            json = Payload,
            headers = self._GetHeaders()
        )
        Response.raise_for_status()

        return Response.json()

    def GetImage(self) -> bytes:
        """
        Retrieve a single thermal image from the camera

        Captures and returns a thermal image with the current color palette.
        The image is returned as raw binary data.

        Args:

        Returns:
            Image binary data as bytes

        Raises:
            requests.HTTPError: If the request fails

        Example:
            >>> ImageData = client.GetImage(Format=0, Palette=0)
            >>> with open("thermal.jpg", "wb") as f:
            ...     f.write(ImageData)
        """

        Response = requests.get(
            f"{self._BaseUrl}/image",
            headers = self._GetHeaders()
        )
        Response.raise_for_status()

        ImageData = Response.content

        return ImageData

    def GetInfo(self) -> Dict[str, Any]:
        """
        Retrieve device information

        Queries the device for static information such as firmware version, model, and serial number.

        Returns:
            Dictionary containing device information

        Raises:
            requests.HTTPError: If the request fails

        Example:
            >>> Info = client.GetInfo()
            >>> print(f"Model: {Info['model']}, Firmware: {Info['firmware_version']}")
        """

        Response = requests.get(
            f"{self._BaseUrl}/info",
            headers = self._GetHeaders()
        )
        Response.raise_for_status()

        return Response.json()

    def GetTelemetry(self) -> TelemetryData:
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
            >>> Telemetry = client.GetTelemetry()
            >>> print(f"FPA: {Telemetry.LeptonFpaC} C")
            >>> print(f"Uptime: {Telemetry.UptimeS}s")
        """

        Response = requests.get(
            f"{self._BaseUrl}/telemetry",
            headers = self._GetHeaders()
        )
        Response.raise_for_status()
        Data = Response.json()

        return TelemetryData(
            UptimeS = Data["device_uptime_s"],
            BatteryVoltageMv = Data["battery_voltage_mv"],
            BatteryPercentage = Data["battery_percentage"],
            BatteryCharging = Data["battery_charging"],
            WifiRssiDbm = Data["wifi_rssi_dbm"],
            LeptonFpaC = Data["lepton_fpa_c"],
            LeptonAuxC = Data["lepton_aux_c"],
            DeviceTemperatureC = Data["device_temp_c"]
        )

    def GetMemoryInfo(self) -> MemoryData:
        """
        Retrieve current memory and storage information from the device

        Queries the device for current memory usage, SD card status, and coredump storage.

        Returns:
            MemoryData object containing memory and storage information
        Raises:
            requests.HTTPError: If the request fails
            KeyError: If response data is malformed
        Example:
            >>> MemInfo = client.GetMemoryInfo()
            >>> print(f"SD Card Present: {MemInfo.SdcardPresent}")
            >>> print(f"Free Memory: {MemInfo.MemoryFreeMb} MB")
        """

        Response = requests.get(
            f"{self._BaseUrl}/memory",
            headers = self._GetHeaders()
        )
        Response.raise_for_status()
        Data = Response.json()

        return MemoryData(
            SdcardPresent = Data["sdcard_present"],
            MemoryFreeMb = Data.get("memory_free_mb"),
            MemoryTotalMb = Data.get("memory_total_mb"),
            MemoryUsedMb = Data.get("memory_used_mb"),
            CoredumpFreeMb = Data.get("coredump_free_mb"),
            CoredumpTotalMb = Data.get("coredump_total_mb"),
            CoredumpUsedMb = Data.get("coredump_used_mb")
        )

    @property
    def Connection(self) -> WebSocketClientProtocol:
        """
        Get the current WebSocket connection

        Returns:
            Active WebSocketClientProtocol instance or None if not connected
        """

        return self._WsConnection

    async def Connect(self):
        """
        Establish WebSocket connection to the device

        Opens a persistent WebSocket connection for streaming data.
        Automatically called by streaming methods.

        Raises:
            websockets.exceptions.WebSocketException: If connection fails
        """

        if (self._WsConnection is None) or self._WsConnection.closed:
            self._WsConnection = await websockets.connect(self._WsUrl, ping_interval = 60, ping_timeout = 60)
            self._Logger.info("WebSocket connection established")

    async def Disconnect(self):
        """
        Close WebSocket connection

        Gracefully closes the WebSocket connection if open.
        """

        if(self._WsConnection and (not self._WsConnection.closed)):
            await self._WsConnection.close()
            self._Logger.info("WebSocket connection closed")

    async def SendCommand(self, Cmd: str, Data: Dict[str, Any] = None):
        """
        Send command via WebSocket

        Sends a JSON command message through the WebSocket connection.
        Automatically establishes connection if not already connected.

        Args:
            Cmd: Command name
            Data: Command data dictionary (optional)

        Raises:
            websockets.exceptions.WebSocketException: If send fails
        """

        if(not(self._WsConnection)):
            await self.Connect()

        Message = {
            "cmd": Cmd,
            "data": Data or {}
        }

        await self._WsConnection.send(json.dumps(Message))

    async def StartImageStream(self,
                               Format: str = "jpeg",
                               Fps: int = 8,
                               Callback: Optional[Callable[[bytes], None]] = None):
        """
        Start live thermal image streaming via WebSocket

        Initiates continuous image streaming at the specified frame rate.
        Images are delivered as binary frames through the WebSocket connection.

        Args:
            Format: Image format - "jpeg" or "png"
            Fps: Target frames per second (1-9 Hz for Lepton 3.5)
            Callback: Optional callback function called for each frame with image bytes

        Raises:
            websockets.exceptions.WebSocketException: If streaming fails

        Example:
            >>> def OnFrame(Data: bytes):
            ...     with open("frame.jpg", "wb") as f:
            ...         f.write(Data)
            >>> await client.StartImageStream(Fps = 8, Callback = OnFrame)
        """

        await self.Connect()

        self._Logger.info(f"Starting image stream: format = {Format}, fps = {Fps}")

        await self.SendCommand("start", {
            "fps": Fps
        })

        try:
            async for Message in self._WsConnection:
                if isinstance(Message, bytes):
                    # Binary frame (image data)
                    if(Callback):
                        Callback(Message)
                else:
                    # JSON message
                    Data = json.loads(Message)

        except websockets.exceptions.ConnectionClosed:
            pass

    async def StopImageStream(self):
        """
        Stop live image streaming

        Sends command to stop the current image stream.

        Raises:
            websockets.exceptions.WebSocketException: If send fails
        """

        await self.SendCommand("stop", {})

    async def SubscribeTelemetry(self,
                                  IntervalMs: int = 1000,
                                  Callback: Optional[Callable[[Dict[str, Any]], None]] = None):
        """
        Subscribe to live telemetry stream via WebSocket

        Receives periodic telemetry updates at the specified interval.

        Args:
            IntervalMs: Update interval in milliseconds (minimum: 100ms)
            Callback: Optional callback function called with telemetry data dictionary

        Raises:
            websockets.exceptions.WebSocketException: If streaming fails

        Example:
            >>> def OnTelemetry(Data: dict):
            ...     print(f"FPA: {Data['lepton_fpa_c']} C")
            >>> await client.SubscribeTelemetry(IntervalMs = 1000, Callback = OnTelemetry)
        """

        await self.Connect()

        # Subscribe
        await self.SendCommand("subscribe", {
            "interval": IntervalMs
        })

        try:
            async for Message in self._WsConnection:
                if isinstance(Message, str):
                    Data = json.loads(Message)

                    if Data.get("cmd") == "telemetry":
                        if Callback:
                            Callback(Data.get("data", {}))

        except websockets.exceptions.ConnectionClosed:
            pass