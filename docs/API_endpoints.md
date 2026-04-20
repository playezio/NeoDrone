# DJI SDK API Endpoints Documentation

## Overview

This document describes the DJI SDK endpoints used for extracting telemetry and camera data from drone flights during the NEODrone dataset collection process.

## Base URL

```
http://localhost:8080/api/v1
```

## Authentication

All API requests require authentication using an API key:

```
Authorization: Bearer YOUR_API_KEY
```

## Telemetry Endpoints

### GET `/telemetry/position`

Retrieves current drone position data.

**Response:**
```json
{
  "latitude": 39.9042,
  "longitude": 116.4074,
  "altitude": 120.5,
  "relative_altitude": 45.2,
  "timestamp": "2024-01-01T12:00:00Z"
}
```

**Fields:**
- `latitude` (float): GPS latitude in degrees (-90 to 90)
- `longitude` (float): GPS longitude in degrees (-180 to 180)
- `altitude` (float): Absolute altitude in meters above sea level
- `relative_altitude` (float): Relative altitude in meters above takeoff point
- `timestamp` (string): ISO 8601 formatted timestamp

### GET `/telemetry/attitude`

Retrieves drone attitude/orientation data.

**Response:**
```json
{
  "heading": 180.5,
  "pitch": -5.2,
  "roll": 2.1,
  "timestamp": "2024-01-01T12:00:00Z"
}
```

**Fields:**
- `heading` (float): Compass heading in degrees (0-360)
- `pitch` (float): Pitch angle in degrees (-90 to 90)
- `roll` (float): Roll angle in degrees (-180 to 180)
- `timestamp` (string): ISO 8601 formatted timestamp

### GET `/telemetry/velocity`

Retrieves drone velocity data.

**Response:**
```json
{
  "velocity_x": 5.2,
  "velocity_y": -3.1,
  "velocity_z": 0.5,
  "ground_speed": 6.0,
  "timestamp": "2024-01-01T12:00:00Z"
}
```

**Fields:**
- `velocity_x` (float): Velocity in X direction (m/s)
- `velocity_y` (float): Velocity in Y direction (m/s)
- `velocity_z` (float): Velocity in Z direction (m/s)
- `ground_speed` (float): Ground speed (m/s)
- `timestamp` (string): ISO 8601 formatted timestamp

### GET `/telemetry/battery`

Retrieves battery status information.

**Response:**
```json
{
  "voltage": 22.5,
  "current": 15.2,
  "percentage": 75,
  "temperature": 35.5,
  "timestamp": "2024-01-01T12:00:00Z"
}
```

**Fields:**
- `voltage` (float): Battery voltage (V)
- `current` (float): Battery current (A)
- `percentage` (int): Battery percentage (0-100)
- `temperature` (float): Battery temperature (°C)
- `timestamp` (string): ISO 8601 formatted timestamp

## Camera Endpoints

### GET `/camera/settings`

Retrieves current camera settings.

**Response:**
```json
{
  "iso": 400,
  "shutter_speed": "1/1000",
  "aperture": "f/2.8",
  "exposure_mode": "auto",
  "white_balance": "auto",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

**Fields:**
- `iso` (int): ISO value
- `shutter_speed` (string): Shutter speed
- `aperture` (string): Aperture value
- `exposure_mode` (string): Exposure mode (auto/manual)
- `white_balance` (string): White balance setting
- `timestamp` (string): ISO 8601 formatted timestamp

### GET `/camera/gimbal`

Retrieves gimbal orientation data.

**Response:**
```json
{
  "pitch": -15.5,
  "roll": 0.2,
  "yaw": 0.0,
  "mode": "follow",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

**Fields:**
- `pitch` (float): Gimbal pitch angle in degrees (-90 to 90)
- `roll` (float): Gimbal roll angle in degrees (-30 to 30)
- `yaw` (float): Gimbal yaw angle in degrees (-180 to 180)
- `mode` (string): Gimbal mode (follow/fpv/locked)
- `timestamp` (string): ISO 8601 formatted timestamp

### GET `/camera/zoom`

Retrieves zoom level information.

**Response:**
```json
{
  "optical_zoom": 2.0,
  "digital_zoom": 1.0,
  "total_zoom": 2.0,
  "focal_length": 24.0,
  "timestamp": "2024-01-01T12:00:00Z"
}
```

**Fields:**
- `optical_zoom` (float): Optical zoom factor
- `digital_zoom` (float): Digital zoom factor
- `total_zoom` (float): Total zoom factor
- `focal_length` (float): Current focal length (mm)
- `timestamp` (string): ISO 8601 formatted timestamp

### GET `/camera/focus`

Retrieves focus information.

**Response:**
```json
{
  "mode": "auto",
  "distance": 50.5,
  "focus_point": { "x": 0.5, "y": 0.5 },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

**Fields:**
- `mode` (string): Focus mode (auto/manual)
- `distance` (float): Focus distance (meters)
- `focus_point` (object): Focus point coordinates (normalized 0-1)
- `timestamp` (string): ISO 8601 formatted timestamp

## Data Streaming

### WebSocket `/stream/telemetry`

Real-time telemetry data stream.

**Connection:**
```
ws://localhost:8080/api/v1/stream/telemetry
```

**Message Format:**
```json
{
  "type": "telemetry",
  "data": {
    "position": { ... },
    "attitude": { ... },
    "velocity": { ... }
  },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### WebSocket `/stream/camera`

Real-time camera data stream.

**Connection:**
```
ws://localhost:8080/api/v1/stream/camera
```

**Message Format:**
```json
{
  "type": "camera",
  "data": {
    "settings": { ... },
    "gimbal": { ... },
    "zoom": { ... }
  },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## Error Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request |
| 401 | Unauthorized |
| 404 | Not Found |
| 500 | Internal Server Error |

## Rate Limiting

- Standard endpoints: 100 requests/minute
- Streaming endpoints: No rate limit
- Burst allowance: 200 requests/minute

## Data Format Notes

### Timestamp Format

All timestamps follow ISO 8601 format:
```
YYYY-MM-DDTHH:MM:SSZ
```

### Coordinate System

- **Latitude**: -90 (South) to +90 (North)
- **Longitude**: -180 (West) to +180 (East)
- **Altitude**: Meters above sea level (WGS84)

### Angle Units

All angles are in degrees:
- **Heading**: 0-360° (0° = North, 90° = East)
- **Pitch**: -90° (down) to +90° (up)
- **Roll**: -180° to +180°

## Best Practices

1. **Caching**: Cache telemetry data locally to reduce API calls
2. **Error Handling**: Implement exponential backoff for failed requests
3. **Data Validation**: Validate all received data against expected ranges
4. **Timestamp Synchronization**: Use NTP to synchronize clocks
5. **Connection Management**: Properly close WebSocket connections when done

## Example Usage

### Python Example

```python
import requests
import json

# Set up authentication
headers = {
    'Authorization': 'Bearer YOUR_API_KEY',
    'Content-Type': 'application/json'
}

# Get position data
response = requests.get(
    'http://localhost:8080/api/v1/telemetry/position',
    headers=headers
)

if response.status_code == 200:
    position_data = response.json()
    print(f"Position: {position_data['latitude']}, {position_data['longitude']}")
else:
    print(f"Error: {response.status_code}")
```

### WebSocket Example

```python
import asyncio
import websockets
import json

async def stream_telemetry():
    uri = "ws://localhost:8080/api/v1/stream/telemetry"
    
    async with websockets.connect(uri) as websocket:
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            
            if data['type'] == 'telemetry':
                print(f"Received: {data['timestamp']}")

# Run the stream
asyncio.run(stream_telemetry())
```

## Support

For issues or questions regarding the DJI SDK API, please contact:
- Email: support@neodrone.org
- GitHub Issues: https://github.com/your-org/NEODrone/issues