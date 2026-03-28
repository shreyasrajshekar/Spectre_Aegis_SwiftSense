
#AEGIS-Polaris : AI-Powered Spectrum Intelligence Platform

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

An intelligent, real-time spectrum monitoring and decision-making system combining **Software-Defined Radio (SDR)**, **Deep Learning**, and **Reinforcement Learning** for autonomous ISM band surveillance, signal classification, and adaptive spectrum management.

## 🎯 Features

- **Real-Time Spectrum Monitoring**: Multi-band ISM frequency surveillance (433 MHz – 5.8 GHz)
- **AI-Powered Signal Classification**: CNN-based deep learning for signal type recognition (Idle, 6G-URLLC, 6G-eMBB, Legacy IoT)
- **Adaptive Decision Making**: D3QN Reinforcement Learning agent for intelligent channel management
- **Digital Twin Simulation**: Synthetic signal generation and physics-based traffic modeling for testing and validation
- **Real-Time Dashboard**: Web-based telemetry interface with WebSocket streaming
- **Hardware Integration**: Full ADALM-Pluto (PlutoSDR) support
- **Database Logging**: MySQL telemetry persistence
- **Low-Latency Processing**: Optimized signal processing pipeline

## 📋 Project Structure

```
spec/
├── app.py                          # Main FastAPI application & WebSocket server
├── requirements_db.txt             # Python dependencies
│
├── ai/
│   ├── sensing_cnn.py             # Spectrum CNN classifier (4-class signal detection)
│   ├── decision_d3qn.py           # D3QN RL agent for channel selection
│   └── __pycache__/
│
├── core/
│   ├── sdr_handler.py             # PlutoSDR interface & configuration
│   ├── dsp.py                     # Digital Signal Processing (spectrogram generation)
│   ├── digital_twin.py            # Physics-based simulator for 6G/IoT signals
│   ├── db_logger.py               # MySQL telemetry logging
│   └── __pycache__/
│
├── static/
│   ├── index.html                 # Web dashboard UI
│   ├── app.js                     # Frontend WebSocket client & controls
│   └── style.css                  # Dashboard styling
│
├── tests/
│   ├── test_latency.py            # Performance benchmarks
│   ├── test_radar.py              # Unit tests for radar detection
│   └── __pycache__/
│
└── .env                           # Environment variables (MySQL credentials)
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **PlutoSDR Hardware** (optional, Digital Twin available for simulation)
- **MySQL Server** (for telemetry logging)
- **pip**

### Installation

1. **Clone or extract the project:**
   ```bash
   cd spec
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements_db.txt
   ```
   
   Additional dependencies you may need:
   ```bash
   pip install fastapi uvicorn torch torchvision torchaudio numpy scipy scikit-learn
   ```

4. **Configure environment variables:**
   Create a `.env` file in the root directory:
   ```env
   MYSQL_HOST=localhost
   MYSQL_USER=root
   MYSQL_PASSWORD=your_password
   MYSQL_DATABASE=aegis_telemetry
   SDR_DEVICE_IP=192.168.2.16
   ```

### Running the Application

**Start the server:**
```bash
python app.py
```

The application will:
- Initialize the PlutoSDR (or Digital Twin if in simulation mode)
- Launch FastAPI server on `http://localhost:8000`
- Start real-time spectrum monitoring
- Begin streaming telemetry to connected clients

**Access the Dashboard:**
Open your browser and navigate to:
```
http://localhost:8000
```

## 🎮 Usage

### Web Dashboard Controls

| Control | Function |
|---------|----------|
| **Digital Twin Toggle** | Switch between PlutoSDR hardware and physics-based simulator |
| **Channel Selection** | Manually select ISM band frequency (433M – 5.8G) |
| **Confidence Threshold** | Adjust AI classification confidence threshold (0.0 – 1.0) |
| **Power Threshold** | Set signal detection power floor (dBm) |
| **Optimizer Mode** | Toggle throughput optimization vs. energy efficiency |

### Telemetry Metrics

The system broadcasts real-time metrics:

```json
{
  "latency_ms": 24.5,
  "class_name": "6G-eMBB",
  "confidence": 0.94,
  "active_beam": 2,
  "reward": 15.3,
  "signal_type": "Legacy IoT",
  "q_values": [0.2, 0.5, 0.8, 0.3],
  "waterfall_slice": [...],
  "sdr_linked": true,
  "pls_score": 98,
  "prediction_horizon": 2.5
}
```

## 🔧 Core Components

### 1. **SDR Handler** (`core/sdr_handler.py`)
- Manages ADALM-Pluto RF tuning and I/Q data acquisition
- Configurable buffer sizes and sampling rates
- Bandwidth management across ISM bands

### 2. **Signal Processing** (`core/dsp.py`)
- Real-time spectrogram generation
- FFT-based frequency analysis
- Waterfall display data preparation

### 3. **CNN Classifier** (`ai/sensing_cnn.py`)
- 4-class signal classification network
- Input: 64×64 spectrogram patches
- Output: [Idle, 6G-URLLC, 6G-eMBB, Legacy IoT]
- Batch normalization and dropout regularization

### 4. **RL Agent** (`ai/decision_d3qn.py`)
- Dueling Double Deep Q-Network implementation
- Channel hopping policy optimization
- Reward function: signal confidence + power efficiency

### 5. **Digital Twin** (`core/digital_twin.py`)
- Simulates realistic 6G/IoT spectrum traffic
- Physics-based ISAC (Integrated Sensing & Communication) modeling
- Configurable interference patterns

### 6. **Database Logger** (`core/db_logger.py`)
- MySQL telemetry persistence
- Async batch logging for high-throughput scenarios

## 📊 System Specifications

| Parameter | Value |
|-----------|-------|
| **Frequency Range** | 70 MHz – 6 GHz (PlutoSDR hardware limit) |
| **ISM Bands Monitored** | 433M, 868M, 915M, 1.2G, 2.4G, 3.5G, 4.9G, 5.2G, 5.5G, 5.8G (MHz/GHz) |
| **Sampling Rate** | ~30 MSPS (configurable) |
| **FFT Size** | 1024 – 16384 points |
| **Update Latency** | <50ms (target) |
| **Signal Classes** | 4 (Idle, URLLC, eMBB, Legacy) |
| **RL State Space** | 10 channels × signal history |
| **RL Action Space** | 10 discrete channel selections |

## 🧪 Testing

Run unit tests:

```bash
pytest tests/test_latency.py -v
pytest tests/test_radar.py -v
```

### Performance Benchmarks

- **Latency**: End-to-end processing latency (<50ms target)
- **Throughput**: Real-time processing at SDR acquisition rate
- **CNN Inference**: Per-spectrogram classification time

## 🛠️ Configuration

### Modifying Monitored Bands

Edit `app.py`:
```python
BANDS = [433e6, 868e6, 915e6, 1.2e9, 2.4e9, 3.5e9, 4.9e9, 5.2e9, 5.5e9, 5.8e9]
```

### Adjusting RL Exploration Rate

Default epsilon (exploration) value in app.py:
```python
aegis.rl_agent.epsilon = 0.2  # Lower = more exploitation
```

### CNN Confidence Threshold

Web dashboard or programmatically:
```python
aegis.cnn.confidence_threshold = 0.5  # Range: 0.0 – 1.0
```

## 📡 Hardware Setup

### PlutoSDR Configuration

1. Install ADALM-Pluto firmware and drivers
2. Connect via USB or Ethernet (IP: `192.168.2.16`)
3. Update `.env` with device IP if needed
4. Application auto-detects PlutoSDR on startup

### Optional: MySQL Setup

```sql
CREATE DATABASE aegis_telemetry;
USE aegis_telemetry;
CREATE TABLE telemetry (
  id INT AUTO_INCREMENT PRIMARY KEY,
  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
  signal_class VARCHAR(50),
  confidence FLOAT,
  power_dbm FLOAT,
  latency_ms FLOAT,
  channel_idx INT
);
```

## 🚨 Troubleshooting

| Issue | Solution |
|-------|----------|
| **PlutoSDR not detected** | Check USB connection, verify IP in `.env`, run `iio_info` |
| **High latency** | Reduce FFT size, increase SDR buffer size, disable database logging |
| **CNN inference errors** | Verify PyTorch installation, check model file paths |
| **MySQL connection failed** | Confirm MySQL server running, validate credentials in `.env` |
| **Dashboard not updating** | Check WebSocket connection, verify `localhost:8000` is accessible |

## 📚 Algorithm Details

### D3QN Reinforcement Learning

The RL agent maximizes the reward function:
$$R(t) = \alpha \cdot \text{confidence}(t) + \beta \cdot \text{power}(t) - \gamma \cdot \text{hopping\_cost}$$

Where:
- $\alpha$ = signal confidence weight
- $\beta$ = power efficiency weight
- $\gamma$ = channel switching penalty

### CNN Architecture

```
Input: (1, 64, 64) spectrogram patch
  ↓
Conv2d(1→16, k=3) + BatchNorm + ReLU + MaxPool2d(2)
  ↓
Conv2d(16→32, k=3) + BatchNorm + ReLU + MaxPool2d(2)
  ↓
Conv2d(32→64, k=3) + BatchNorm + ReLU + MaxPool2d(2)
  ↓
Flatten → Linear(64*8*8→128) + Dropout(0.5)
  ↓
Linear(128→4) [Softmax for classification]
```

## 🔐 Security Considerations

- **Remote Command Execution**: WebSocket `exit` command terminates app (disable in production)
- **API Security**: No authentication on FastAPI endpoints (add Bearer tokens for deployment)
- **SDR State**: Digital Twin toggle allows unrestricted mode switching
- **Database Credentials**: Store in `.env`, never commit to version control

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit changes with clear messages
4. Submit pull requests with test coverage

## 📧 Support & Contact

For issues, questions, or feature requests:
- Check existing GitHub Issues
- Review documentation above
- Monitor system logs for diagnostics

---

**Built with ❤️ for 6G Spectrum Intelligence**

*Last Updated: March 2026*
