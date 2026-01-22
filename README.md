# TTK Care – AI Diagnostics Service

An IoT predictive maintenance system for smart home appliances using Machine Learning.

## 🎯 Overview

This project demonstrates AI-powered health monitoring for:
- **Smart Kettle (Smart-1.7)** - Detects mineral scaling buildup
- **Kitchen Chimney (Oscar-600)** - Detects grease accumulation with AI auto-clean

## 🧠 ML Model

Uses **Linear Regression** to calculate degradation slope from sensor telemetry:
- Slope ≈ 0 → Healthy
- Slope > threshold → Degradation detected → Alert/Action

## 🚀 Quick Start

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run the Application
```bash
python app.py
```

### Open Dashboard
Navigate to: http://127.0.0.1:8000

## 📁 Project Structure

```
ai_service/
├── app.py                 # Flask web application
├── data_logger.py         # CSV telemetry logging
├── rule_engine.py         # AI decision rules
├── moving_average.py      # Data smoothing analysis
├── regression.py          # Linear regression analysis
├── visualize.py           # Matplotlib visualizations
├── requirements.txt       # Python dependencies
├── simulator/
│   ├── kettle_sim.py      # Kettle physical simulator
│   └── chimney_sim.py     # Chimney physical simulator
├── templates/
│   └── dashboard.html     # Web UI
├── static/
│   └── style.css          # Styling
└── data/
    └── telemetry_log.csv  # Sensor data storage
```

## 🎮 Demo Features

### Kettle
- **Simulate Boil** - Simulates usage cycles (scaling builds up)
- **Descale** - Resets kettle to healthy state

### Chimney
- **Simulate Usage** - Simulates cooking cycles (grease builds up)
- **Manual Auto-Clean** - User-triggered thermal cleaning
- **Toggle Auto-Clean** - Enable/disable AI automatic cleaning

## 📊 How It Works

1. **Simulators** generate realistic sensor data (boil time, motor current)
2. **Data Logger** stores telemetry to CSV
3. **ML Model** calculates degradation slope using linear regression
4. **Rule Engine** classifies health status and triggers actions
5. **Dashboard** visualizes trends and allows user interaction

## 🔧 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Dashboard UI |
| `/api/device_status/<model>` | GET | Get device health |
| `/simulate/kettle` | POST | Simulate kettle cycles |
| `/simulate/kettle/descale` | POST | Descale kettle |
| `/simulate/chimney` | POST | Simulate chimney cycles |
| `/simulate/chimney/auto_clean` | POST | Trigger auto-clean |
| `/chimney/auto_clean/toggle` | POST | Toggle AI auto-clean |

## 📈 Future Improvements

- [ ] Polynomial regression for non-linear degradation
- [ ] Anomaly detection for sudden failures
- [ ] Predictive "cycles to failure" forecast
- [ ] Multi-feature analysis (temperature, vibration)
- [ ] LSTM deep learning model

## 📄 License

MIT License
