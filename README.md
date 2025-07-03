
# Container and Microservices Health Orchestrator

## 🔍 Problem Statement

Create a system that monitors the health of containerized microservices and automatically orchestrates healing actions such as:
- Service restarts
- Auto-scaling
- Traffic rerouting

The goal is to maintain system stability and high availability through intelligent, automated interventions.

---

## ⚙️ Features

- **Health Monitoring**: Periodically collects service-level metrics (CPU, memory, latency, etc.)
- **Health Assessment**: Classifies service status as Healthy, Degraded, Unhealthy, or Failed
- **Failure Prediction**: Uses a machine learning model (RandomForestClassifier) to predict potential service failures
- **Healing Decision Engine**: Determines optimal healing actions (restart, scale up/down, reroute)
- **Orchestration Module**: Simulates execution of healing actions using mock Kubernetes-like logic
- **Simulation Mode**: Includes synthetic metric generation for testing the orchestration flow

---

## 🧠 Skills Demonstrated

- **AI/ML**: Built health prediction models, failure classification, optimal recovery decisioning
- **Critical Thinking**: Balanced automation with manual intervention; accounted for cascade failures
- **Problem Solving**: Addressed partial service failures, dependencies, and network anomalies
- **Modular Design**: Clear separation of metric collection, assessment, prediction, decision, and execution
- **Clean Architecture**: 
  ```
  Service Metrics ➜ Health Assessment ➜ Failure Prediction ➜ Healing Actions
  ```

---

## 🗂 Project Structure

```plaintext
MICROSERVICES-ORCHESTRATOR/
├── venv/                       # Python virtual environment
├── healthorchestrator.py      # Main orchestration logic
├── README.md                  # Project documentation (this file)
```

---

## 🚀 How to Run

1. **Install dependencies** (inside virtual environment)

2. **Run the orchestrator**:

    ```bash
    python healthorchestrator.py
    ```

3. **Stop the system**:
    - Press `Ctrl+C` or wait until timeout (default: 5 minutes)

---

## 🛠 Configuration

The `config` dictionary at the bottom of `healthorchestrator.py` includes:
- `service_endpoints`: Simulated service names and endpoints
- `service_dependencies`: Maps of inter-service relationships
- `k8s_api_endpoint`: (Mock) Kubernetes API endpoint

---

## 📊 Simulated Metrics

The system generates randomized service metrics, simulating real-world behavior like:
- CPU/memory spikes
- Latency surges
- Error rate increases
- Random partial or cascading failures

---

## 📦 Deliverable

A **self-healing microservices health orchestrator** that continuously monitors, predicts, and responds to service degradation or failure with intelligent automation.

---

## 🧑‍💻 Author

**Bandla Venkata Divya**  
_Health Orchestration Developer_

---

## 📜 License

This project is open-source and available for use under the [MIT License](LICENSE).
