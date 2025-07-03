import time
import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum, auto
import logging
from threading import Thread
from queue import Queue
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('HealthOrchestrator')

class ServiceStatus(Enum):
    HEALTHY = auto()
    DEGRADED = auto()
    UNHEALTHY = auto()
    FAILED = auto()

class HealingAction(Enum):
    RESTART = auto()
    SCALE_UP = auto()
    SCALE_DOWN = auto()
    REROUTE_TRAFFIC = auto()
    NOTHING = auto()

@dataclass
class ServiceMetrics:
    service_name: str
    cpu_usage: float
    memory_usage: float
    latency: float
    error_rate: float
    request_rate: float
    instance_count: int

@dataclass
class ServiceHealth:
    service_name: str
    status: ServiceStatus
    confidence: float
    predicted_failure: bool
    timestamp: float

@dataclass
class HealingDecision:
    action: HealingAction
    service_name: str
    confidence: float
    details: str

class MetricsCollector:
    def __init__(self, service_endpoints: Dict[str, str]):
        self.service_endpoints = service_endpoints
        self.metrics_queue = Queue()
        self.running = False
        
    def start_collection(self, interval: int = 5):
        """Start collecting metrics from all services at regular intervals"""
        self.running = True
        def collect():
            while self.running:
                for service, endpoint in self.service_endpoints.items():
                    try:
                        metrics = self._simulate_metrics_fetch(service)
                        self.metrics_queue.put(metrics)
                    except Exception as e:
                        logger.error(f"Error collecting metrics for {service}: {e}")
                time.sleep(interval)
        
        self.collector_thread = Thread(target=collect, daemon=True)
        self.collector_thread.start()
        logger.info("Metrics collection started")
    
    def stop_collection(self):
        self.running = False
        self.collector_thread.join()
        logger.info("Metrics collection stopped")
    
    def get_metrics(self) -> Optional[ServiceMetrics]:
        """Get the next available metrics from the queue"""
        try:
            return self.metrics_queue.get_nowait()
        except:
            return None
    
    def _simulate_metrics_fetch(self, service_name: str) -> ServiceMetrics:
        """Simulate fetching metrics from a service"""
        base_cpu = random.uniform(10, 30)
        base_mem = random.uniform(20, 40)
        base_latency = random.uniform(50, 150)
        base_error = random.uniform(0, 0.5)
        
        if random.random() < 0.25:
            anomaly_factor = random.uniform(2, 8)
            cpu = min(100, base_cpu * anomaly_factor)
            mem = min(100, base_mem * anomaly_factor)
            latency = base_latency * anomaly_factor
            error_rate = min(1.0, base_error * anomaly_factor * 3)
            if random.random() < 0.05:
                cpu = 100
                mem = 100
                latency = 1000
                error_rate = 100.0
        else:
            cpu, mem, latency, error_rate = base_cpu, base_mem, base_latency, base_error
        
        return ServiceMetrics(
            service_name=service_name,
            cpu_usage=cpu,
            memory_usage=mem,
            latency=latency,
            error_rate=error_rate,
            request_rate=random.uniform(100, 1000),
            instance_count=random.randint(1, 5)
        )

class HealthAssessor:
    def __init__(self):
        self.thresholds = {
            'cpu_usage': {'warning': 50, 'critical': 70},
            'memory_usage': {'warning': 60, 'critical': 75},
            'latency': {'warning': 200, 'critical': 400},
            'error_rate': {'warning': 0.5, 'critical': 1.0}
        }
    
    def assess_health(self, metrics: ServiceMetrics) -> ServiceHealth:
        status = ServiceStatus.HEALTHY
        violations = 0
        warnings = 0
        
        if metrics.cpu_usage > self.thresholds['cpu_usage']['critical']:
            violations += 1
        if metrics.memory_usage > self.thresholds['memory_usage']['critical']:
            violations += 1
        if metrics.latency > self.thresholds['latency']['critical']:
            violations += 1
        if metrics.error_rate > self.thresholds['error_rate']['critical']:
            violations += 1
            
        if violations >= 2:
            status = ServiceStatus.FAILED
        elif violations >= 1:
            status = ServiceStatus.UNHEALTHY
        else:
            if metrics.cpu_usage > self.thresholds['cpu_usage']['warning']:
                warnings += 1
            if metrics.memory_usage > self.thresholds['memory_usage']['warning']:
                warnings += 1
            if metrics.latency > self.thresholds['latency']['warning']:
                warnings += 1
            if metrics.error_rate > self.thresholds['error_rate']['warning']:
                warnings += 1
                
            if warnings >= 2:
                status = ServiceStatus.DEGRADED
        
        confidence = 1.0 - (0.1 * violations + 0.05 * warnings)
        
        return ServiceHealth(
            service_name=metrics.service_name,
            status=status,
            confidence=max(0.5, confidence),
            predicted_failure=False,
            timestamp=time.time()
        )

class FailurePredictor:
    def __init__(self):
        self.model = None
        self.last_trained = 0
        self.training_interval = 3600
        self.load_model()
    
    def load_model(self):
        try:
            self.model = RandomForestClassifier(n_estimators=100)
            self._train_with_synthetic_data()
            logger.info("Failure prediction model initialized")
        except Exception as e:
            logger.error(f"Error initializing prediction model: {e}")
            raise
    
    def predict_failure(self, metrics: ServiceMetrics, health: ServiceHealth) -> Tuple[bool, float]:
        if time.time() - self.last_trained > self.training_interval:
            self._train_with_synthetic_data()
        
        features = np.array([
            metrics.cpu_usage,
            metrics.memory_usage,
            metrics.latency,
            metrics.error_rate,
            metrics.request_rate,
            health.confidence
        ]).reshape(1, -1)
        
        try:
            proba = self.model.predict_proba(features)[0][1]
            prediction = proba > 0.3
            return prediction, proba
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return False, 0.0
    
    def _train_with_synthetic_data(self):
        np.random.seed(42)
        n_samples = 1000
        X = np.random.rand(n_samples, 6) * 100
        X[:, 5] = np.random.rand(n_samples)
        
        y = np.zeros(n_samples)
        failure_conditions = (
            (X[:, 0] > 60) |
            (X[:, 1] > 65) |
            (X[:, 3] > 0.8) |
            ((X[:, 2] > 300) & (X[:, 5] < 0.8))
        )
        
        y[failure_conditions] = 1
        
        # Fixed the negative dimensions issue here
        target_failures = int(n_samples * 0.4)
        current_failures = np.sum(y)
        needed_failures = max(0, target_failures - current_failures)
        
        if needed_failures > 0:
            healthy_indices = np.where(y == 0)[0]
            if len(healthy_indices) >= needed_failures:
                additional_failures = np.random.choice(
                    healthy_indices,
                    size=needed_failures,
                    replace=False
                )
                y[additional_failures] = 1
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Model retrained. Test accuracy: {accuracy:.2f}")
        self.last_trained = time.time()

class HealingDecisionEngine:
    def __init__(self, service_dependencies: Dict[str, List[str]]):
        self.service_dependencies = service_dependencies
        self.action_history = []
    
    def decide_healing_action(self, health: ServiceHealth, metrics: ServiceMetrics) -> HealingDecision:
        action = HealingAction.NOTHING
        details = "No action needed"
        confidence = health.confidence
        
        if health.status == ServiceStatus.FAILED:
            action = HealingAction.RESTART
            details = "Critical failure detected. Immediate restart required."
            confidence = 1.0
        elif health.predicted_failure:
            if metrics.cpu_usage > 70 or metrics.memory_usage > 70:
                action = HealingAction.SCALE_UP
                details = "High resource usage predicted to cause failure. Scaling up."
                confidence *= 0.9
            elif metrics.error_rate > 1.0:
                action = HealingAction.REROUTE_TRAFFIC
                details = "High error rate predicted. Rerouting traffic to healthy instances."
                confidence *= 0.8
        elif health.status == ServiceStatus.UNHEALTHY:
            if metrics.instance_count > 1:
                action = HealingAction.REROUTE_TRAFFIC
                details = "Service unhealthy. Rerouting traffic."
                confidence *= 0.7
            else:
                action = HealingAction.RESTART
                details = "Service unhealthy with single instance. Restarting."
                confidence *= 0.6
        elif health.status == ServiceStatus.DEGRADED:
            if metrics.request_rate > 600 and metrics.instance_count < 3:
                action = HealingAction.SCALE_UP
                details = "High traffic causing degraded performance. Scaling up."
                confidence *= 0.5
            elif metrics.request_rate < 300 and metrics.instance_count > 1:
                action = HealingAction.SCALE_DOWN
                details = "Low traffic with excess capacity. Scaling down."
                confidence *= 0.4
        
        if action != HealingAction.NOTHING:
            dependent_services = self.service_dependencies.get(health.service_name, [])
            if dependent_services:
                details += f" Warning: This may affect dependent services: {', '.join(dependent_services)}"
                confidence *= 0.9
        
        decision = HealingDecision(
            action=action,
            service_name=health.service_name,
            confidence=confidence,
            details=details
        )
        self.action_history.append(decision)
        
        return decision

class Orchestrator:
    def __init__(self, k8s_api_endpoint: str):
        self.k8s_api_endpoint = k8s_api_endpoint
        self.running = False
    
    def execute_action(self, decision: HealingDecision) -> bool:
        logger.info(f"Executing action: {decision.action.name} for {decision.service_name}")
        logger.info(f"Details: {decision.details}")
        
        try:
            if decision.action == HealingAction.RESTART:
                self._simulate_restart(decision.service_name)
            elif decision.action == HealingAction.SCALE_UP:
                self._simulate_scale(decision.service_name, 1)
            elif decision.action == HealingAction.SCALE_DOWN:
                self._simulate_scale(decision.service_name, -1)
            elif decision.action == HealingAction.REROUTE_TRAFFIC:
                self._simulate_reroute(decision.service_name)
            
            logger.info(f"Successfully executed action: {decision.action.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to execute action {decision.action.name}: {e}")
            return False
    
    def _simulate_restart(self, service_name: str):
        logger.info(f"Simulating restart of {service_name}")
        time.sleep(1)
    
    def _simulate_scale(self, service_name: str, delta: int):
        logger.info(f"Simulating scaling {service_name} by {delta} instances")
        time.sleep(2)
    
    def _simulate_reroute(self, service_name: str):
        logger.info(f"Simulating traffic rerouting for {service_name}")
        time.sleep(1.5)

class HealthOrchestrator:
    def __init__(self, config: Dict):
        self.config = config
        self.running = False
        self.metrics_collector = MetricsCollector(config['service_endpoints'])
        self.health_assessor = HealthAssessor()
        self.failure_predictor = FailurePredictor()
        self.decision_engine = HealingDecisionEngine(config['service_dependencies'])
        self.orchestrator = Orchestrator(config['k8s_api_endpoint'])
        self.processing_queue = Queue()
    
    def start(self):
        self.running = True
        self.metrics_collector.start_collection()
        self.processing_thread = Thread(target=self._process_metrics, daemon=True)
        self.processing_thread.start()
        logger.info("Health Orchestrator started")
    
    def stop(self):
        self.running = False
        self.metrics_collector.stop_collection()
        self.processing_thread.join()
        logger.info("Health Orchestrator stopped")
    
    def _process_metrics(self):
        while self.running:
            metrics = self.metrics_collector.get_metrics()
            if metrics is None:
                time.sleep(0.1)
                continue
            
            try:
                health = self.health_assessor.assess_health(metrics)
                will_fail, failure_prob = self.failure_predictor.predict_failure(metrics, health)
                health.predicted_failure = will_fail
                health.confidence = (health.confidence + (1 - failure_prob)) / 2
                
                logger.info(f"Service {health.service_name} status: {health.status.name}, "
                          f"Predicted failure: {will_fail}, Confidence: {health.confidence:.2f}")
                
                decision = self.decision_engine.decide_healing_action(health, metrics)
                
                if decision.action != HealingAction.NOTHING:
                    logger.info(f"Recommended action for {decision.service_name}: {decision.action.name} "
                              f"(confidence: {decision.confidence:.2f})")
                    
                    if decision.confidence > 0.5:
                        self.orchestrator.execute_action(decision)
                    else:
                        logger.warning(f"Low confidence ({decision.confidence:.2f}) - skipping action")
                
            except Exception as e:
                logger.error(f"Error processing metrics for {metrics.service_name if metrics else 'unknown'}: {e}")

# Example configuration
config = {
    'service_endpoints': {
        'user-service': 'http://user-service/metrics',
        'order-service': 'http://order-service/metrics',
        'payment-service': 'http://payment-service/metrics',
        'inventory-service': 'http://inventory-service/metrics'
    },
    'service_dependencies': {
        'user-service': ['order-service', 'payment-service'],
        'inventory-service': ['order-service'],
        'payment-service': ['order-service']
    },
    'k8s_api_endpoint': 'https://kubernetes-api.example.com'
}

if __name__ == "__main__":
    orchestrator = HealthOrchestrator(config)
    orchestrator.start()
    
    try:
        time.sleep(300)
    except KeyboardInterrupt:
        pass
    finally:
        orchestrator.stop();