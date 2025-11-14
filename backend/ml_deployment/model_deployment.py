"""
ML Model Deployment with Canary Strategy

Uses BentoML for:
- Model containerization
- API serving
- Traffic splitting (canary deployment)
- Performance monitoring

Architecture:
- BentoML service with FastAPI
- Kubernetes Deployment with HorizontalPodAutoscaler
- Istio/Linkerd for traffic splitting
- Prometheus metrics integration
"""

from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
import os

import bentoml
from bentoml.io import JSON, NumpyNdarray
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


# ============================================================================
# BentoML Service Definition
# ============================================================================

@bentoml.service(
    resources={"cpu": "2", "memory": "4Gi"},
    traffic={"timeout": 30},
)
class WastePredictor:
    """
    BentoML service for waste prediction.
    
    Serves waste classification & regression models with:
    - Automatic batching
    - Request/response logging
    - Performance metrics
    - Model version tracking
    """
    
    # Model references (loaded on init)
    waste_classifier = bentoml.sklearn.get("waste_classifier:latest")
    waste_regressor = bentoml.sklearn.get("waste_regressor:latest")
    
    def __init__(self):
        """Initialize service."""
        # Load models
        self.classifier = self.waste_classifier.to_runner()
        self.regressor = self.waste_regressor.to_runner()
        
        # Get model versions
        self.classifier_version = self.waste_classifier.tag.version
        self.regressor_version = self.waste_regressor.tag.version
        
        logger.info(
            f"WastePredictor initialized "
            f"(classifier: v{self.classifier_version}, "
            f"regressor: v{self.regressor_version})"
        )
    
    @bentoml.api
    async def predict(
        self,
        features: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Predict waste risk for pantry item.
        
        Args:
            features: Feature dictionary with keys:
                - age_days: float
                - days_to_expiry: float
                - quantity_on_hand: float
                - avg_daily_consumption_7d: float
                - household_size: int
                - category_waste_rate: float
                - ... (50+ features)
        
        Returns:
            Prediction result:
                - waste_probability: float [0-1]
                - risk_class: str ("LOW", "MEDIUM", "HIGH")
                - days_until_waste: float
                - model_versions: dict
                - timestamp: str
        """
        start_time = datetime.utcnow()
        
        try:
            # Convert features to DataFrame
            feature_df = pd.DataFrame([features])
            
            # Classification - waste probability
            waste_proba = await self.classifier.predict_proba.async_run(feature_df)
            waste_probability = float(waste_proba[0][1])
            
            # Classification risk class
            if waste_probability >= 0.7:
                risk_class = "HIGH"
            elif waste_probability >= 0.4:
                risk_class = "MEDIUM"
            else:
                risk_class = "LOW"
            
            # Regression - days until waste
            days_until_waste_pred = await self.regressor.predict.async_run(feature_df)
            days_until_waste = float(days_until_waste_pred[0])
            
            # Build response
            result = {
                "waste_probability": waste_probability,
                "risk_class": risk_class,
                "days_until_waste": max(0.0, days_until_waste),
                "model_versions": {
                    "classifier": self.classifier_version,
                    "regressor": self.regressor_version,
                },
                "timestamp": datetime.utcnow().isoformat(),
            }
            
            # Log metrics
            processing_time_ms = (
                datetime.utcnow() - start_time
            ).total_seconds() * 1000
            
            logger.info(
                f"Prediction: waste_prob={waste_probability:.3f}, "
                f"days_to_waste={days_until_waste:.1f}, "
                f"time={processing_time_ms:.0f}ms"
            )
            
            return result
        
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    @bentoml.api
    async def predict_batch(
        self,
        features_list: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Batch prediction for multiple items.
        
        Args:
            features_list: List of feature dictionaries
        
        Returns:
            List of prediction results
        """
        # Convert to DataFrame
        feature_df = pd.DataFrame(features_list)
        
        # Batch predictions
        waste_proba_batch = await self.classifier.predict_proba.async_run(feature_df)
        days_until_waste_batch = await self.regressor.predict.async_run(feature_df)
        
        # Build results
        results = []
        for i in range(len(features_list)):
            waste_probability = float(waste_proba_batch[i][1])
            days_until_waste = float(days_until_waste_batch[i])
            
            if waste_probability >= 0.7:
                risk_class = "HIGH"
            elif waste_probability >= 0.4:
                risk_class = "MEDIUM"
            else:
                risk_class = "LOW"
            
            results.append({
                "waste_probability": waste_probability,
                "risk_class": risk_class,
                "days_until_waste": max(0.0, days_until_waste),
                "model_versions": {
                    "classifier": self.classifier_version,
                    "regressor": self.regressor_version,
                },
                "timestamp": datetime.utcnow().isoformat(),
            })
        
        logger.info(f"Batch prediction completed ({len(results)} items)")
        
        return results
    
    @bentoml.api
    def health(self) -> Dict[str, str]:
        """Health check endpoint."""
        return {
            "status": "healthy",
            "classifier_version": self.classifier_version,
            "regressor_version": self.regressor_version,
            "timestamp": datetime.utcnow().isoformat(),
        }


# ============================================================================
# Model Deployment Manager
# ============================================================================

class ModelDeploymentManager:
    """Manager for model deployment lifecycle."""
    
    def __init__(
        self,
        bentoml_registry: str = "gcr.io/pantrypal-prod/bentoml",
        k8s_namespace: str = "ml-serving",
    ):
        """
        Initialize deployment manager.
        
        Args:
            bentoml_registry: Container registry for BentoML images
            k8s_namespace: Kubernetes namespace for deployments
        """
        self.registry = bentoml_registry
        self.namespace = k8s_namespace
        
        logger.info(f"ModelDeploymentManager initialized (registry: {bentoml_registry})")
    
    def build_bento(
        self,
        service_name: str,
        models: Dict[str, str],
        python_packages: Optional[List[str]] = None,
    ) -> str:
        """
        Build BentoML bundle.
        
        Args:
            service_name: Service name (e.g., "waste_predictor")
            models: Dict of model_name -> model_tag
            python_packages: Additional Python packages
        
        Returns:
            Bento tag
        """
        logger.info(f"Building Bento for {service_name}")
        
        # Create bentofile.yaml
        bentofile_content = f"""
service: "{service_name}:WastePredictor"
labels:
  owner: ml-team
  project: pantrypal
include:
  - "*.py"
python:
  packages:
    - scikit-learn>=1.3.0
    - lightgbm>=4.0.0
    - pandas>=2.0.0
    - numpy>=1.24.0
    {f"    - {chr(10).join(python_packages)}" if python_packages else ""}
models:
{chr(10).join([f'  - "{name}:{tag}"' for name, tag in models.items()])}
"""
        
        # Write bentofile
        with open("bentofile.yaml", "w") as f:
            f.write(bentofile_content)
        
        # Build Bento
        import subprocess
        
        result = subprocess.run(
            ["bentoml", "build"],
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Bento build failed: {result.stderr}")
        
        # Parse bento tag from output
        # Output: "Successfully built Bento(tag='waste_predictor:abc123')"
        import re
        match = re.search(r"tag='([^']+)'", result.stdout)
        if not match:
            raise RuntimeError("Failed to parse Bento tag")
        
        bento_tag = match.group(1)
        
        logger.info(f"Built Bento: {bento_tag}")
        
        return bento_tag
    
    def containerize_bento(self, bento_tag: str) -> str:
        """
        Containerize Bento as Docker image.
        
        Args:
            bento_tag: Bento tag (e.g., "waste_predictor:abc123")
        
        Returns:
            Docker image name
        """
        logger.info(f"Containerizing Bento {bento_tag}")
        
        # Get service name and version
        service_name, version = bento_tag.split(":")
        
        # Build image name
        image_name = f"{self.registry}/{service_name}:{version}"
        
        # Containerize
        import subprocess
        
        result = subprocess.run(
            ["bentoml", "containerize", bento_tag, "-t", image_name],
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Containerization failed: {result.stderr}")
        
        logger.info(f"Containerized: {image_name}")
        
        # Push to registry
        result = subprocess.run(
            ["docker", "push", image_name],
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Push failed: {result.stderr}")
        
        logger.info(f"Pushed to registry: {image_name}")
        
        return image_name
    
    def deploy_to_k8s(
        self,
        image_name: str,
        service_name: str,
        replicas: int = 2,
        cpu_request: str = "1",
        memory_request: str = "2Gi",
        cpu_limit: str = "2",
        memory_limit: str = "4Gi",
        max_replicas: int = 10,
    ) -> str:
        """
        Deploy to Kubernetes with HorizontalPodAutoscaler.
        
        Args:
            image_name: Docker image name
            service_name: Service name
            replicas: Initial replicas
            cpu_request: CPU request
            memory_request: Memory request
            cpu_limit: CPU limit
            memory_limit: Memory limit
            max_replicas: Max replicas for autoscaling
        
        Returns:
            Deployment name
        """
        deployment_name = f"{service_name}-deployment"
        
        # Create Kubernetes manifests
        deployment_yaml = f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {deployment_name}
  namespace: {self.namespace}
  labels:
    app: {service_name}
    version: latest
spec:
  replicas: {replicas}
  selector:
    matchLabels:
      app: {service_name}
  template:
    metadata:
      labels:
        app: {service_name}
        version: latest
    spec:
      containers:
      - name: {service_name}
        image: {image_name}
        ports:
        - containerPort: 3000
          name: http
        resources:
          requests:
            cpu: "{cpu_request}"
            memory: "{memory_request}"
          limits:
            cpu: "{cpu_limit}"
            memory: "{memory_limit}"
        livenessProbe:
          httpGet:
            path: /health
            port: 3000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 3000
          initialDelaySeconds: 10
          periodSeconds: 5
        env:
        - name: BENTOML_PORT
          value: "3000"
        - name: BENTOML_NUM_WORKERS
          value: "4"
---
apiVersion: v1
kind: Service
metadata:
  name: {service_name}
  namespace: {self.namespace}
spec:
  selector:
    app: {service_name}
  ports:
  - protocol: TCP
    port: 80
    targetPort: 3000
  type: ClusterIP
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {service_name}-hpa
  namespace: {self.namespace}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {deployment_name}
  minReplicas: {replicas}
  maxReplicas: {max_replicas}
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
"""
        
        # Write manifest
        manifest_file = f"{deployment_name}.yaml"
        with open(manifest_file, "w") as f:
            f.write(deployment_yaml)
        
        # Apply to cluster
        import subprocess
        
        result = subprocess.run(
            ["kubectl", "apply", "-f", manifest_file],
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Deployment failed: {result.stderr}")
        
        logger.info(f"Deployed {deployment_name} to Kubernetes")
        
        return deployment_name
    
    def deploy_canary(
        self,
        new_image: str,
        service_name: str,
        traffic_percent: int = 10,
    ):
        """
        Deploy canary version with traffic splitting.
        
        Uses Istio VirtualService for traffic splitting.
        
        Args:
            new_image: New model image
            service_name: Service name
            traffic_percent: Percentage of traffic to canary (0-100)
        """
        canary_deployment_name = f"{service_name}-canary"
        
        # Deploy canary deployment (similar to main deployment)
        canary_yaml = f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {canary_deployment_name}
  namespace: {self.namespace}
  labels:
    app: {service_name}
    version: canary
spec:
  replicas: 1
  selector:
    matchLabels:
      app: {service_name}
      version: canary
  template:
    metadata:
      labels:
        app: {service_name}
        version: canary
    spec:
      containers:
      - name: {service_name}
        image: {new_image}
        ports:
        - containerPort: 3000
        # ... (same resources/probes as main)
"""
        
        # Istio VirtualService for traffic splitting
        virtual_service_yaml = f"""
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: {service_name}
  namespace: {self.namespace}
spec:
  hosts:
  - {service_name}
  http:
  - match:
    - headers:
        x-canary:
          exact: "true"
    route:
    - destination:
        host: {service_name}
        subset: canary
  - route:
    - destination:
        host: {service_name}
        subset: stable
      weight: {100 - traffic_percent}
    - destination:
        host: {service_name}
        subset: canary
      weight: {traffic_percent}
---
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: {service_name}
  namespace: {self.namespace}
spec:
  host: {service_name}
  subsets:
  - name: stable
    labels:
      version: latest
  - name: canary
    labels:
      version: canary
"""
        
        # Apply manifests
        import subprocess
        
        # Deploy canary
        with open(f"{canary_deployment_name}.yaml", "w") as f:
            f.write(canary_yaml)
        
        subprocess.run(["kubectl", "apply", "-f", f"{canary_deployment_name}.yaml"])
        
        # Apply traffic splitting
        with open(f"{service_name}-vs.yaml", "w") as f:
            f.write(virtual_service_yaml)
        
        subprocess.run(["kubectl", "apply", "-f", f"{service_name}-vs.yaml"])
        
        logger.info(
            f"Deployed canary for {service_name} "
            f"(traffic: {traffic_percent}%)"
        )


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Initialize manager
    manager = ModelDeploymentManager(
        bentoml_registry="gcr.io/pantrypal-prod/bentoml",
        k8s_namespace="ml-serving",
    )
    
    # Build Bento
    bento_tag = manager.build_bento(
        service_name="waste_predictor",
        models={
            "waste_classifier": "v1.2.0",
            "waste_regressor": "v1.2.0",
        },
    )
    
    # Containerize
    image_name = manager.containerize_bento(bento_tag)
    
    # Deploy to Kubernetes
    deployment_name = manager.deploy_to_k8s(
        image_name=image_name,
        service_name="waste-predictor",
        replicas=2,
        max_replicas=10,
    )
    
    print(f"\nDeployed {deployment_name}")
    
    # Canary deployment (10% traffic)
    # manager.deploy_canary(
    #     new_image="gcr.io/pantrypal-prod/bentoml/waste_predictor:v1.3.0",
    #     service_name="waste-predictor",
    #     traffic_percent=10,
    # )
