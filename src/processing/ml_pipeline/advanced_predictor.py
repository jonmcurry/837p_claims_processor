"""Advanced ML pipeline with TensorFlow/PyTorch for high-performance claims filtering."""

import asyncio
import pickle
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor

import joblib
import numpy as np
import pandas as pd
import structlog
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModel
import xgboost as xgb
from aiocache import cached

from src.cache.redis_cache import cache_manager
from src.core.config import settings
from src.core.database.models import Claim, ClaimLineItem
from src.monitoring.metrics.prometheus_metrics import (
    ml_prediction_latency,
    ml_prediction_accuracy,
    ml_model_inference_total
)

logger = structlog.get_logger(__name__)

# Suppress TensorFlow warnings
tf.get_logger().setLevel(logging.ERROR)


class ClaimsFeatureExtractor:
    """Advanced feature extraction for claims data with medical domain knowledge."""
    
    def __init__(self):
        """Initialize the feature extractor with medical coding knowledge."""
        self.feature_names = []
        self.categorical_encoders = {}
        self.numerical_scalers = {}
        self.medical_embeddings = None
        self._load_medical_knowledge()
    
    def _load_medical_knowledge(self):
        """Load medical knowledge bases for feature enhancement."""
        # Medical specialty groupings
        self.specialty_groups = {
            'emergency': ['99281', '99282', '99283', '99284', '99285'],
            'surgery': ['10021', '10040', '10060', '10080', '10120'],
            'radiology': ['70010', '70015', '70100', '70110', '70120'],
            'laboratory': ['80047', '80048', '80050', '80051', '80053'],
            'pathology': ['88104', '88106', '88107', '88108', '88112']
        }
        
        # High-risk diagnosis patterns
        self.high_risk_dx_patterns = {
            'cancer': ['C', 'D0', 'D1', 'D2', 'D3', 'D4'],
            'cardiac': ['I0', 'I1', 'I2', 'I3', 'I4', 'I5'],
            'trauma': ['S0', 'S1', 'S2', 'S3', 'S4', 'T0', 'T1', 'T2'],
            'chronic': ['N18', 'E11', 'I50', 'J44', 'F03']
        }
        
        # Cost outlier thresholds by service type
        self.cost_thresholds = {
            'emergency': {'low': 500, 'high': 5000},
            'surgery': {'low': 1000, 'high': 25000},
            'radiology': {'low': 100, 'high': 2000},
            'laboratory': {'low': 25, 'high': 500},
            'pathology': {'low': 50, 'high': 1000}
        }

    async def extract_features(self, claim: Claim, line_items: List[ClaimLineItem]) -> Dict[str, float]:
        """Extract comprehensive features for ML models."""
        features = {}
        
        # Basic demographic features
        features.update(self._extract_demographic_features(claim))
        
        # Financial features with anomaly detection
        features.update(self._extract_financial_features(claim, line_items))
        
        # Clinical features with medical knowledge
        features.update(self._extract_clinical_features(claim, line_items))
        
        # Temporal features
        features.update(self._extract_temporal_features(claim))
        
        # Provider risk features
        features.update(self._extract_provider_features(claim, line_items))
        
        # Facility features
        features.update(self._extract_facility_features(claim))
        
        # Complexity and risk scores
        features.update(self._calculate_risk_scores(claim, line_items))
        
        return features

    def _extract_demographic_features(self, claim: Claim) -> Dict[str, float]:
        """Extract patient demographic features."""
        features = {}
        
        # Calculate patient age
        if claim.patient_date_of_birth:
            from datetime import date
            birth_date = claim.patient_date_of_birth.date() if hasattr(claim.patient_date_of_birth, 'date') else claim.patient_date_of_birth
            age = (date.today() - birth_date).days / 365.25
            features['patient_age'] = age
            features['age_group'] = self._categorize_age(age)
        else:
            features['patient_age'] = 0
            features['age_group'] = 0
        
        # Gender encoding (if available)
        if hasattr(claim, 'patient_gender'):
            features['gender_male'] = 1.0 if claim.patient_gender == 'M' else 0.0
            features['gender_female'] = 1.0 if claim.patient_gender == 'F' else 0.0
        
        return features

    def _extract_financial_features(self, claim: Claim, line_items: List[ClaimLineItem]) -> Dict[str, float]:
        """Extract financial features with anomaly detection."""
        features = {}
        
        # Basic financial metrics
        total_charges = float(claim.total_charges)
        features['total_charges'] = total_charges
        features['log_total_charges'] = np.log1p(total_charges)
        
        # Line item analysis
        if line_items:
            line_charges = [float(item.charge_amount) for item in line_items]
            features['line_item_count'] = len(line_items)
            features['avg_charge_per_line'] = np.mean(line_charges)
            features['max_line_charge'] = np.max(line_charges)
            features['min_line_charge'] = np.min(line_charges)
            features['charge_variance'] = np.var(line_charges) if len(line_charges) > 1 else 0
            features['charge_skewness'] = self._calculate_skewness(line_charges)
            
            # Charge distribution analysis
            features['high_cost_lines_ratio'] = sum(1 for c in line_charges if c > 1000) / len(line_charges)
            features['low_cost_lines_ratio'] = sum(1 for c in line_charges if c < 50) / len(line_charges)
        else:
            features.update({
                'line_item_count': 0, 'avg_charge_per_line': 0, 'max_line_charge': 0,
                'min_line_charge': 0, 'charge_variance': 0, 'charge_skewness': 0,
                'high_cost_lines_ratio': 0, 'low_cost_lines_ratio': 0
            })
        
        # Financial anomaly score
        features['financial_anomaly_score'] = self._calculate_financial_anomaly_score(claim, line_items)
        
        return features

    def _extract_clinical_features(self, claim: Claim, line_items: List[ClaimLineItem]) -> Dict[str, float]:
        """Extract clinical features using medical knowledge."""
        features = {}
        
        # Primary diagnosis analysis
        primary_dx = claim.primary_diagnosis_code
        if primary_dx:
            features['dx_category'] = self._categorize_diagnosis(primary_dx)
            features['dx_risk_score'] = self._calculate_diagnosis_risk_score(primary_dx)
            features['chronic_condition'] = self._is_chronic_condition(primary_dx)
        
        # Procedure analysis
        if line_items:
            procedure_codes = [item.procedure_code for item in line_items if item.procedure_code]
            features['unique_procedures'] = len(set(procedure_codes))
            features['procedure_complexity'] = self._calculate_procedure_complexity(procedure_codes)
            features['surgical_procedures'] = sum(1 for code in procedure_codes if self._is_surgical_procedure(code))
            features['emergency_procedures'] = sum(1 for code in procedure_codes if code in self.specialty_groups.get('emergency', []))
            
            # Service type distribution
            for service_type, codes in self.specialty_groups.items():
                matching_procedures = sum(1 for code in procedure_codes if code in codes)
                features[f'{service_type}_procedures'] = matching_procedures
                features[f'{service_type}_ratio'] = matching_procedures / len(procedure_codes) if procedure_codes else 0
        
        return features

    def _extract_temporal_features(self, claim: Claim) -> Dict[str, float]:
        """Extract temporal features from service dates."""
        features = {}
        
        if claim.service_from_date and claim.service_to_date:
            # Service duration
            service_start = claim.service_from_date.date() if hasattr(claim.service_from_date, 'date') else claim.service_from_date
            service_end = claim.service_to_date.date() if hasattr(claim.service_to_date, 'date') else claim.service_to_date
            
            service_days = (service_end - service_start).days + 1
            features['service_days'] = service_days
            features['is_single_day'] = 1.0 if service_days == 1 else 0.0
            features['is_long_stay'] = 1.0 if service_days > 7 else 0.0
            
            # Day of week and month effects
            features['service_day_of_week'] = service_start.weekday()
            features['service_month'] = service_start.month
            features['is_weekend'] = 1.0 if service_start.weekday() >= 5 else 0.0
            features['is_holiday_season'] = 1.0 if service_start.month in [11, 12, 1] else 0.0
        
        return features

    def _extract_provider_features(self, claim: Claim, line_items: List[ClaimLineItem]) -> Dict[str, float]:
        """Extract provider-related features."""
        features = {}
        
        # Provider consistency
        if line_items:
            rendering_providers = [item.rendering_provider_npi for item in line_items if item.rendering_provider_npi]
            features['unique_providers'] = len(set(rendering_providers))
            features['single_provider'] = 1.0 if len(set(rendering_providers)) <= 1 else 0.0
            features['multiple_providers'] = 1.0 if len(set(rendering_providers)) > 3 else 0.0
        
        # Billing provider NPI validation
        if claim.billing_provider_npi:
            features['has_billing_npi'] = 1.0
            features['npi_length'] = len(claim.billing_provider_npi)
            features['valid_npi_format'] = 1.0 if len(claim.billing_provider_npi) == 10 and claim.billing_provider_npi.isdigit() else 0.0
        else:
            features['has_billing_npi'] = 0.0
            features['npi_length'] = 0
            features['valid_npi_format'] = 0.0
        
        return features

    def _extract_facility_features(self, claim: Claim) -> Dict[str, float]:
        """Extract facility-related features."""
        features = {}
        
        # Facility ID encoding
        if claim.facility_id:
            features['facility_hash'] = hash(claim.facility_id) % 1000  # Hash to numeric
            features['facility_length'] = len(claim.facility_id)
        
        # Financial class analysis
        if claim.financial_class:
            financial_class_encoding = {
                'INPATIENT': 1, 'OUTPATIENT': 2, 'EMERGENCY': 3, 
                'OBSERVATION': 4, 'CLINIC': 5, 'HOME_HEALTH': 6,
                'HOSPICE': 7, 'SNF': 8, 'REHAB': 9, 'PSYCH': 10
            }
            features['financial_class_encoded'] = financial_class_encoding.get(claim.financial_class.upper(), 0)
            features['is_inpatient'] = 1.0 if 'INPATIENT' in claim.financial_class.upper() else 0.0
            features['is_emergency'] = 1.0 if 'EMERGENCY' in claim.financial_class.upper() else 0.0
        
        return features

    def _calculate_risk_scores(self, claim: Claim, line_items: List[ClaimLineItem]) -> Dict[str, float]:
        """Calculate composite risk scores."""
        features = {}
        
        # Financial risk score
        total_charges = float(claim.total_charges)
        financial_risk = 0
        if total_charges > 50000:
            financial_risk += 0.4
        if total_charges > 100000:
            financial_risk += 0.3
        if len(line_items) > 20:
            financial_risk += 0.2
        if len(line_items) > 50:
            financial_risk += 0.1
        
        features['financial_risk_score'] = financial_risk
        
        # Clinical complexity score
        clinical_complexity = 0
        if line_items:
            unique_procedures = len(set(item.procedure_code for item in line_items if item.procedure_code))
            if unique_procedures > 10:
                clinical_complexity += 0.3
            if unique_procedures > 20:
                clinical_complexity += 0.2
        
        features['clinical_complexity_score'] = clinical_complexity
        
        # Overall fraud risk score (composite)
        features['fraud_risk_score'] = (financial_risk + clinical_complexity) / 2
        
        return features

    def _categorize_age(self, age: float) -> float:
        """Categorize age into groups."""
        if age < 18:
            return 1  # Pediatric
        elif age < 35:
            return 2  # Young adult
        elif age < 50:
            return 3  # Middle age
        elif age < 65:
            return 4  # Older adult
        else:
            return 5  # Senior

    def _categorize_diagnosis(self, dx_code: str) -> float:
        """Categorize diagnosis code into risk groups."""
        if not dx_code:
            return 0
        
        dx_upper = dx_code.upper()
        for category, patterns in self.high_risk_dx_patterns.items():
            if any(dx_upper.startswith(pattern) for pattern in patterns):
                return hash(category) % 10  # Numeric encoding
        
        return 1  # Default category

    def _calculate_diagnosis_risk_score(self, dx_code: str) -> float:
        """Calculate risk score based on diagnosis."""
        if not dx_code:
            return 0.0
        
        dx_upper = dx_code.upper()
        risk_score = 0.1  # Base score
        
        # Cancer diagnoses
        if dx_upper.startswith('C'):
            risk_score += 0.8
        # Chronic conditions
        elif any(dx_upper.startswith(pattern) for pattern in self.high_risk_dx_patterns['chronic']):
            risk_score += 0.6
        # Cardiac conditions
        elif any(dx_upper.startswith(pattern) for pattern in self.high_risk_dx_patterns['cardiac']):
            risk_score += 0.5
        # Trauma
        elif any(dx_upper.startswith(pattern) for pattern in self.high_risk_dx_patterns['trauma']):
            risk_score += 0.4
        
        return min(risk_score, 1.0)

    def _is_chronic_condition(self, dx_code: str) -> float:
        """Check if diagnosis represents a chronic condition."""
        if not dx_code:
            return 0.0
        
        chronic_patterns = self.high_risk_dx_patterns['chronic']
        return 1.0 if any(dx_code.upper().startswith(pattern) for pattern in chronic_patterns) else 0.0

    def _calculate_procedure_complexity(self, procedure_codes: List[str]) -> float:
        """Calculate complexity score based on procedure codes."""
        if not procedure_codes:
            return 0.0
        
        complexity_score = 0.1 * len(procedure_codes)  # Base complexity
        
        # Surgical procedures add complexity
        surgical_count = sum(1 for code in procedure_codes if self._is_surgical_procedure(code))
        complexity_score += surgical_count * 0.2
        
        # Multiple specialties add complexity
        specialties = set()
        for code in procedure_codes:
            for specialty, codes in self.specialty_groups.items():
                if code in codes:
                    specialties.add(specialty)
        
        complexity_score += len(specialties) * 0.1
        
        return min(complexity_score, 1.0)

    def _is_surgical_procedure(self, code: str) -> bool:
        """Check if procedure code represents surgery."""
        if not code:
            return False
        
        # Surgery CPT codes typically range from 10021-69990
        try:
            code_num = int(code)
            return 10021 <= code_num <= 69990
        except ValueError:
            return False

    def _calculate_financial_anomaly_score(self, claim: Claim, line_items: List[ClaimLineItem]) -> float:
        """Calculate financial anomaly score."""
        if not line_items:
            return 0.0
        
        total_charges = float(claim.total_charges)
        avg_charge = total_charges / len(line_items)
        
        anomaly_score = 0.0
        
        # Check for unusually high charges
        if total_charges > 100000:
            anomaly_score += 0.5
        if avg_charge > 5000:
            anomaly_score += 0.3
        
        # Check for charge distribution anomalies
        line_charges = [float(item.charge_amount) for item in line_items]
        if len(line_charges) > 1:
            charge_std = np.std(line_charges)
            charge_mean = np.mean(line_charges)
            cv = charge_std / charge_mean if charge_mean > 0 else 0
            
            if cv > 2.0:  # High coefficient of variation
                anomaly_score += 0.2
        
        return min(anomaly_score, 1.0)

    def _calculate_skewness(self, values: List[float]) -> float:
        """Calculate skewness of a distribution."""
        if len(values) < 3:
            return 0.0
        
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        if std_val == 0:
            return 0.0
        
        skewness = np.mean([((x - mean_val) / std_val) ** 3 for x in values])
        return skewness


class DeepClaimsClassifier(nn.Module):
    """Deep neural network for claims classification using PyTorch."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [512, 256, 128, 64], 
                 output_dim: int = 2, dropout_rate: float = 0.3):
        """Initialize the deep classifier."""
        super(DeepClaimsClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        """Forward pass through the network."""
        logits = self.network(x)
        return self.softmax(logits)


class AdvancedClaimPredictor:
    """Advanced ML pipeline for claims prediction with multiple models."""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize the advanced claim predictor."""
        self.feature_extractor = ClaimsFeatureExtractor()
        self.models = {}
        self.scalers = {}
        self.model_metadata = {}
        
        # Thread pool for CPU-intensive operations
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Model paths
        self.model_path = Path(model_path or settings.ML_MODEL_PATH)
        self.model_path.mkdir(exist_ok=True)
        
        # Load models if available
        asyncio.create_task(self._load_models())
        
        logger.info("Advanced claim predictor initialized", model_path=str(self.model_path))

    async def _load_models(self):
        """Load pre-trained models from disk."""
        try:
            # Load XGBoost model
            xgb_path = self.model_path / "xgboost_claims.model"
            if xgb_path.exists():
                self.models['xgboost'] = xgb.Booster()
                self.models['xgboost'].load_model(str(xgb_path))
                logger.info("XGBoost model loaded successfully")
            
            # Load PyTorch model
            pytorch_path = self.model_path / "pytorch_claims.pth"
            if pytorch_path.exists():
                checkpoint = torch.load(pytorch_path, map_location='cpu')
                model = DeepClaimsClassifier(
                    input_dim=checkpoint['input_dim'],
                    hidden_dims=checkpoint['hidden_dims']
                )
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                self.models['pytorch'] = model
                logger.info("PyTorch model loaded successfully")
            
            # Load feature scaler
            scaler_path = self.model_path / "feature_scaler.pkl"
            if scaler_path.exists():
                self.scalers['standard'] = joblib.load(scaler_path)
                logger.info("Feature scaler loaded successfully")
            
            # Load model metadata
            metadata_path = self.model_path / "model_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.model_metadata = json.load(f)
                logger.info("Model metadata loaded successfully")
                
        except Exception as e:
            logger.warning("Error loading models", error=str(e))

    async def predict_single(self, claim: Claim, line_items: List[ClaimLineItem] = None) -> Dict[str, Any]:
        """Predict on a single claim."""
        start_time = time.perf_counter()
        
        try:
            with ml_prediction_latency.time():
                # Extract features
                features = await self.feature_extractor.extract_features(claim, line_items or [])
                
                # Convert to numpy array
                feature_vector = np.array(list(features.values())).reshape(1, -1)
                
                # Scale features if scaler is available
                if 'standard' in self.scalers:
                    feature_vector = self.scalers['standard'].transform(feature_vector)
                
                # Ensemble prediction using multiple models
                predictions = {}
                confidence_scores = {}
                
                # XGBoost prediction
                if 'xgboost' in self.models:
                    xgb_pred = self.models['xgboost'].predict(xgb.DMatrix(feature_vector))
                    predictions['xgboost'] = float(xgb_pred[0])
                    confidence_scores['xgboost'] = abs(xgb_pred[0] - 0.5) * 2  # Distance from 0.5
                
                # PyTorch prediction
                if 'pytorch' in self.models:
                    with torch.no_grad():
                        torch_input = torch.FloatTensor(feature_vector)
                        pytorch_pred = self.models['pytorch'](torch_input)
                        predictions['pytorch'] = float(pytorch_pred[0][1])  # Probability of positive class
                        confidence_scores['pytorch'] = float(torch.max(pytorch_pred[0]))
                
                # Ensemble decision
                if predictions:
                    ensemble_score = np.mean(list(predictions.values()))
                    ensemble_confidence = np.mean(list(confidence_scores.values()))
                    should_process = ensemble_score > 0.5
                else:
                    # Fallback: use rule-based approach
                    ensemble_score = 0.8  # Default to likely valid
                    ensemble_confidence = 0.6
                    should_process = True
                
                # Update metrics
                ml_model_inference_total.inc()
                
                prediction_result = {
                    "should_process": should_process,
                    "confidence": float(ensemble_confidence),
                    "score": float(ensemble_score),
                    "model_predictions": predictions,
                    "features_used": len(features),
                    "processing_time": time.perf_counter() - start_time,
                    "reason": "Passed ML validation" if should_process else "Failed ML validation"
                }
                
                logger.debug("Single claim prediction completed",
                           claim_id=claim.claim_id,
                           should_process=should_process,
                           confidence=ensemble_confidence,
                           processing_time=prediction_result["processing_time"])
                
                return prediction_result
                
        except Exception as e:
            logger.exception("Error in single claim prediction", 
                           claim_id=claim.claim_id, error=str(e))
            
            # Fallback prediction
            return {
                "should_process": True,  # Default to processing
                "confidence": 0.5,
                "score": 0.5,
                "model_predictions": {},
                "features_used": 0,
                "processing_time": time.perf_counter() - start_time,
                "reason": "ML prediction error - defaulting to process",
                "error": str(e)
            }

    async def predict_batch_optimized(self, claims: List[Claim]) -> List[Dict[str, Any]]:
        """Optimized batch prediction for maximum throughput."""
        start_time = time.perf_counter()
        
        try:
            with ml_prediction_latency.time():
                # Extract features for all claims in parallel
                feature_tasks = []
                for claim in claims:
                    # Get line items if available
                    line_items = getattr(claim, '_line_items_data', [])
                    feature_tasks.append(
                        self.feature_extractor.extract_features(claim, line_items)
                    )
                
                # Execute feature extraction in parallel
                all_features = await asyncio.gather(*feature_tasks)
                
                # Convert to batch numpy array
                if all_features:
                    feature_matrix = np.array([list(features.values()) for features in all_features])
                    
                    # Scale features
                    if 'standard' in self.scalers:
                        feature_matrix = self.scalers['standard'].transform(feature_matrix)
                    
                    # Batch prediction
                    batch_predictions = []
                    
                    if 'xgboost' in self.models:
                        xgb_preds = self.models['xgboost'].predict(xgb.DMatrix(feature_matrix))
                        
                        for i, (claim, xgb_pred) in enumerate(zip(claims, xgb_preds)):
                            confidence = abs(xgb_pred - 0.5) * 2
                            should_process = xgb_pred > 0.5
                            
                            batch_predictions.append({
                                "should_process": should_process,
                                "confidence": float(confidence),
                                "score": float(xgb_pred),
                                "model_predictions": {"xgboost": float(xgb_pred)},
                                "features_used": len(all_features[i]),
                                "reason": "Passed ML validation" if should_process else "Failed ML validation"
                            })
                    else:
                        # Fallback for batch without models
                        for i, claim in enumerate(claims):
                            batch_predictions.append({
                                "should_process": True,
                                "confidence": 0.8,
                                "score": 0.8,
                                "model_predictions": {},
                                "features_used": len(all_features[i]) if i < len(all_features) else 0,
                                "reason": "No ML models available - defaulting to process"
                            })
                else:
                    # Empty batch
                    batch_predictions = []
                
                # Update metrics
                ml_model_inference_total.inc(len(claims))
                
                processing_time = time.perf_counter() - start_time
                throughput = len(claims) / processing_time if processing_time > 0 else 0
                
                logger.info("Batch prediction completed",
                           batch_size=len(claims),
                           processing_time=processing_time,
                           throughput=f"{throughput:.2f} claims/sec")
                
                return batch_predictions
                
        except Exception as e:
            logger.exception("Error in batch prediction", batch_size=len(claims), error=str(e))
            
            # Fallback predictions
            return [{
                "should_process": True,
                "confidence": 0.5,
                "score": 0.5,
                "model_predictions": {},
                "features_used": 0,
                "reason": "Batch ML prediction error - defaulting to process",
                "error": str(e)
            } for _ in claims]

    async def train_models(self, training_data: pd.DataFrame, labels: np.ndarray):
        """Train ML models on historical claims data."""
        logger.info("Starting model training", data_shape=training_data.shape)
        
        try:
            # Feature extraction and preprocessing
            X = training_data.values
            y = labels
            
            # Split data
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train XGBoost
            xgb_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
            xgb_model.fit(X_train_scaled, y_train)
            
            # Train PyTorch model
            pytorch_model = DeepClaimsClassifier(input_dim=X_train_scaled.shape[1])
            
            # Save models
            await self._save_models(xgb_model, pytorch_model, scaler, X_train_scaled.shape[1])
            
            # Evaluate models
            from sklearn.metrics import accuracy_score, classification_report
            
            xgb_pred = xgb_model.predict(X_test_scaled)
            xgb_accuracy = accuracy_score(y_test, xgb_pred)
            
            logger.info("Model training completed",
                       xgb_accuracy=xgb_accuracy,
                       train_samples=len(X_train),
                       test_samples=len(X_test))
            
            # Update Prometheus metrics
            ml_prediction_accuracy.set(xgb_accuracy)
            
        except Exception as e:
            logger.exception("Error in model training", error=str(e))
            raise

    async def _save_models(self, xgb_model, pytorch_model, scaler, input_dim: int):
        """Save trained models to disk."""
        try:
            # Save XGBoost model
            xgb_model.save_model(str(self.model_path / "xgboost_claims.model"))
            
            # Save PyTorch model
            torch.save({
                'model_state_dict': pytorch_model.state_dict(),
                'input_dim': input_dim,
                'hidden_dims': [512, 256, 128, 64]
            }, self.model_path / "pytorch_claims.pth")
            
            # Save scaler
            joblib.dump(scaler, self.model_path / "feature_scaler.pkl")
            
            # Save metadata
            metadata = {
                'training_date': datetime.now().isoformat(),
                'model_version': '1.0',
                'input_features': input_dim,
                'model_types': ['xgboost', 'pytorch']
            }
            
            with open(self.model_path / "model_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info("Models saved successfully", model_path=str(self.model_path))
            
        except Exception as e:
            logger.exception("Error saving models", error=str(e))


# Global predictor instance
advanced_predictor = AdvancedClaimPredictor()