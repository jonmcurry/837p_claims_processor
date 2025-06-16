"""ML-based claims prediction and filtering system."""

import asyncio
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import structlog
import tensorflow as tf
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from src.cache.redis_cache import cache_manager
from src.core.config import settings
from src.core.database.models import Claim, ClaimLineItem

logger = structlog.get_logger(__name__)


class FeatureExtractor:
    """Extract features from claims for ML models."""

    def __init__(self):
        """Initialize feature extractor."""
        self.categorical_encoders = {}
        self.numerical_scalers = {}
        self.feature_names = []

    def extract_claim_features(self, claim: Claim, line_items: List[ClaimLineItem]) -> Dict[str, Any]:
        """Extract comprehensive features from a claim."""
        features = {}

        # Basic claim features
        features.update(self._extract_basic_features(claim))
        
        # Patient demographics
        features.update(self._extract_demographic_features(claim))
        
        # Financial features
        features.update(self._extract_financial_features(claim, line_items))
        
        # Provider features
        features.update(self._extract_provider_features(claim, line_items))
        
        # Temporal features
        features.update(self._extract_temporal_features(claim))
        
        # Diagnosis features
        features.update(self._extract_diagnosis_features(claim))
        
        # Procedure features
        features.update(self._extract_procedure_features(line_items))
        
        # Statistical features
        features.update(self._extract_statistical_features(claim, line_items))

        return features

    def _extract_basic_features(self, claim: Claim) -> Dict[str, Any]:
        """Extract basic claim features."""
        return {
            "facility_id_hash": hash(claim.facility_id) % 10000,  # Anonymized facility
            "insurance_type_encoded": self._encode_categorical(claim.insurance_type, "insurance_type"),
            "financial_class_encoded": self._encode_categorical(claim.financial_class, "financial_class"),
        }

    def _extract_demographic_features(self, claim: Claim) -> Dict[str, Any]:
        """Extract patient demographic features."""
        # Calculate age
        age = (datetime.utcnow() - claim.patient_date_of_birth).days // 365
        
        return {
            "patient_age": age,
            "patient_age_group": self._get_age_group(age),
            "has_middle_name": 1 if claim.patient_middle_name else 0,
        }

    def _extract_financial_features(self, claim: Claim, line_items: List[ClaimLineItem]) -> Dict[str, Any]:
        """Extract financial features."""
        line_item_charges = [item.charge_amount for item in line_items]
        
        features = {
            "total_charges": float(claim.total_charges),
            "log_total_charges": np.log1p(float(claim.total_charges)),
            "line_item_count": len(line_items),
        }

        if line_item_charges:
            features.update({
                "avg_line_item_charge": np.mean(line_item_charges),
                "max_line_item_charge": np.max(line_item_charges),
                "min_line_item_charge": np.min(line_item_charges),
                "std_line_item_charge": np.std(line_item_charges),
                "charge_variance": np.var(line_item_charges),
            })
        else:
            features.update({
                "avg_line_item_charge": 0,
                "max_line_item_charge": 0,
                "min_line_item_charge": 0,
                "std_line_item_charge": 0,
                "charge_variance": 0,
            })

        return features

    def _extract_provider_features(self, claim: Claim, line_items: List[ClaimLineItem]) -> Dict[str, Any]:
        """Extract provider-related features."""
        unique_rendering_providers = set()
        for item in line_items:
            if item.rendering_provider_npi:
                unique_rendering_providers.add(item.rendering_provider_npi)

        return {
            "billing_provider_hash": hash(claim.billing_provider_npi) % 10000,
            "has_attending_provider": 1 if claim.attending_provider_npi else 0,
            "unique_rendering_providers": len(unique_rendering_providers),
        }

    def _extract_temporal_features(self, claim: Claim) -> Dict[str, Any]:
        """Extract temporal features."""
        service_duration = (claim.service_to_date - claim.service_from_date).days + 1
        admission_duration = (claim.discharge_date - claim.admission_date).days + 1
        
        # Time-based features
        service_month = claim.service_from_date.month
        service_weekday = claim.service_from_date.weekday()
        
        return {
            "service_duration_days": service_duration,
            "admission_duration_days": admission_duration,
            "service_month": service_month,
            "service_weekday": service_weekday,
            "is_weekend_service": 1 if service_weekday >= 5 else 0,
            "service_quarter": (service_month - 1) // 3 + 1,
        }

    def _extract_diagnosis_features(self, claim: Claim) -> Dict[str, Any]:
        """Extract diagnosis-related features."""
        primary_dx = claim.primary_diagnosis_code
        dx_codes = claim.diagnosis_codes or []
        
        features = {
            "primary_dx_category": primary_dx[0] if primary_dx else 'Z',  # First letter
            "total_diagnosis_codes": len(dx_codes) + 1,  # Include primary
        }

        # Common diagnosis categories
        mental_health_codes = ['F', 'Z']
        injury_codes = ['S', 'T']
        
        features["is_mental_health"] = 1 if primary_dx and primary_dx[0] in mental_health_codes else 0
        features["is_injury"] = 1 if primary_dx and primary_dx[0] in injury_codes else 0

        return features

    def _extract_procedure_features(self, line_items: List[ClaimLineItem]) -> Dict[str, Any]:
        """Extract procedure-related features."""
        if not line_items:
            return {
                "unique_procedures": 0,
                "total_units": 0,
                "avg_units_per_procedure": 0,
                "has_surgery_codes": 0,
            }

        procedure_codes = [item.procedure_code for item in line_items]
        unique_procedures = len(set(procedure_codes))
        total_units = sum(item.units for item in line_items)
        
        # Check for surgery codes (typically 10000-69999)
        surgery_codes = [code for code in procedure_codes 
                        if code.isdigit() and 10000 <= int(code) <= 69999]

        return {
            "unique_procedures": unique_procedures,
            "total_units": total_units,
            "avg_units_per_procedure": total_units / len(line_items),
            "has_surgery_codes": 1 if surgery_codes else 0,
            "surgery_procedure_ratio": len(surgery_codes) / len(procedure_codes),
        }

    def _extract_statistical_features(self, claim: Claim, line_items: List[ClaimLineItem]) -> Dict[str, Any]:
        """Extract statistical features for anomaly detection."""
        features = {}

        # Historical comparison features (would require historical data)
        # For now, we'll use simple heuristics
        
        # Charge per day ratios
        service_days = (claim.service_to_date - claim.service_from_date).days + 1
        features["charge_per_service_day"] = float(claim.total_charges) / service_days

        # Units per dollar ratios
        if line_items:
            total_units = sum(item.units for item in line_items)
            features["units_per_dollar"] = total_units / float(claim.total_charges) if claim.total_charges > 0 else 0

        return features

    def _encode_categorical(self, value: str, category: str) -> int:
        """Encode categorical values to integers."""
        if category not in self.categorical_encoders:
            self.categorical_encoders[category] = {}
        
        encoder = self.categorical_encoders[category]
        if value not in encoder:
            encoder[value] = len(encoder)
        
        return encoder[value]

    def _get_age_group(self, age: int) -> int:
        """Convert age to age group."""
        if age < 18:
            return 0
        elif age < 30:
            return 1
        elif age < 50:
            return 2
        elif age < 65:
            return 3
        else:
            return 4


class ClaimPredictor:
    """ML-based claim prediction system."""

    def __init__(self):
        """Initialize the predictor."""
        self.feature_extractor = FeatureExtractor()
        self.models = {}
        self.model_metadata = {}
        self.is_loaded = False
        
    async def load_models(self) -> None:
        """Load trained ML models."""
        try:
            model_path = settings.ml_model_path
            
            if model_path.exists():
                # Load different model types
                await self._load_tensorflow_model(model_path)
                await self._load_sklearn_models(model_path.parent)
                
                self.is_loaded = True
                logger.info("ML models loaded successfully", model_path=str(model_path))
            else:
                logger.warning("ML model not found, using rule-based fallback", 
                             model_path=str(model_path))
                await self._initialize_fallback_model()
                
        except Exception as e:
            logger.exception("Failed to load ML models", error=str(e))
            await self._initialize_fallback_model()

    async def _load_tensorflow_model(self, model_path: Path) -> None:
        """Load TensorFlow/Keras model."""
        try:
            tf_model_path = model_path.parent / "tensorflow_model.h5"
            if tf_model_path.exists():
                self.models["tensorflow"] = tf.keras.models.load_model(str(tf_model_path))
                logger.info("TensorFlow model loaded", path=str(tf_model_path))
        except Exception as e:
            logger.warning("Failed to load TensorFlow model", error=str(e))

    async def _load_sklearn_models(self, model_dir: Path) -> None:
        """Load scikit-learn models."""
        try:
            # Load main classifier
            classifier_path = model_dir / "random_forest_classifier.pkl"
            if classifier_path.exists():
                self.models["classifier"] = joblib.load(classifier_path)
                
            # Load scaler
            scaler_path = model_dir / "feature_scaler.pkl"
            if scaler_path.exists():
                self.models["scaler"] = joblib.load(scaler_path)
                
            # Load feature metadata
            metadata_path = model_dir / "model_metadata.pkl"
            if metadata_path.exists():
                self.model_metadata = joblib.load(metadata_path)
                
            logger.info("Scikit-learn models loaded", model_dir=str(model_dir))
            
        except Exception as e:
            logger.warning("Failed to load scikit-learn models", error=str(e))

    async def _initialize_fallback_model(self) -> None:
        """Initialize rule-based fallback model."""
        # Simple rule-based model as fallback
        self.models["fallback"] = RuleBasedPredictor()
        self.is_loaded = True
        logger.info("Initialized rule-based fallback model")

    async def predict_single(self, claim: Claim, line_items: List[ClaimLineItem] = None) -> Dict[str, Any]:
        """Predict processing outcome for a single claim."""
        if not self.is_loaded:
            await self.load_models()

        line_items = line_items or []
        
        try:
            # Extract features
            features = self.feature_extractor.extract_claim_features(claim, line_items)
            
            # Get predictions from available models
            predictions = {}
            
            if "tensorflow" in self.models:
                tf_pred = await self._predict_tensorflow(features)
                predictions["tensorflow"] = tf_pred
                
            if "classifier" in self.models:
                sklearn_pred = await self._predict_sklearn(features)
                predictions["sklearn"] = sklearn_pred
                
            if "fallback" in self.models:
                fallback_pred = await self._predict_fallback(claim, line_items)
                predictions["fallback"] = fallback_pred

            # Ensemble predictions
            final_prediction = await self._ensemble_predictions(predictions)
            
            return final_prediction
            
        except Exception as e:
            logger.exception("Prediction failed for claim", claim_id=claim.claim_id, error=str(e))
            # Return safe fallback
            return {
                "should_process": True,
                "confidence": 0.5,
                "reason": "prediction_error",
                "model_used": "error_fallback"
            }

    async def predict_batch(self, claims: List[Claim], line_items_map: Dict[int, List[ClaimLineItem]] = None) -> List[Dict[str, Any]]:
        """Predict processing outcomes for a batch of claims."""
        if not self.is_loaded:
            await self.load_models()

        line_items_map = line_items_map or {}
        
        # Process claims in smaller batches for memory efficiency
        batch_size = settings.ml_batch_size
        results = []
        
        for i in range(0, len(claims), batch_size):
            batch_claims = claims[i:i + batch_size]
            
            # Create batch features
            batch_features = []
            for claim in batch_claims:
                claim_line_items = line_items_map.get(claim.id, [])
                features = self.feature_extractor.extract_claim_features(claim, claim_line_items)
                batch_features.append(features)

            # Batch prediction
            batch_predictions = await self._predict_batch_features(batch_features, batch_claims)
            results.extend(batch_predictions)

        return results

    async def _predict_tensorflow(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction using TensorFlow model."""
        model = self.models["tensorflow"]
        
        # Convert features to numpy array (assuming model expects specific feature order)
        feature_vector = self._features_to_vector(features)
        feature_array = np.array([feature_vector])
        
        # Make prediction
        prediction = model.predict(feature_array, verbose=0)[0]
        
        return {
            "should_process": prediction[0] > settings.ml_prediction_threshold,
            "confidence": float(prediction[0]),
            "model": "tensorflow"
        }

    async def _predict_sklearn(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction using scikit-learn model."""
        classifier = self.models["classifier"]
        scaler = self.models.get("scaler")
        
        # Convert features to numpy array
        feature_vector = self._features_to_vector(features)
        
        if scaler:
            feature_vector = scaler.transform([feature_vector])[0]
        
        # Make prediction
        prediction_proba = classifier.predict_proba([feature_vector])[0]
        
        # Assuming binary classification (reject=0, approve=1)
        approval_confidence = prediction_proba[1] if len(prediction_proba) > 1 else prediction_proba[0]
        
        return {
            "should_process": approval_confidence > settings.ml_prediction_threshold,
            "confidence": float(approval_confidence),
            "model": "sklearn"
        }

    async def _predict_fallback(self, claim: Claim, line_items: List[ClaimLineItem]) -> Dict[str, Any]:
        """Make prediction using rule-based fallback."""
        predictor = self.models["fallback"]
        return await predictor.predict(claim, line_items)

    async def _predict_batch_features(self, batch_features: List[Dict], claims: List[Claim]) -> List[Dict[str, Any]]:
        """Predict on a batch of feature vectors."""
        results = []
        
        # Use the best available model for batch prediction
        if "tensorflow" in self.models:
            results = await self._predict_tensorflow_batch(batch_features)
        elif "classifier" in self.models:
            results = await self._predict_sklearn_batch(batch_features)
        else:
            # Fallback to individual predictions
            for i, claim in enumerate(claims):
                result = await self._predict_fallback(claim, [])
                results.append(result)

        return results

    async def _predict_tensorflow_batch(self, batch_features: List[Dict]) -> List[Dict[str, Any]]:
        """Batch prediction using TensorFlow model."""
        model = self.models["tensorflow"]
        
        # Convert batch features to numpy array
        feature_vectors = [self._features_to_vector(features) for features in batch_features]
        feature_array = np.array(feature_vectors)
        
        # Batch prediction
        predictions = model.predict(feature_array, verbose=0)
        
        results = []
        for prediction in predictions:
            results.append({
                "should_process": prediction[0] > settings.ml_prediction_threshold,
                "confidence": float(prediction[0]),
                "model": "tensorflow"
            })
        
        return results

    async def _predict_sklearn_batch(self, batch_features: List[Dict]) -> List[Dict[str, Any]]:
        """Batch prediction using scikit-learn model."""
        classifier = self.models["classifier"]
        scaler = self.models.get("scaler")
        
        # Convert batch features to numpy array
        feature_vectors = [self._features_to_vector(features) for features in batch_features]
        feature_array = np.array(feature_vectors)
        
        if scaler:
            feature_array = scaler.transform(feature_array)
        
        # Batch prediction
        predictions_proba = classifier.predict_proba(feature_array)
        
        results = []
        for prediction_proba in predictions_proba:
            approval_confidence = prediction_proba[1] if len(prediction_proba) > 1 else prediction_proba[0]
            results.append({
                "should_process": approval_confidence > settings.ml_prediction_threshold,
                "confidence": float(approval_confidence),
                "model": "sklearn"
            })
        
        return results

    async def _ensemble_predictions(self, predictions: Dict[str, Dict]) -> Dict[str, Any]:
        """Combine predictions from multiple models."""
        if not predictions:
            return {
                "should_process": True,
                "confidence": 0.5,
                "reason": "no_predictions",
                "model_used": "default"
            }

        # Weighted ensemble based on model reliability
        weights = {
            "tensorflow": 0.4,
            "sklearn": 0.4,
            "fallback": 0.2
        }

        weighted_confidence = 0.0
        total_weight = 0.0
        models_used = []

        for model_name, prediction in predictions.items():
            if model_name in weights:
                weight = weights[model_name]
                weighted_confidence += prediction["confidence"] * weight
                total_weight += weight
                models_used.append(model_name)

        final_confidence = weighted_confidence / total_weight if total_weight > 0 else 0.5
        should_process = final_confidence > settings.ml_prediction_threshold

        return {
            "should_process": should_process,
            "confidence": final_confidence,
            "reason": "ensemble_decision",
            "model_used": "+".join(models_used),
            "individual_predictions": predictions
        }

    def _features_to_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """Convert feature dictionary to numpy vector."""
        # This should match the feature order used during training
        # For now, we'll use a simple ordered approach
        
        expected_features = [
            "facility_id_hash", "insurance_type_encoded", "financial_class_encoded",
            "patient_age", "patient_age_group", "has_middle_name",
            "total_charges", "log_total_charges", "line_item_count",
            "avg_line_item_charge", "max_line_item_charge", "min_line_item_charge",
            "std_line_item_charge", "charge_variance",
            "billing_provider_hash", "has_attending_provider", "unique_rendering_providers",
            "service_duration_days", "admission_duration_days", "service_month",
            "service_weekday", "is_weekend_service", "service_quarter",
            "primary_dx_category", "total_diagnosis_codes", "is_mental_health", "is_injury",
            "unique_procedures", "total_units", "avg_units_per_procedure",
            "has_surgery_codes", "surgery_procedure_ratio",
            "charge_per_service_day", "units_per_dollar"
        ]

        vector = []
        for feature_name in expected_features:
            value = features.get(feature_name, 0)
            
            # Handle categorical encoding
            if isinstance(value, str):
                value = hash(value) % 1000  # Simple hash encoding
            elif value is None:
                value = 0
            
            vector.append(float(value))

        return np.array(vector)


class RuleBasedPredictor:
    """Rule-based fallback predictor."""

    async def predict(self, claim: Claim, line_items: List[ClaimLineItem]) -> Dict[str, Any]:
        """Simple rule-based prediction."""
        # Basic rules for claim acceptance
        
        # Reject if charges are unreasonably high
        if claim.total_charges > 100000:
            return {
                "should_process": False,
                "confidence": 0.9,
                "reason": "high_charges",
                "model": "rule_based"
            }
        
        # Reject if no line items
        if not line_items:
            return {
                "should_process": False,
                "confidence": 0.95,
                "reason": "no_line_items",
                "model": "rule_based"
            }
        
        # Accept most other claims
        return {
            "should_process": True,
            "confidence": 0.7,
            "reason": "rule_based_approval",
            "model": "rule_based"
        }


# Global predictor instance
claim_predictor = ClaimPredictor()