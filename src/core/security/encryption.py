"""Field-level encryption for PHI/PII data."""

import base64
import json
from typing import Any, Dict, List, Optional, Union

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from src.core.config import settings


class PHIEncryption:
    """Handle encryption/decryption of PHI fields for HIPAA compliance."""

    # Fields that must be encrypted
    PHI_FIELDS = {
        "patient_ssn",
        "patient_first_name",
        "patient_last_name",
        "patient_middle_name",
        "patient_date_of_birth",
        "medical_record_number",
        "subscriber_id",
    }

    def __init__(self, encryption_key: Optional[str] = None):
        """Initialize encryption handler."""
        key = encryption_key or settings.encryption_key.get_secret_value()
        self._setup_cipher(key)

    def _setup_cipher(self, key: str) -> None:
        """Setup Fernet cipher with derived key."""
        # Derive a proper key from the provided key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b"stable_salt_for_app",  # In production, use unique salt per field
            iterations=100000,
        )
        derived_key = base64.urlsafe_b64encode(kdf.derive(key.encode()))
        self.cipher = Fernet(derived_key)

    def encrypt_field(self, value: Any) -> str:
        """Encrypt a single field value."""
        if value is None:
            return None

        # Convert value to string for encryption
        str_value = str(value)
        encrypted = self.cipher.encrypt(str_value.encode())
        return base64.urlsafe_b64encode(encrypted).decode()

    def decrypt_field(self, encrypted_value: str) -> str:
        """Decrypt a single field value."""
        if encrypted_value is None:
            return None

        try:
            decoded = base64.urlsafe_b64decode(encrypted_value.encode())
            decrypted = self.cipher.decrypt(decoded)
            return decrypted.decode()
        except Exception:
            # Log decryption failure but don't expose error details
            return "[DECRYPTION_ERROR]"

    def encrypt_dict(self, data: Dict[str, Any], fields_to_encrypt: Optional[List[str]] = None) -> Dict[str, Any]:
        """Encrypt specified fields in a dictionary."""
        if not data:
            return data

        fields = fields_to_encrypt or self.PHI_FIELDS
        encrypted_data = data.copy()

        for field in fields:
            if field in encrypted_data and encrypted_data[field] is not None:
                encrypted_data[field] = self.encrypt_field(encrypted_data[field])

        return encrypted_data

    def decrypt_dict(self, data: Dict[str, Any], fields_to_decrypt: Optional[List[str]] = None) -> Dict[str, Any]:
        """Decrypt specified fields in a dictionary."""
        if not data:
            return data

        fields = fields_to_decrypt or self.PHI_FIELDS
        decrypted_data = data.copy()

        for field in fields:
            if field in decrypted_data and decrypted_data[field] is not None:
                decrypted_data[field] = self.decrypt_field(decrypted_data[field])

        return decrypted_data

    def mask_phi(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Mask PHI fields for non-production environments."""
        if settings.is_production:
            return data

        masked_data = data.copy()
        mask_patterns = {
            "patient_ssn": "XXX-XX-####",
            "patient_first_name": "Patient",
            "patient_last_name": "####",
            "patient_middle_name": "X",
            "medical_record_number": "MRN####",
            "subscriber_id": "SUB####",
        }

        for field, pattern in mask_patterns.items():
            if field in masked_data and masked_data[field]:
                if "####" in pattern:
                    # Preserve last 4 characters
                    original = str(masked_data[field])
                    if len(original) >= 4:
                        masked_data[field] = pattern.replace("####", original[-4:])
                    else:
                        masked_data[field] = pattern.replace("####", "0000")
                else:
                    masked_data[field] = pattern

        return masked_data

    def tokenize_phi(self, value: str, field_name: str) -> str:
        """Create a reversible token for PHI data."""
        if not value:
            return None

        # Create a token that includes field type for proper detokenization
        token_data = {
            "field": field_name,
            "value": value,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        token_str = json.dumps(token_data)
        encrypted_token = self.cipher.encrypt(token_str.encode())
        return f"TOK_{base64.urlsafe_b64encode(encrypted_token).decode()}"

    def detokenize_phi(self, token: str) -> Dict[str, Any]:
        """Retrieve original PHI data from token."""
        if not token or not token.startswith("TOK_"):
            raise ValueError("Invalid token format")

        try:
            token_data = token[4:]  # Remove TOK_ prefix
            decoded = base64.urlsafe_b64decode(token_data.encode())
            decrypted = self.cipher.decrypt(decoded)
            return json.loads(decrypted.decode())
        except Exception as e:
            raise ValueError(f"Failed to detokenize: {str(e)}")


class DataAnonymizer:
    """Anonymize data for analytics and reporting."""

    def __init__(self):
        """Initialize anonymizer."""
        self.encryption = PHIEncryption()

    def anonymize_claim(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """Anonymize a claim for analytics."""
        anonymized = claim_data.copy()

        # Remove direct identifiers
        identifiers_to_remove = [
            "patient_first_name",
            "patient_last_name",
            "patient_middle_name",
            "patient_ssn",
            "medical_record_number",
            "subscriber_id",
        ]

        for field in identifiers_to_remove:
            anonymized.pop(field, None)

        # Generalize quasi-identifiers
        if "patient_date_of_birth" in anonymized:
            # Convert to age group
            dob = anonymized["patient_date_of_birth"]
            age = self._calculate_age(dob)
            anonymized["age_group"] = self._get_age_group(age)
            del anonymized["patient_date_of_birth"]

        # Hash facility and provider IDs for consistency
        if "facility_id" in anonymized:
            anonymized["facility_id"] = self._hash_identifier(anonymized["facility_id"])

        if "billing_provider_npi" in anonymized:
            anonymized["provider_id"] = self._hash_identifier(anonymized["billing_provider_npi"])
            del anonymized["billing_provider_npi"]

        return anonymized

    def _calculate_age(self, dob: Union[str, datetime]) -> int:
        """Calculate age from date of birth."""
        if isinstance(dob, str):
            dob = datetime.fromisoformat(dob)
        
        today = datetime.utcnow()
        age = today.year - dob.year
        
        if today.month < dob.month or (today.month == dob.month and today.day < dob.day):
            age -= 1
            
        return age

    def _get_age_group(self, age: int) -> str:
        """Convert age to age group."""
        if age < 18:
            return "0-17"
        elif age < 30:
            return "18-29"
        elif age < 40:
            return "30-39"
        elif age < 50:
            return "40-49"
        elif age < 65:
            return "50-64"
        else:
            return "65+"

    def _hash_identifier(self, identifier: str) -> str:
        """Create consistent hash of identifier."""
        import hashlib
        
        # Use SHA256 with app-specific salt
        salted = f"{identifier}:{settings.secret_key.get_secret_value()}"
        return hashlib.sha256(salted.encode()).hexdigest()[:16]


# Global instances
phi_encryption = PHIEncryption()
data_anonymizer = DataAnonymizer()