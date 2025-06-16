"""Comprehensive HIPAA-compliant security implementation for claims processing."""

import asyncio
import hashlib
import secrets
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Union
from dataclasses import dataclass
from enum import Enum
import json
import ipaddress
import re

from cryptography.fernet import Fernet, MultiFernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend
import bcrypt
import jwt
from passlib.context import CryptContext
import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.core.database.base import get_postgres_session
from src.core.database.models import AuditLog

logger = structlog.get_logger(__name__)


class SecurityLevel(str, Enum):
    """Security classification levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    PHI = "phi"  # Protected Health Information
    PII = "pii"  # Personally Identifiable Information


class AccessType(str, Enum):
    """Types of data access."""
    READ = "read"
    WRITE = "write"
    UPDATE = "update"
    DELETE = "delete"
    EXPORT = "export"
    PRINT = "print"


@dataclass
class EncryptedField:
    """Represents an encrypted field with metadata."""
    field_name: str
    encrypted_value: bytes
    encryption_version: str
    security_level: SecurityLevel
    encrypted_at: datetime


@dataclass
class AccessContext:
    """Context for access control decisions."""
    user_id: str
    user_role: str
    ip_address: str
    session_id: str
    request_id: str
    access_type: AccessType
    resource_type: str
    resource_id: Optional[str] = None
    business_justification: Optional[str] = None


class HIPAAEncryption:
    """HIPAA-compliant encryption for PHI/PII data with key rotation support."""
    
    # Fields requiring encryption
    PHI_FIELDS = {
        "patient_ssn", "patient_first_name", "patient_last_name", 
        "patient_middle_name", "patient_date_of_birth", "medical_record_number",
        "subscriber_id", "patient_phone", "patient_email", "patient_address"
    }
    
    PII_FIELDS = {
        "billing_address", "emergency_contact", "insurance_member_id",
        "provider_ssn", "provider_dea_number"
    }
    
    def __init__(self):
        """Initialize HIPAA encryption with multiple keys for rotation."""
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self._initialize_encryption_keys()
        self.current_version = "v1"
        
        logger.info("HIPAA encryption initialized", 
                   phi_fields=len(self.PHI_FIELDS),
                   pii_fields=len(self.PII_FIELDS))

    def _initialize_encryption_keys(self):
        """Initialize encryption keys with rotation support."""
        try:
            # Primary encryption key (current)
            primary_key = settings.ENCRYPTION_KEY.encode()
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b'stable_salt_for_claims',  # In production, use random salt
                iterations=100000,
                backend=default_backend()
            )
            key = base64.urlsafe_b64encode(kdf.derive(primary_key))
            self.primary_fernet = Fernet(key)
            
            # Secondary key for rotation (if available)
            if hasattr(settings, 'ENCRYPTION_KEY_SECONDARY'):
                secondary_key = settings.ENCRYPTION_KEY_SECONDARY.encode()
                secondary_derived = base64.urlsafe_b64encode(kdf.derive(secondary_key))
                self.secondary_fernet = Fernet(secondary_derived)
                
                # MultiFernet for key rotation
                self.multi_fernet = MultiFernet([self.primary_fernet, self.secondary_fernet])
            else:
                self.multi_fernet = MultiFernet([self.primary_fernet])
                
        except Exception as e:
            logger.error("Failed to initialize encryption keys", error=str(e))
            raise

    def encrypt_field(self, field_name: str, value: Any, 
                     security_level: SecurityLevel = SecurityLevel.PHI) -> EncryptedField:
        """Encrypt a field value with metadata."""
        if value is None:
            return None
        
        try:
            # Convert value to string if not already
            str_value = str(value).encode('utf-8')
            
            # Encrypt using primary key
            encrypted_bytes = self.primary_fernet.encrypt(str_value)
            
            return EncryptedField(
                field_name=field_name,
                encrypted_value=encrypted_bytes,
                encryption_version=self.current_version,
                security_level=security_level,
                encrypted_at=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error("Field encryption failed", 
                        field=field_name, error=str(e))
            raise

    def decrypt_field(self, encrypted_field: EncryptedField) -> str:
        """Decrypt a field value."""
        if not encrypted_field or not encrypted_field.encrypted_value:
            return None
        
        try:
            # Use MultiFernet for backward compatibility during key rotation
            decrypted_bytes = self.multi_fernet.decrypt(encrypted_field.encrypted_value)
            return decrypted_bytes.decode('utf-8')
            
        except Exception as e:
            logger.error("Field decryption failed", 
                        field=encrypted_field.field_name, error=str(e))
            raise

    def encrypt_claim_phi(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt all PHI fields in a claim."""
        encrypted_claim = claim_data.copy()
        
        for field_name, value in claim_data.items():
            if field_name in self.PHI_FIELDS and value is not None:
                encrypted_field = self.encrypt_field(
                    field_name, value, SecurityLevel.PHI
                )
                # Store as base64 encoded string for database storage
                encrypted_claim[f"{field_name}_encrypted"] = base64.b64encode(
                    encrypted_field.encrypted_value
                ).decode('utf-8')
                
                # Remove original field
                encrypted_claim.pop(field_name, None)
            
            elif field_name in self.PII_FIELDS and value is not None:
                encrypted_field = self.encrypt_field(
                    field_name, value, SecurityLevel.PII
                )
                encrypted_claim[f"{field_name}_encrypted"] = base64.b64encode(
                    encrypted_field.encrypted_value
                ).decode('utf-8')
                
                # Remove original field
                encrypted_claim.pop(field_name, None)
        
        return encrypted_claim

    def decrypt_claim_phi(self, encrypted_claim: Dict[str, Any], 
                         access_context: AccessContext) -> Dict[str, Any]:
        """Decrypt PHI fields in a claim with access control."""
        # Verify access authorization
        if not self._authorize_phi_access(access_context):
            logger.warning("Unauthorized PHI access attempted",
                          user_id=access_context.user_id,
                          resource_id=access_context.resource_id)
            raise PermissionError("Insufficient privileges for PHI access")
        
        decrypted_claim = encrypted_claim.copy()
        
        for field_name in list(encrypted_claim.keys()):
            if field_name.endswith('_encrypted'):
                original_field = field_name.replace('_encrypted', '')
                
                if original_field in self.PHI_FIELDS or original_field in self.PII_FIELDS:
                    try:
                        # Decode from base64 and decrypt
                        encrypted_bytes = base64.b64decode(encrypted_claim[field_name])
                        encrypted_field = EncryptedField(
                            field_name=original_field,
                            encrypted_value=encrypted_bytes,
                            encryption_version=self.current_version,
                            security_level=SecurityLevel.PHI,
                            encrypted_at=datetime.utcnow()
                        )
                        
                        decrypted_value = self.decrypt_field(encrypted_field)
                        decrypted_claim[original_field] = decrypted_value
                        
                        # Remove encrypted field
                        decrypted_claim.pop(field_name, None)
                        
                    except Exception as e:
                        logger.error("Failed to decrypt field",
                                   field=original_field, error=str(e))
        
        return decrypted_claim

    def _authorize_phi_access(self, access_context: AccessContext) -> bool:
        """Authorize PHI access based on context."""
        # Implement role-based access control
        authorized_roles = {
            'admin', 'claims_processor', 'medical_reviewer', 
            'compliance_officer', 'auditor'
        }
        
        if access_context.user_role not in authorized_roles:
            return False
        
        # Check business justification for certain roles
        if access_context.user_role in ['auditor'] and not access_context.business_justification:
            return False
        
        # Additional IP-based restrictions could be added here
        
        return True

    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt."""
        return self.pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash."""
        return self.pwd_context.verify(plain_password, hashed_password)


class JWTManager:
    """JWT token management for authentication and authorization."""
    
    def __init__(self):
        """Initialize JWT manager."""
        self.secret_key = settings.JWT_SECRET_KEY
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 30
        self.refresh_token_expire_days = 7
        
    def create_access_token(self, data: Dict[str, Any], 
                          expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token."""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        })
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt

    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """Create JWT refresh token."""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh"
        })
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.JWTError as e:
            logger.warning("Token verification failed", error=str(e))
            return None


class AuditLogger:
    """HIPAA-compliant audit logging system."""
    
    def __init__(self):
        """Initialize audit logger."""
        self.encryption = HIPAAEncryption()
    
    async def log_phi_access(self, access_context: AccessContext, 
                           phi_fields: List[str], success: bool = True):
        """Log PHI access for HIPAA compliance."""
        try:
            async with get_postgres_session() as session:
                audit_entry = AuditLog(
                    action_type=access_context.access_type.value,
                    action_description=f"PHI access - {access_context.resource_type}",
                    user_id=access_context.user_id,
                    user_name="", # Will be populated from user lookup
                    user_role=access_context.user_role,
                    ip_address=access_context.ip_address,
                    resource_type=access_context.resource_type,
                    resource_id=access_context.resource_id,
                    accessed_phi=True,
                    phi_fields_accessed=phi_fields,
                    business_justification=access_context.business_justification,
                    request_id=access_context.request_id,
                    session_id=access_context.session_id,
                    additional_context={
                        "access_successful": success,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
                
                session.add(audit_entry)
                await session.commit()
                
                logger.info("PHI access logged",
                          user_id=access_context.user_id,
                          resource_type=access_context.resource_type,
                          phi_fields=len(phi_fields),
                          success=success)
                
        except Exception as e:
            logger.error("Failed to log PHI access", error=str(e))
            # In production, this should trigger alerts

    async def log_system_event(self, event_type: str, description: str, 
                             user_context: Optional[AccessContext] = None,
                             additional_data: Optional[Dict] = None):
        """Log system events for security monitoring."""
        try:
            async with get_postgres_session() as session:
                audit_entry = AuditLog(
                    action_type=event_type,
                    action_description=description,
                    user_id=user_context.user_id if user_context else "system",
                    user_name="",
                    user_role=user_context.user_role if user_context else "system",
                    ip_address=user_context.ip_address if user_context else "127.0.0.1",
                    resource_type="system",
                    accessed_phi=False,
                    request_id=user_context.request_id if user_context else str(uuid.uuid4()),
                    additional_context=additional_data or {}
                )
                
                session.add(audit_entry)
                await session.commit()
                
        except Exception as e:
            logger.error("Failed to log system event", error=str(e))


class SecurityManager:
    """Comprehensive security manager for the claims processing system."""
    
    def __init__(self):
        """Initialize security manager."""
        self.encryption = HIPAAEncryption()
        self.jwt_manager = JWTManager()
        self.audit_logger = AuditLogger()
        self.active_sessions: Dict[str, Dict] = {}
        self.failed_login_attempts: Dict[str, List[datetime]] = {}
        self.max_failed_attempts = 5
        self.lockout_duration = timedelta(minutes=30)

    async def authenticate_user(self, username: str, password: str, 
                              ip_address: str) -> Optional[Dict[str, Any]]:
        """Authenticate user with security controls."""
        # Check for account lockout
        if self._is_account_locked(username):
            await self.audit_logger.log_system_event(
                "login_attempt_blocked",
                f"Login attempt blocked for locked account: {username}",
                additional_data={"ip_address": ip_address}
            )
            return None
        
        try:
            # Verify credentials (implement actual user lookup)
            user_data = await self._verify_user_credentials(username, password)
            
            if user_data:
                # Create session
                session_id = str(uuid.uuid4())
                access_token = self.jwt_manager.create_access_token({
                    "sub": user_data["user_id"],
                    "username": username,
                    "role": user_data["role"],
                    "session_id": session_id
                })
                
                refresh_token = self.jwt_manager.create_refresh_token({
                    "sub": user_data["user_id"],
                    "session_id": session_id
                })
                
                # Store active session
                self.active_sessions[session_id] = {
                    "user_id": user_data["user_id"],
                    "username": username,
                    "role": user_data["role"],
                    "ip_address": ip_address,
                    "login_time": datetime.utcnow(),
                    "last_activity": datetime.utcnow()
                }
                
                # Clear failed attempts
                self.failed_login_attempts.pop(username, None)
                
                # Log successful authentication
                await self.audit_logger.log_system_event(
                    "login_success",
                    f"User {username} logged in successfully",
                    additional_data={
                        "ip_address": ip_address,
                        "session_id": session_id
                    }
                )
                
                return {
                    "access_token": access_token,
                    "refresh_token": refresh_token,
                    "token_type": "bearer",
                    "user_data": user_data,
                    "session_id": session_id
                }
            else:
                # Record failed attempt
                self._record_failed_attempt(username)
                
                await self.audit_logger.log_system_event(
                    "login_failed",
                    f"Failed login attempt for user: {username}",
                    additional_data={"ip_address": ip_address}
                )
                
                return None
                
        except Exception as e:
            logger.error("Authentication error", username=username, error=str(e))
            return None

    def _is_account_locked(self, username: str) -> bool:
        """Check if account is locked due to failed attempts."""
        if username not in self.failed_login_attempts:
            return False
        
        recent_attempts = [
            attempt for attempt in self.failed_login_attempts[username]
            if datetime.utcnow() - attempt < self.lockout_duration
        ]
        
        return len(recent_attempts) >= self.max_failed_attempts

    def _record_failed_attempt(self, username: str):
        """Record failed login attempt."""
        if username not in self.failed_login_attempts:
            self.failed_login_attempts[username] = []
        
        self.failed_login_attempts[username].append(datetime.utcnow())
        
        # Clean up old attempts
        cutoff = datetime.utcnow() - self.lockout_duration
        self.failed_login_attempts[username] = [
            attempt for attempt in self.failed_login_attempts[username]
            if attempt > cutoff
        ]

    async def _verify_user_credentials(self, username: str, password: str) -> Optional[Dict]:
        """Verify user credentials against database."""
        # This would typically query your user database
        # For now, return a mock user for demonstration
        if username == "admin" and password == "admin123":
            return {
                "user_id": "admin-001",
                "username": username,
                "role": "admin",
                "email": "admin@example.com",
                "permissions": ["read", "write", "delete", "phi_access"]
            }
        return None

    async def authorize_action(self, token: str, required_permission: str,
                             resource_type: str, resource_id: Optional[str] = None) -> bool:
        """Authorize user action based on token and permissions."""
        payload = self.jwt_manager.verify_token(token)
        
        if not payload:
            return False
        
        # Check session validity
        session_id = payload.get("session_id")
        if session_id not in self.active_sessions:
            return False
        
        # Update last activity
        self.active_sessions[session_id]["last_activity"] = datetime.utcnow()
        
        # Check permissions based on role
        user_role = payload.get("role")
        return self._check_role_permissions(user_role, required_permission, resource_type)

    def _check_role_permissions(self, role: str, permission: str, resource_type: str) -> bool:
        """Check if role has required permission for resource type."""
        role_permissions = {
            "admin": ["read", "write", "delete", "phi_access", "export"],
            "claims_processor": ["read", "write", "phi_access"],
            "medical_reviewer": ["read", "phi_access"],
            "auditor": ["read", "phi_access", "export"],
            "analyst": ["read"],
            "viewer": ["read"]
        }
        
        allowed_permissions = role_permissions.get(role, [])
        return permission in allowed_permissions

    async def get_access_context(self, token: str, ip_address: str, 
                               request_id: str) -> Optional[AccessContext]:
        """Get access context from token."""
        payload = self.jwt_manager.verify_token(token)
        
        if not payload:
            return None
        
        session_id = payload.get("session_id")
        if session_id not in self.active_sessions:
            return None
        
        session_data = self.active_sessions[session_id]
        
        return AccessContext(
            user_id=payload.get("sub"),
            user_role=payload.get("role"),
            ip_address=ip_address,
            session_id=session_id,
            request_id=request_id,
            access_type=AccessType.READ  # Default, should be set by caller
        )

    async def cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        cutoff = datetime.utcnow() - timedelta(hours=8)  # 8-hour session timeout
        
        expired_sessions = [
            session_id for session_id, data in self.active_sessions.items()
            if data["last_activity"] < cutoff
        ]
        
        for session_id in expired_sessions:
            session_data = self.active_sessions.pop(session_id)
            await self.audit_logger.log_system_event(
                "session_expired",
                f"Session expired for user {session_data['username']}",
                additional_data={"session_id": session_id}
            )


# Global security manager instance
security_manager = SecurityManager()