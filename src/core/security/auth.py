"""Authentication and authorization for API access."""

from datetime import datetime, timedelta
from typing import Dict, List, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from src.core.config import settings
from src.core.database.models import AuditLog


class UserRole(str):
    """User role definitions."""

    ADMIN = "admin"
    CLAIMS_PROCESSOR = "claims_processor"
    AUDITOR = "auditor"
    UI_USER = "ui_user"
    API_USER = "api_user"


class RolePermissions:
    """Role-based permissions mapping."""

    PERMISSIONS = {
        UserRole.ADMIN: [
            "full_access",
            "manage_users",
            "view_audit_logs",
            "configure_system",
        ],
        UserRole.CLAIMS_PROCESSOR: [
            "read_claims",
            "write_processing_status",
            "submit_batches",
            "view_failed_claims",
            "reprocess_claims",
        ],
        UserRole.AUDITOR: [
            "read_only",
            "view_audit_logs",
            "export_reports",
            "view_phi_with_justification",
        ],
        UserRole.UI_USER: [
            "view_dashboards",
            "view_failed_claims",
            "export_claims",
            "view_metrics",
        ],
        UserRole.API_USER: [
            "submit_claims",
            "check_status",
            "retrieve_results",
        ],
    }


class Token(BaseModel):
    """JWT token model."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenData(BaseModel):
    """Token payload data."""

    username: str
    user_id: str
    roles: List[str]
    permissions: List[str]
    exp: datetime
    iat: datetime
    jti: str  # JWT ID for token revocation


class User(BaseModel):
    """User model for authentication."""

    user_id: str
    username: str
    email: str
    full_name: str
    roles: List[str]
    is_active: bool = True
    requires_mfa: bool = True
    last_login: Optional[datetime] = None
    password_changed_at: datetime


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.api_prefix}/auth/token")


class AuthHandler:
    """Handle authentication and authorization."""

    def __init__(self):
        """Initialize auth handler."""
        self.secret_key = settings.jwt_secret_key.get_secret_value()
        self.algorithm = settings.jwt_algorithm
        self.expiration_minutes = settings.jwt_expiration_minutes

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash."""
        return pwd_context.verify(plain_password, hashed_password)

    def get_password_hash(self, password: str) -> str:
        """Hash a password."""
        return pwd_context.hash(password)

    def create_access_token(self, user: User) -> Token:
        """Create JWT access token."""
        # Gather all permissions based on roles
        permissions = set()
        for role in user.roles:
            permissions.update(RolePermissions.PERMISSIONS.get(role, []))

        # Token payload
        now = datetime.utcnow()
        expires = now + timedelta(minutes=self.expiration_minutes)
        
        payload = {
            "sub": user.username,
            "user_id": user.user_id,
            "roles": user.roles,
            "permissions": list(permissions),
            "exp": expires,
            "iat": now,
            "jti": self._generate_jti(),
        }

        # Create token
        access_token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        
        return Token(
            access_token=access_token,
            expires_in=self.expiration_minutes * 60,
        )

    def decode_token(self, token: str) -> TokenData:
        """Decode and validate JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            return TokenData(
                username=payload["sub"],
                user_id=payload["user_id"],
                roles=payload["roles"],
                permissions=payload["permissions"],
                exp=datetime.fromtimestamp(payload["exp"]),
                iat=datetime.fromtimestamp(payload["iat"]),
                jti=payload["jti"],
            )
            
        except JWTError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

    def _generate_jti(self) -> str:
        """Generate unique JWT ID."""
        import uuid
        return str(uuid.uuid4())

    async def get_current_user(self, token: str = Depends(oauth2_scheme)) -> TokenData:
        """Get current authenticated user from token."""
        return self.decode_token(token)

    def require_permission(self, permission: str):
        """Decorator to require specific permission."""
        async def permission_checker(
            current_user: TokenData = Depends(self.get_current_user),
        ):
            if permission not in current_user.permissions:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission '{permission}' required",
                )
            return current_user

        return permission_checker

    def require_roles(self, roles: List[str]):
        """Decorator to require specific roles."""
        async def role_checker(
            current_user: TokenData = Depends(self.get_current_user),
        ):
            if not any(role in current_user.roles for role in roles):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"One of roles {roles} required",
                )
            return current_user

        return role_checker


class SessionManager:
    """Manage user sessions and token revocation."""

    def __init__(self):
        """Initialize session manager."""
        self.revoked_tokens = set()  # In production, use Redis
        self.active_sessions = {}  # Track active sessions

    async def create_session(self, user: User, ip_address: str, user_agent: str) -> str:
        """Create new user session."""
        session_id = self._generate_session_id()
        
        self.active_sessions[session_id] = {
            "user_id": user.user_id,
            "username": user.username,
            "ip_address": ip_address,
            "user_agent": user_agent,
            "created_at": datetime.utcnow(),
            "last_activity": datetime.utcnow(),
        }
        
        return session_id

    async def validate_session(self, session_id: str) -> bool:
        """Validate if session is still active."""
        if session_id not in self.active_sessions:
            return False
            
        session = self.active_sessions[session_id]
        
        # Check session timeout
        if datetime.utcnow() - session["last_activity"] > timedelta(minutes=30):
            await self.end_session(session_id)
            return False
            
        # Update last activity
        session["last_activity"] = datetime.utcnow()
        return True

    async def end_session(self, session_id: str) -> None:
        """End user session."""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]

    async def revoke_token(self, jti: str) -> None:
        """Revoke a JWT token."""
        self.revoked_tokens.add(jti)

    async def is_token_revoked(self, jti: str) -> bool:
        """Check if token is revoked."""
        return jti in self.revoked_tokens

    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        import uuid
        return str(uuid.uuid4())


# Global instances
auth_handler = AuthHandler()
session_manager = SessionManager()