"""Role-based access control (RBAC) system for claims processing."""

from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass
from enum import Enum
import structlog
from functools import wraps
from fastapi import HTTPException, status

logger = structlog.get_logger(__name__)


class Permission(str, Enum):
    """System permissions."""
    # Basic permissions
    READ_CLAIMS = "claims:read"
    WRITE_CLAIMS = "claims:write"
    DELETE_CLAIMS = "claims:delete"
    
    # PHI access permissions
    VIEW_PHI = "phi:view"
    EXPORT_PHI = "phi:export"
    PRINT_PHI = "phi:print"
    
    # Administrative permissions
    MANAGE_USERS = "users:manage"
    MANAGE_ROLES = "roles:manage"
    VIEW_AUDIT_LOGS = "audit:view"
    EXPORT_AUDIT_LOGS = "audit:export"
    
    # System permissions
    SYSTEM_CONFIG = "system:config"
    SYSTEM_MONITOR = "system:monitor"
    
    # Batch processing permissions
    SUBMIT_BATCH = "batch:submit"
    APPROVE_BATCH = "batch:approve"
    CANCEL_BATCH = "batch:cancel"
    
    # Reporting permissions
    VIEW_REPORTS = "reports:view"
    CREATE_REPORTS = "reports:create"
    EXPORT_REPORTS = "reports:export"
    
    # Failed claims management
    VIEW_FAILED_CLAIMS = "failed_claims:view"
    RESOLVE_FAILED_CLAIMS = "failed_claims:resolve"
    ASSIGN_FAILED_CLAIMS = "failed_claims:assign"


class Role(str, Enum):
    """System roles."""
    SUPER_ADMIN = "super_admin"
    ADMIN = "admin"
    CLAIMS_MANAGER = "claims_manager"
    CLAIMS_PROCESSOR = "claims_processor"
    MEDICAL_REVIEWER = "medical_reviewer"
    COMPLIANCE_OFFICER = "compliance_officer"
    AUDITOR = "auditor"
    ANALYST = "analyst"
    VIEWER = "viewer"
    BATCH_PROCESSOR = "batch_processor"


@dataclass
class RoleDefinition:
    """Definition of a role with its permissions."""
    name: str
    description: str
    permissions: Set[Permission]
    can_access_phi: bool = False
    requires_business_justification: bool = False
    session_timeout_minutes: int = 30
    max_concurrent_sessions: int = 1


class RoleBasedAccessControl:
    """Role-based access control system."""
    
    def __init__(self):
        """Initialize RBAC system with predefined roles."""
        self.role_definitions = self._initialize_roles()
        self.user_roles: Dict[str, Set[Role]] = {}  # user_id -> roles
        self.custom_permissions: Dict[str, Set[Permission]] = {}  # user_id -> additional permissions
        
        logger.info("RBAC system initialized", 
                   roles=len(self.role_definitions),
                   permissions=len(Permission))
    
    def _initialize_roles(self) -> Dict[Role, RoleDefinition]:
        """Initialize predefined role definitions."""
        roles = {}
        
        # Super Admin - Full system access
        roles[Role.SUPER_ADMIN] = RoleDefinition(
            name="Super Administrator",
            description="Full system access including user management",
            permissions={
                Permission.READ_CLAIMS, Permission.WRITE_CLAIMS, Permission.DELETE_CLAIMS,
                Permission.VIEW_PHI, Permission.EXPORT_PHI, Permission.PRINT_PHI,
                Permission.MANAGE_USERS, Permission.MANAGE_ROLES,
                Permission.VIEW_AUDIT_LOGS, Permission.EXPORT_AUDIT_LOGS,
                Permission.SYSTEM_CONFIG, Permission.SYSTEM_MONITOR,
                Permission.SUBMIT_BATCH, Permission.APPROVE_BATCH, Permission.CANCEL_BATCH,
                Permission.VIEW_REPORTS, Permission.CREATE_REPORTS, Permission.EXPORT_REPORTS,
                Permission.VIEW_FAILED_CLAIMS, Permission.RESOLVE_FAILED_CLAIMS, Permission.ASSIGN_FAILED_CLAIMS
            },
            can_access_phi=True,
            session_timeout_minutes=60,
            max_concurrent_sessions=3
        )
        
        # Admin - System administration without user management
        roles[Role.ADMIN] = RoleDefinition(
            name="Administrator",
            description="System administration and configuration",
            permissions={
                Permission.READ_CLAIMS, Permission.WRITE_CLAIMS,
                Permission.VIEW_PHI, Permission.EXPORT_PHI,
                Permission.VIEW_AUDIT_LOGS,
                Permission.SYSTEM_MONITOR,
                Permission.SUBMIT_BATCH, Permission.APPROVE_BATCH, Permission.CANCEL_BATCH,
                Permission.VIEW_REPORTS, Permission.CREATE_REPORTS, Permission.EXPORT_REPORTS,
                Permission.VIEW_FAILED_CLAIMS, Permission.RESOLVE_FAILED_CLAIMS, Permission.ASSIGN_FAILED_CLAIMS
            },
            can_access_phi=True,
            session_timeout_minutes=45
        )
        
        # Claims Manager - Manage claims processing workflow
        roles[Role.CLAIMS_MANAGER] = RoleDefinition(
            name="Claims Manager",
            description="Manage claims processing operations and team",
            permissions={
                Permission.READ_CLAIMS, Permission.WRITE_CLAIMS,
                Permission.VIEW_PHI, Permission.EXPORT_PHI,
                Permission.SUBMIT_BATCH, Permission.APPROVE_BATCH,
                Permission.VIEW_REPORTS, Permission.CREATE_REPORTS,
                Permission.VIEW_FAILED_CLAIMS, Permission.RESOLVE_FAILED_CLAIMS, Permission.ASSIGN_FAILED_CLAIMS
            },
            can_access_phi=True,
            session_timeout_minutes=45
        )
        
        # Claims Processor - Process individual claims
        roles[Role.CLAIMS_PROCESSOR] = RoleDefinition(
            name="Claims Processor",
            description="Process and validate individual claims",
            permissions={
                Permission.READ_CLAIMS, Permission.WRITE_CLAIMS,
                Permission.VIEW_PHI,
                Permission.SUBMIT_BATCH,
                Permission.VIEW_FAILED_CLAIMS, Permission.RESOLVE_FAILED_CLAIMS
            },
            can_access_phi=True,
            requires_business_justification=True,
            session_timeout_minutes=30
        )
        
        # Medical Reviewer - Review clinical aspects
        roles[Role.MEDICAL_REVIEWER] = RoleDefinition(
            name="Medical Reviewer",
            description="Review medical coding and clinical accuracy",
            permissions={
                Permission.READ_CLAIMS,
                Permission.VIEW_PHI,
                Permission.VIEW_FAILED_CLAIMS, Permission.RESOLVE_FAILED_CLAIMS
            },
            can_access_phi=True,
            requires_business_justification=True,
            session_timeout_minutes=30
        )
        
        # Compliance Officer - Ensure regulatory compliance
        roles[Role.COMPLIANCE_OFFICER] = RoleDefinition(
            name="Compliance Officer",
            description="Monitor compliance and audit system usage",
            permissions={
                Permission.READ_CLAIMS,
                Permission.VIEW_PHI, Permission.EXPORT_PHI,
                Permission.VIEW_AUDIT_LOGS, Permission.EXPORT_AUDIT_LOGS,
                Permission.VIEW_REPORTS, Permission.CREATE_REPORTS, Permission.EXPORT_REPORTS,
                Permission.VIEW_FAILED_CLAIMS
            },
            can_access_phi=True,
            requires_business_justification=True,
            session_timeout_minutes=60
        )
        
        # Auditor - Audit system and data access
        roles[Role.AUDITOR] = RoleDefinition(
            name="Auditor",
            description="Audit system access and data integrity",
            permissions={
                Permission.READ_CLAIMS,
                Permission.VIEW_PHI, Permission.EXPORT_PHI,
                Permission.VIEW_AUDIT_LOGS, Permission.EXPORT_AUDIT_LOGS,
                Permission.VIEW_REPORTS, Permission.EXPORT_REPORTS,
                Permission.VIEW_FAILED_CLAIMS
            },
            can_access_phi=True,
            requires_business_justification=True,
            session_timeout_minutes=45
        )
        
        # Analyst - Data analysis and reporting
        roles[Role.ANALYST] = RoleDefinition(
            name="Data Analyst",
            description="Analyze claims data and generate reports",
            permissions={
                Permission.READ_CLAIMS,
                Permission.VIEW_REPORTS, Permission.CREATE_REPORTS, Permission.EXPORT_REPORTS,
                Permission.VIEW_FAILED_CLAIMS
            },
            can_access_phi=False,
            session_timeout_minutes=30
        )
        
        # Viewer - Read-only access
        roles[Role.VIEWER] = RoleDefinition(
            name="Viewer",
            description="Read-only access to claims data",
            permissions={
                Permission.READ_CLAIMS,
                Permission.VIEW_REPORTS,
                Permission.VIEW_FAILED_CLAIMS
            },
            can_access_phi=False,
            session_timeout_minutes=30
        )
        
        # Batch Processor - Automated batch processing
        roles[Role.BATCH_PROCESSOR] = RoleDefinition(
            name="Batch Processor",
            description="Automated system for batch processing",
            permissions={
                Permission.READ_CLAIMS, Permission.WRITE_CLAIMS,
                Permission.SUBMIT_BATCH,
                Permission.SYSTEM_MONITOR
            },
            can_access_phi=False,
            session_timeout_minutes=480,  # 8 hours for batch jobs
            max_concurrent_sessions=5
        )
        
        return roles
    
    def assign_role(self, user_id: str, role: Role) -> bool:
        """Assign a role to a user."""
        try:
            if user_id not in self.user_roles:
                self.user_roles[user_id] = set()
            
            self.user_roles[user_id].add(role)
            
            logger.info("Role assigned to user",
                       user_id=user_id, role=role.value)
            return True
            
        except Exception as e:
            logger.error("Failed to assign role",
                        user_id=user_id, role=role.value, error=str(e))
            return False
    
    def revoke_role(self, user_id: str, role: Role) -> bool:
        """Revoke a role from a user."""
        try:
            if user_id in self.user_roles:
                self.user_roles[user_id].discard(role)
                
                logger.info("Role revoked from user",
                           user_id=user_id, role=role.value)
                return True
            
            return False
            
        except Exception as e:
            logger.error("Failed to revoke role",
                        user_id=user_id, role=role.value, error=str(e))
            return False
    
    def grant_permission(self, user_id: str, permission: Permission) -> bool:
        """Grant a specific permission to a user."""
        try:
            if user_id not in self.custom_permissions:
                self.custom_permissions[user_id] = set()
            
            self.custom_permissions[user_id].add(permission)
            
            logger.info("Permission granted to user",
                       user_id=user_id, permission=permission.value)
            return True
            
        except Exception as e:
            logger.error("Failed to grant permission",
                        user_id=user_id, permission=permission.value, error=str(e))
            return False
    
    def has_permission(self, user_id: str, permission: Permission) -> bool:
        """Check if user has a specific permission."""
        # Check custom permissions first
        if user_id in self.custom_permissions:
            if permission in self.custom_permissions[user_id]:
                return True
        
        # Check role-based permissions
        if user_id in self.user_roles:
            for role in self.user_roles[user_id]:
                role_def = self.role_definitions.get(role)
                if role_def and permission in role_def.permissions:
                    return True
        
        return False
    
    def can_access_phi(self, user_id: str) -> bool:
        """Check if user can access PHI data."""
        if user_id not in self.user_roles:
            return False
        
        for role in self.user_roles[user_id]:
            role_def = self.role_definitions.get(role)
            if role_def and role_def.can_access_phi:
                return True
        
        return False
    
    def requires_business_justification(self, user_id: str) -> bool:
        """Check if user requires business justification for PHI access."""
        if user_id not in self.user_roles:
            return True  # Default to requiring justification
        
        for role in self.user_roles[user_id]:
            role_def = self.role_definitions.get(role)
            if role_def and role_def.requires_business_justification:
                return True
        
        return False
    
    def get_user_permissions(self, user_id: str) -> Set[Permission]:
        """Get all permissions for a user."""
        permissions = set()
        
        # Add custom permissions
        if user_id in self.custom_permissions:
            permissions.update(self.custom_permissions[user_id])
        
        # Add role-based permissions
        if user_id in self.user_roles:
            for role in self.user_roles[user_id]:
                role_def = self.role_definitions.get(role)
                if role_def:
                    permissions.update(role_def.permissions)
        
        return permissions
    
    def get_user_roles(self, user_id: str) -> Set[Role]:
        """Get all roles for a user."""
        return self.user_roles.get(user_id, set())
    
    def get_session_timeout(self, user_id: str) -> int:
        """Get session timeout for user in minutes."""
        max_timeout = 30  # Default timeout
        
        if user_id in self.user_roles:
            for role in self.user_roles[user_id]:
                role_def = self.role_definitions.get(role)
                if role_def:
                    max_timeout = max(max_timeout, role_def.session_timeout_minutes)
        
        return max_timeout
    
    def get_max_concurrent_sessions(self, user_id: str) -> int:
        """Get maximum concurrent sessions for user."""
        max_sessions = 1  # Default
        
        if user_id in self.user_roles:
            for role in self.user_roles[user_id]:
                role_def = self.role_definitions.get(role)
                if role_def:
                    max_sessions = max(max_sessions, role_def.max_concurrent_sessions)
        
        return max_sessions


def require_permission(permission: Permission):
    """Decorator to require specific permission for endpoint access."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract user context from request
            # This would typically be done via dependency injection in FastAPI
            user_id = kwargs.get('current_user_id')
            
            if not user_id:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            # Check permission
            rbac = RoleBasedAccessControl()
            if not rbac.has_permission(user_id, permission):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission {permission.value} required"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def require_phi_access():
    """Decorator to require PHI access authorization."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            user_id = kwargs.get('current_user_id')
            business_justification = kwargs.get('business_justification')
            
            if not user_id:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            rbac = RoleBasedAccessControl()
            
            # Check if user can access PHI
            if not rbac.can_access_phi(user_id):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="PHI access not authorized for this user"
                )
            
            # Check if business justification is required
            if rbac.requires_business_justification(user_id) and not business_justification:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Business justification required for PHI access"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


# Global RBAC instance
rbac_system = RoleBasedAccessControl()