"""Base database configuration and session management."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy import MetaData
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, declarative_base
from sqlalchemy.pool import NullPool, QueuePool

from src.core.config import settings

# Naming convention for constraints
convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

metadata = MetaData(naming_convention=convention)


class Base(DeclarativeBase):
    """Base class for all database models."""

    metadata = metadata


# PostgreSQL engine for staging database
postgres_engine = create_async_engine(
    settings.postgres_url,
    echo=settings.debug,
    pool_size=settings.pg_pool_min,
    max_overflow=settings.pg_pool_max - settings.pg_pool_min,
    pool_timeout=settings.pg_pool_timeout,
    pool_recycle=settings.connection_pool_recycle,
    pool_pre_ping=True,
    connect_args={
        "server_settings": {"jit": "off"},
        "command_timeout": settings.pg_command_timeout,
    },
)

# SQL Server engine for production database
# Note: SQL Server async support requires aioodbc or similar async driver
# For now, we'll create a placeholder or use sync operations when needed
try:
    sqlserver_engine = create_async_engine(
        settings.sqlserver_url,
        echo=settings.debug,
        pool_size=settings.sql_pool_size,
        pool_timeout=settings.sql_pool_timeout,
        pool_recycle=settings.connection_pool_recycle,
        pool_pre_ping=True,
        connect_args={
            "timeout": settings.sql_command_timeout,
        },
    )
except Exception as e:
    # SQL Server async driver not available, create a None placeholder
    import logging
    logging.warning(f"SQL Server async engine creation failed: {e}. Using PostgreSQL only.")
    sqlserver_engine = None

# Session factories
PostgresSessionLocal = async_sessionmaker(
    postgres_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# Create SQL Server session factory only if engine is available
if sqlserver_engine:
    SqlServerSessionLocal = async_sessionmaker(
        sqlserver_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )
else:
    SqlServerSessionLocal = None


@asynccontextmanager
async def get_postgres_session() -> AsyncGenerator[AsyncSession, None]:
    """Get PostgreSQL database session."""
    async with PostgresSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def get_sqlserver_session() -> AsyncGenerator[AsyncSession, None]:
    """Get SQL Server database session."""
    if SqlServerSessionLocal is None:
        raise RuntimeError("SQL Server database is not available. Check configuration and async driver support.")
    
    async with SqlServerSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """Initialize database connections."""
    # Test connections
    async with postgres_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with sqlserver_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    """Close database connections."""
    await postgres_engine.dispose()
    await sqlserver_engine.dispose()