"""Optimized database connection pool manager with warm-up functionality."""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, List, Optional

import asyncpg
import pymssql
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import QueuePool

from src.core.config.settings import settings
from src.core.logging import get_logger, log_error

# Get structured logger with file output
logger = get_logger(__name__, "system", structured=True)


class OptimizedPoolManager:
    """High-performance database pool manager with warm-up and monitoring."""
    
    def __init__(self):
        self.postgres_engine: Optional[AsyncEngine] = None
        self.sqlserver_engine: Optional[AsyncEngine] = None
        self.postgres_session_maker: Optional[async_sessionmaker] = None
        self.sqlserver_session_maker: Optional[async_sessionmaker] = None
        self._pool_stats = {
            'postgres': {'active': 0, 'idle': 0, 'total': 0},
            'sqlserver': {'active': 0, 'idle': 0, 'total': 0}
        }
        self._is_initialized = False
        
    async def initialize(self):
        """Initialize optimized connection pools with warm-up."""
        if self._is_initialized:
            return
            
        logger.info("Initializing optimized database pools...")
        start_time = time.time()
        
        # Create optimized PostgreSQL engine
        self.postgres_engine = create_async_engine(
            settings.postgres_url,
            echo=False,  # Disable echoing for performance
            poolclass=QueuePool,
            pool_size=100,  # Increased from 10-50
            max_overflow=50,  # Additional connections on demand
            pool_timeout=10,  # Reduced timeout for faster failures
            pool_recycle=1800,  # Recycle connections every 30 minutes
            pool_pre_ping=True,
            pool_reset_on_return='commit',
            connect_args={
                "server_settings": {
                    "jit": "off",
                    "shared_preload_libraries": "pg_stat_statements",
                    "statement_timeout": "60000",  # 60 seconds
                    "idle_in_transaction_session_timeout": "30000",  # 30 seconds
                },
                "command_timeout": 60,
                "prepared_statement_cache_size": 100,
            },
        )
        
        # Create optimized SQL Server engine  
        self.sqlserver_engine = create_async_engine(
            settings.sqlserver_url,
            echo=False,
            poolclass=QueuePool,
            pool_size=75,  # Increased from 25
            max_overflow=25,
            pool_timeout=10,
            pool_recycle=1800,
            pool_pre_ping=True,
            pool_reset_on_return='commit',
            connect_args={
                "timeout": 60,
                "login_timeout": 30,
                "autocommit": False,
                "ansi": True,
                "as_dict": True,
            },
        )
        
        # Create session makers
        self.postgres_session_maker = async_sessionmaker(
            self.postgres_engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )
        
        self.sqlserver_session_maker = async_sessionmaker(
            self.sqlserver_engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )
        
        # Warm up connections
        await self._warm_up_pools()
        
        self._is_initialized = True
        initialization_time = time.time() - start_time
        logger.info(f"Database pools initialized in {initialization_time:.2f}s")
        
    async def _warm_up_pools(self):
        """Pre-create connections to warm up the pools."""
        logger.info("Warming up database connection pools...")
        
        # Warm up PostgreSQL pool
        pg_warmup_tasks = []
        for i in range(20):  # Create 20 initial connections
            pg_warmup_tasks.append(self._warmup_postgres_connection())
            
        # Warm up SQL Server pool  
        ss_warmup_tasks = []
        for i in range(15):  # Create 15 initial connections
            ss_warmup_tasks.append(self._warmup_sqlserver_connection())
            
        # Execute warmup tasks concurrently
        try:
            await asyncio.gather(*pg_warmup_tasks, *ss_warmup_tasks, return_exceptions=True)
            logger.info("Database pools warmed up successfully")
        except Exception as e:
            logger.warning(f"Pool warmup had some failures: {e}")
            log_error(__name__, e, {"operation": "pool_warmup"})
            
    async def _warmup_postgres_connection(self):
        """Create and test a PostgreSQL connection."""
        try:
            async with self.postgres_engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
        except Exception as e:
            logger.warning(f"PostgreSQL warmup connection failed: {e}")
            log_error(__name__, e, {"operation": "postgres_warmup", "database": "postgresql"})
            
    async def _warmup_sqlserver_connection(self):
        """Create and test a SQL Server connection."""
        try:
            async with self.sqlserver_engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
        except Exception as e:
            logger.warning(f"SQL Server warmup connection failed: {e}")
            log_error(__name__, e, {"operation": "sqlserver_warmup", "database": "sqlserver"})
            
    @asynccontextmanager
    async def get_postgres_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get optimized PostgreSQL session with connection pooling."""
        if not self._is_initialized:
            await self.initialize()
            
        async with self.postgres_session_maker() as session:
            try:
                self._pool_stats['postgres']['active'] += 1
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                self._pool_stats['postgres']['active'] -= 1
                await session.close()
                
    @asynccontextmanager  
    async def get_sqlserver_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get optimized SQL Server session with connection pooling."""
        if not self._is_initialized:
            await self.initialize()
            
        async with self.sqlserver_session_maker() as session:
            try:
                self._pool_stats['sqlserver']['active'] += 1
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                self._pool_stats['sqlserver']['active'] -= 1
                await session.close()
                
    async def get_batch_postgres_sessions(self, count: int) -> List[AsyncSession]:
        """Get multiple PostgreSQL sessions for batch operations."""
        if not self._is_initialized:
            await self.initialize()
            
        sessions = []
        for _ in range(count):
            session = self.postgres_session_maker()
            sessions.append(session)
            self._pool_stats['postgres']['active'] += 1
            
        return sessions
        
    async def get_batch_sqlserver_sessions(self, count: int) -> List[AsyncSession]:
        """Get multiple SQL Server sessions for batch operations.""" 
        if not self._is_initialized:
            await self.initialize()
            
        sessions = []
        for _ in range(count):
            session = self.sqlserver_session_maker()
            sessions.append(session)
            self._pool_stats['sqlserver']['active'] += 1
            
        return sessions
        
    async def close_batch_sessions(self, sessions: List[AsyncSession], db_type: str):
        """Close multiple sessions and update stats."""
        for session in sessions:
            await session.close()
        self._pool_stats[db_type]['active'] -= len(sessions)
        
    def get_pool_stats(self) -> Dict:
        """Get current pool statistics."""
        stats = self._pool_stats.copy()
        
        if self.postgres_engine:
            pg_pool = self.postgres_engine.pool
            stats['postgres']['total'] = pg_pool.size()
            stats['postgres']['checked_out'] = pg_pool.checkedout()
            stats['postgres']['checked_in'] = pg_pool.checkedin()
            stats['postgres']['overflow'] = pg_pool.overflow()
            
        if self.sqlserver_engine:
            ss_pool = self.sqlserver_engine.pool  
            stats['sqlserver']['total'] = ss_pool.size()
            stats['sqlserver']['checked_out'] = ss_pool.checkedout()
            stats['sqlserver']['checked_in'] = ss_pool.checkedin()
            stats['sqlserver']['overflow'] = ss_pool.overflow()
            
        return stats
        
    async def health_check(self) -> Dict[str, bool]:
        """Perform health checks on all database connections."""
        results = {'postgres': False, 'sqlserver': False}
        
        try:
            async with self.get_postgres_session() as session:
                await session.execute(text("SELECT 1"))
                results['postgres'] = True
        except Exception as e:
            logger.error(f"PostgreSQL health check failed: {e}")
            
        try:
            async with self.get_sqlserver_session() as session:
                await session.execute(text("SELECT 1"))
                results['sqlserver'] = True
        except Exception as e:
            logger.error(f"SQL Server health check failed: {e}")
            
        return results
        
    async def close(self):
        """Close all database connections and engines."""
        logger.info("Closing database pools...")
        
        if self.postgres_engine:
            await self.postgres_engine.dispose()
            
        if self.sqlserver_engine:
            await self.sqlserver_engine.dispose()
            
        self._is_initialized = False
        logger.info("Database pools closed")


# Global pool manager instance
pool_manager = OptimizedPoolManager()


# Convenience functions
async def get_postgres_session() -> AsyncGenerator[AsyncSession, None]:
    """Get PostgreSQL session from optimized pool."""
    async with pool_manager.get_postgres_session() as session:
        yield session


async def get_sqlserver_session() -> AsyncGenerator[AsyncSession, None]:
    """Get SQL Server session from optimized pool."""
    async with pool_manager.get_sqlserver_session() as session:
        yield session


async def initialize_pools():
    """Initialize database pools."""
    await pool_manager.initialize()


async def close_pools():
    """Close database pools."""
    await pool_manager.close()