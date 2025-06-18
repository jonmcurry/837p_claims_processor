"""Optimized database connection pool manager with warm-up functionality."""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, List, Optional

import asyncpg
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from src.core.config.settings import settings
from src.core.logging import get_logger, log_error

# Get structured logger with file output
logger = get_logger(__name__, "system", structured=True)


class OptimizedPoolManager:
    """High-performance database pool manager with warm-up and monitoring."""
    
    def __init__(self):
        self.postgres_staging_engine: Optional[AsyncEngine] = None
        self.postgres_production_engine: Optional[AsyncEngine] = None
        self.postgres_staging_session_maker: Optional[async_sessionmaker] = None
        self.postgres_production_session_maker: Optional[async_sessionmaker] = None
        self._pool_stats = {
            'postgres_staging': {'active': 0, 'idle': 0, 'total': 0},
            'postgres_production': {'active': 0, 'idle': 0, 'total': 0}
        }
        self._is_initialized = False
        
    async def initialize(self):
        """Initialize optimized connection pools with warm-up."""
        if self._is_initialized:
            return
            
        logger.info("Initializing optimized PostgreSQL database pools...")
        start_time = time.time()
        
        # Create optimized PostgreSQL staging engine (claims_staging)
        self.postgres_staging_engine = create_async_engine(
            settings.postgres_url,
            echo=False,  # Disable echoing for performance
            poolclass=NullPool,
            connect_args={
                "server_settings": {
                    "jit": "off",
                    "statement_timeout": "60000",  # 60 seconds
                    "idle_in_transaction_session_timeout": "30000",  # 30 seconds
                },
                "command_timeout": 60,
                "prepared_statement_cache_size": 100,
            },
        )
        
        # Create optimized PostgreSQL production engine (smart_pro_claims)
        self.postgres_production_engine = create_async_engine(
            settings.postgres_prod_url,
            echo=False,  # Disable echoing for performance
            poolclass=NullPool,
            connect_args={
                "server_settings": {
                    "jit": "off",
                    "statement_timeout": "60000",  # 60 seconds
                    "idle_in_transaction_session_timeout": "30000",  # 30 seconds
                },
                "command_timeout": 120,  # Longer timeout for production operations
                "prepared_statement_cache_size": 100,
            },
        )
        
        # Create session makers
        self.postgres_staging_session_maker = async_sessionmaker(
            self.postgres_staging_engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )
        
        self.postgres_production_session_maker = async_sessionmaker(
            self.postgres_production_engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )
        
        # Warm up connections
        await self._warm_up_pools()
        
        self._is_initialized = True
        initialization_time = time.time() - start_time
        logger.info(f"PostgreSQL database pools initialized in {initialization_time:.2f}s")
        
    async def _warm_up_pools(self):
        """Pre-create connections to warm up the pools."""
        logger.info("Warming up PostgreSQL connection pools...")
        
        # Warm up PostgreSQL staging pool
        staging_warmup_tasks = []
        for i in range(5):  # Create 5 initial connections
            staging_warmup_tasks.append(self._warmup_postgres_staging_connection())
            
        # Warm up PostgreSQL production pool
        production_warmup_tasks = []
        for i in range(3):  # Create 3 initial connections
            production_warmup_tasks.append(self._warmup_postgres_production_connection())
            
        # Execute warmup tasks concurrently
        try:
            await asyncio.gather(*staging_warmup_tasks, *production_warmup_tasks, return_exceptions=True)
            logger.info("PostgreSQL pools warmed up successfully")
        except Exception as e:
            logger.warning(f"Pool warmup had some failures: {e}")
            log_error(__name__, e, {"operation": "pool_warmup"})
            
    async def _warmup_postgres_staging_connection(self):
        """Create and test a PostgreSQL staging connection."""
        try:
            async with self.postgres_staging_engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
        except Exception as e:
            logger.warning(f"PostgreSQL staging warmup connection failed: {e}")
            log_error(__name__, e, {"operation": "postgres_staging_warmup", "database": "claims_staging"})
            
    async def _warmup_postgres_production_connection(self):
        """Create and test a PostgreSQL production connection."""
        try:
            async with self.postgres_production_engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
        except Exception as e:
            logger.warning(f"PostgreSQL production warmup connection failed: {e}")
            log_error(__name__, e, {"operation": "postgres_production_warmup", "database": "smart_pro_claims"})
            
    @asynccontextmanager
    async def get_postgres_staging_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get optimized PostgreSQL staging session with connection pooling."""
        if not self._is_initialized:
            await self.initialize()
            
        async with self.postgres_staging_session_maker() as session:
            try:
                self._pool_stats['postgres_staging']['active'] += 1
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                self._pool_stats['postgres_staging']['active'] -= 1
                await session.close()
                
    @asynccontextmanager  
    async def get_postgres_production_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get optimized PostgreSQL production session with connection pooling."""
        if not self._is_initialized:
            await self.initialize()
            
        async with self.postgres_production_session_maker() as session:
            try:
                self._pool_stats['postgres_production']['active'] += 1
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                self._pool_stats['postgres_production']['active'] -= 1
                await session.close()
    
    # Legacy method names for backward compatibility
    @asynccontextmanager
    async def get_postgres_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get PostgreSQL staging session (legacy compatibility)."""
        async with self.get_postgres_staging_session() as session:
            yield session
                
    async def get_batch_postgres_staging_sessions(self, count: int) -> List[AsyncSession]:
        """Get multiple PostgreSQL staging sessions for batch operations."""
        if not self._is_initialized:
            await self.initialize()
            
        sessions = []
        for _ in range(count):
            session = self.postgres_staging_session_maker()
            sessions.append(session)
            self._pool_stats['postgres_staging']['active'] += 1
            
        return sessions
        
    async def get_batch_postgres_production_sessions(self, count: int) -> List[AsyncSession]:
        """Get multiple PostgreSQL production sessions for batch operations.""" 
        if not self._is_initialized:
            await self.initialize()
            
        sessions = []
        for _ in range(count):
            session = self.postgres_production_session_maker()
            sessions.append(session)
            self._pool_stats['postgres_production']['active'] += 1
            
        return sessions
        
    async def close_batch_sessions(self, sessions: List[AsyncSession], db_type: str):
        """Close multiple sessions and update stats."""
        for session in sessions:
            await session.close()
        self._pool_stats[db_type]['active'] -= len(sessions)
        
    def get_pool_stats(self) -> Dict:
        """Get current pool statistics."""
        stats = self._pool_stats.copy()
        
        if self.postgres_staging_engine:
            pg_staging_pool = self.postgres_staging_engine.pool
            # Handle NullPool which doesn't have size() method
            if hasattr(pg_staging_pool, 'size'):
                stats['postgres_staging']['total'] = pg_staging_pool.size()
                stats['postgres_staging']['checked_out'] = pg_staging_pool.checkedout()
                stats['postgres_staging']['checked_in'] = pg_staging_pool.checkedin()
                stats['postgres_staging']['overflow'] = pg_staging_pool.overflow()
            else:
                # NullPool - no actual pooling
                stats['postgres_staging']['total'] = 1
                stats['postgres_staging']['checked_out'] = 0
                stats['postgres_staging']['checked_in'] = 1
                stats['postgres_staging']['overflow'] = 0
            
        if self.postgres_production_engine:
            pg_production_pool = self.postgres_production_engine.pool  
            # Handle NullPool which doesn't have size() method
            if hasattr(pg_production_pool, 'size'):
                stats['postgres_production']['total'] = pg_production_pool.size()
                stats['postgres_production']['checked_out'] = pg_production_pool.checkedout()
                stats['postgres_production']['checked_in'] = pg_production_pool.checkedin()
                stats['postgres_production']['overflow'] = pg_production_pool.overflow()
            else:
                # NullPool - no actual pooling
                stats['postgres_production']['total'] = 1
                stats['postgres_production']['checked_out'] = 0
                stats['postgres_production']['checked_in'] = 1
                stats['postgres_production']['overflow'] = 0
            
        return stats
        
    async def health_check(self) -> Dict[str, bool]:
        """Perform health checks on all database connections."""
        results = {'postgres_staging': False, 'postgres_production': False}
        
        try:
            async with self.get_postgres_staging_session() as session:
                await session.execute(text("SELECT 1"))
                results['postgres_staging'] = True
        except Exception as e:
            logger.error(f"PostgreSQL staging health check failed: {e}")
            
        try:
            async with self.get_postgres_production_session() as session:
                await session.execute(text("SELECT 1"))
                results['postgres_production'] = True
        except Exception as e:
            logger.error(f"PostgreSQL production health check failed: {e}")
            
        return results
        
    async def close(self):
        """Close all database connections and engines."""
        logger.info("Closing PostgreSQL database pools...")
        
        if self.postgres_staging_engine:
            await self.postgres_staging_engine.dispose()
            
        if self.postgres_production_engine:
            await self.postgres_production_engine.dispose()
            
        self._is_initialized = False
        logger.info("PostgreSQL database pools closed")


# Global pool manager instance
pool_manager = OptimizedPoolManager()


# Convenience functions
@asynccontextmanager
async def get_postgres_session() -> AsyncGenerator[AsyncSession, None]:
    """Get PostgreSQL staging session from optimized pool (legacy compatibility)."""
    async with pool_manager.get_postgres_staging_session() as session:
        yield session


@asynccontextmanager
async def get_postgres_staging_session() -> AsyncGenerator[AsyncSession, None]:
    """Get PostgreSQL staging session from optimized pool."""
    async with pool_manager.get_postgres_staging_session() as session:
        yield session


@asynccontextmanager
async def get_postgres_production_session() -> AsyncGenerator[AsyncSession, None]:
    """Get PostgreSQL production session from optimized pool."""
    async with pool_manager.get_postgres_production_session() as session:
        yield session


async def initialize_pools():
    """Initialize database pools."""
    await pool_manager.initialize()


async def close_pools():
    """Close database pools."""
    await pool_manager.close()