# Redis to Memcached Migration Guide

## Overview

The application has been migrated from Redis/Memurai to Memcached for caching RVU data. This change simplifies the caching infrastructure and provides better Windows compatibility.

## Changes Made

### 1. RVU Cache Implementation
- **File**: `src/core/cache/rvu_cache.py`
- Replaced `redis.asyncio` with `aiomcache` client
- Updated all cache operations to use Memcached API
- Changed from Redis hash operations to Memcached key-value with pickle serialization
- Maintained all existing functionality (preloading, batch operations, local cache)

### 2. Configuration Updates
- **File**: `src/core/config/settings.py`
  - Removed Redis configuration properties
  - Added Memcached configuration:
    - `memcached_host` (default: localhost)
    - `memcached_port` (default: 11211)
    - `memcached_pool_min/max`
    - `memcached_ttl_seconds`

### 3. Environment Variables
- **Files**: `.env`, `.env.example`
  - Replaced `REDIS_*` variables with `MEMCACHED_*`
  - Simplified configuration (no password/db needed)

### 4. Dependencies
- **File**: `requirements.txt`
  - Removed: `redis[hiredis]`, `aiocache[redis]`
  - Added: `aiomcache>=0.8.1`

## Installation

### Windows
1. Download Memcached for Windows from: https://github.com/nono303/memcached
2. Extract and run `memcached.exe`
3. Or install as a service: `memcached.exe -d install`

### Linux/Mac
```bash
# Ubuntu/Debian
sudo apt-get install memcached

# macOS
brew install memcached

# Start service
sudo systemctl start memcached  # Linux
brew services start memcached   # macOS
```

## Configuration

Update your `.env` file:
```env
MEMCACHED_HOST=localhost
MEMCACHED_PORT=11211
MEMCACHED_POOL_MIN=10
MEMCACHED_POOL_MAX=50
MEMCACHED_TTL_SECONDS=3600
```

## Key Differences

### Data Storage
- **Redis**: Used hash sets for structured data
- **Memcached**: Uses key-value pairs with pickled Python objects

### Features
- **Redis**: Advanced data structures, pub/sub, persistence
- **Memcached**: Simple key-value cache, no persistence

### Performance
- Both provide sub-millisecond response times
- Memcached has lower memory overhead
- Local cache layer minimizes remote lookups

## Migration Notes

1. **No data migration needed** - Cache will rebuild automatically
2. **Graceful fallback** - System works without Memcached using local cache
3. **Compatible API** - No changes needed to calling code

## Monitoring

Check Memcached stats:
```bash
echo "stats" | nc localhost 11211
```

Python check:
```python
import aiomcache
client = aiomcache.Client("localhost", 11211)
stats = await client.stats()
```

## Troubleshooting

### Connection Failed
- Verify Memcached is running: `ps aux | grep memcached`
- Check port availability: `netstat -an | grep 11211`
- Test connection: `telnet localhost 11211`

### Cache Misses
- Check logs for "Failed to initialize RVU cache"
- Verify database connection for cache rebuilding
- Monitor cache hit/miss ratio in application logs

### Performance Issues
- Increase `MEMCACHED_POOL_MAX` for more connections
- Monitor Memcached memory usage
- Consider increasing Memcached memory limit (`-m` flag)

## Rollback Plan

To rollback to Redis:
1. Revert the code changes
2. Update `.env` with Redis settings
3. Install Redis/Memurai
4. Update `requirements.txt`

The application will automatically use whichever cache backend is configured.