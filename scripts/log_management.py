#!/usr/bin/env python3
"""
Log management utilities for the 837P Claims Processor.

Provides functionality for:
- Log rotation
- Log cleanup
- Log archiving
- Log monitoring
"""

import os
import sys
import gzip
import shutil
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.logging import get_logger, logger_config

logger = get_logger(__name__, "system")


class LogManager:
    """Manages log files, rotation, and cleanup."""
    
    def __init__(self, log_base_dir: str = "logs"):
        self.log_base_dir = Path(log_base_dir)
        self.archive_dir = self.log_base_dir / "archive"
        self.archive_dir.mkdir(exist_ok=True)
        
    def rotate_logs(self, max_size_mb: int = 100) -> Dict[str, int]:
        """Manually rotate logs that exceed the specified size."""
        rotated_count = 0
        results = {}
        
        for log_type, log_dir in logger_config.log_dirs.items():
            type_rotated = 0
            
            for log_file in log_dir.glob("*.log"):
                if log_file.stat().st_size > max_size_mb * 1024 * 1024:
                    try:
                        self._rotate_single_log(log_file)
                        type_rotated += 1
                        rotated_count += 1
                    except Exception as e:
                        logger.error(f"Failed to rotate {log_file}: {e}")
                        
            results[log_type] = type_rotated
            
        logger.info(f"Log rotation completed: {rotated_count} files rotated", results=results)
        return results
    
    def _rotate_single_log(self, log_file: Path):
        """Rotate a single log file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rotated_name = f"{log_file.stem}_{timestamp}.log"
        rotated_path = log_file.parent / rotated_name
        
        # Move current log to rotated name
        shutil.move(str(log_file), str(rotated_path))
        
        # Compress the rotated log
        self._compress_log(rotated_path)
        
        logger.info(f"Rotated log: {log_file} -> {rotated_path}.gz")
    
    def _compress_log(self, log_file: Path):
        """Compress a log file using gzip."""
        compressed_path = log_file.with_suffix(log_file.suffix + '.gz')
        
        with open(log_file, 'rb') as f_in:
            with gzip.open(compressed_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
                
        # Remove original file
        log_file.unlink()
    
    def cleanup_old_logs(self, days_to_keep: int = 30) -> Dict[str, int]:
        """Clean up log files older than specified days."""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        cutoff_timestamp = cutoff_date.timestamp()
        
        cleaned_count = 0
        results = {}
        
        for log_type, log_dir in logger_config.log_dirs.items():
            type_cleaned = 0
            
            # Clean regular log files
            for log_file in log_dir.glob("*.log*"):
                if log_file.stat().st_mtime < cutoff_timestamp:
                    try:
                        log_file.unlink()
                        type_cleaned += 1
                        cleaned_count += 1
                    except Exception as e:
                        logger.error(f"Failed to delete {log_file}: {e}")
                        
            results[log_type] = type_cleaned
            
        logger.info(f"Log cleanup completed: {cleaned_count} files removed", results=results)
        return results
    
    def archive_logs(self, days_to_archive: int = 7) -> Dict[str, int]:
        """Archive logs older than specified days."""
        cutoff_date = datetime.now() - timedelta(days=days_to_archive)
        cutoff_timestamp = cutoff_date.timestamp()
        
        archived_count = 0
        results = {}
        
        for log_type, log_dir in logger_config.log_dirs.items():
            type_archived = 0
            archive_type_dir = self.archive_dir / log_type
            archive_type_dir.mkdir(exist_ok=True)
            
            for log_file in log_dir.glob("*.log*"):
                if log_file.stat().st_mtime < cutoff_timestamp:
                    try:
                        # Create archive path
                        archive_path = archive_type_dir / log_file.name
                        
                        # Compress if not already compressed
                        if not log_file.name.endswith('.gz'):
                            self._compress_log_to_archive(log_file, archive_path.with_suffix('.gz'))
                        else:
                            shutil.move(str(log_file), str(archive_path))
                            
                        type_archived += 1
                        archived_count += 1
                    except Exception as e:
                        logger.error(f"Failed to archive {log_file}: {e}")
                        
            results[log_type] = type_archived
            
        logger.info(f"Log archiving completed: {archived_count} files archived", results=results)
        return results
    
    def _compress_log_to_archive(self, source_file: Path, archive_path: Path):
        """Compress a log file directly to archive."""
        with open(source_file, 'rb') as f_in:
            with gzip.open(archive_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
                
        # Remove original file
        source_file.unlink()
    
    def get_log_statistics(self) -> Dict[str, Dict[str, any]]:
        """Get statistics about log files."""
        stats = {}
        
        for log_type, log_dir in logger_config.log_dirs.items():
            log_files = list(log_dir.glob("*.log*"))
            
            if log_files:
                total_size = sum(f.stat().st_size for f in log_files)
                oldest_file = min(log_files, key=lambda f: f.stat().st_mtime)
                newest_file = max(log_files, key=lambda f: f.stat().st_mtime)
                
                stats[log_type] = {
                    "file_count": len(log_files),
                    "total_size_mb": round(total_size / (1024 * 1024), 2),
                    "oldest_file": {
                        "name": oldest_file.name,
                        "modified": datetime.fromtimestamp(oldest_file.stat().st_mtime).isoformat()
                    },
                    "newest_file": {
                        "name": newest_file.name,
                        "modified": datetime.fromtimestamp(newest_file.stat().st_mtime).isoformat()
                    }
                }
            else:
                stats[log_type] = {
                    "file_count": 0,
                    "total_size_mb": 0,
                    "oldest_file": None,
                    "newest_file": None
                }
                
        return stats
    
    def monitor_log_growth(self, threshold_mb: int = 500) -> List[Dict[str, any]]:
        """Monitor log directories for excessive growth."""
        warnings = []
        
        for log_type, log_dir in logger_config.log_dirs.items():
            log_files = list(log_dir.glob("*.log*"))
            total_size = sum(f.stat().st_size for f in log_files)
            size_mb = total_size / (1024 * 1024)
            
            if size_mb > threshold_mb:
                warnings.append({
                    "log_type": log_type,
                    "size_mb": round(size_mb, 2),
                    "file_count": len(log_files),
                    "threshold_mb": threshold_mb,
                    "recommendation": "Consider log rotation or cleanup"
                })
                
        if warnings:
            logger.warning(f"Log size warnings detected", warnings=warnings)
            
        return warnings
    
    def create_log_report(self) -> Dict[str, any]:
        """Create a comprehensive log report."""
        stats = self.get_log_statistics()
        
        # Calculate totals
        total_files = sum(s["file_count"] for s in stats.values())
        total_size_mb = sum(s["total_size_mb"] for s in stats.values())
        
        # Check for warnings
        warnings = self.monitor_log_growth()
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_log_files": total_files,
                "total_size_mb": round(total_size_mb, 2),
                "log_types": len(stats),
                "warnings_count": len(warnings)
            },
            "by_type": stats,
            "warnings": warnings,
            "recommendations": self._get_recommendations(stats, warnings)
        }
        
        logger.info("Log report generated", report_summary=report["summary"])
        return report
    
    def _get_recommendations(self, stats: Dict, warnings: List) -> List[str]:
        """Generate recommendations based on log analysis."""
        recommendations = []
        
        # Check for large logs
        for log_type, stat in stats.items():
            if stat["total_size_mb"] > 100:
                recommendations.append(f"Consider rotating {log_type} logs (current: {stat['total_size_mb']}MB)")
                
        # Check for old logs
        for log_type, stat in stats.items():
            if stat["oldest_file"]:
                oldest_date = datetime.fromisoformat(stat["oldest_file"]["modified"])
                days_old = (datetime.now() - oldest_date).days
                if days_old > 30:
                    recommendations.append(f"Consider archiving old {log_type} logs (oldest: {days_old} days)")
                    
        # General recommendations
        if not recommendations:
            recommendations.append("Log management is healthy")
            
        return recommendations


def main():
    """Command-line interface for log management."""
    parser = argparse.ArgumentParser(description="Log Management Tool")
    parser.add_argument("--rotate", action="store_true", help="Rotate large log files")
    parser.add_argument("--cleanup", type=int, metavar="DAYS", help="Clean up logs older than DAYS")
    parser.add_argument("--archive", type=int, metavar="DAYS", help="Archive logs older than DAYS")
    parser.add_argument("--stats", action="store_true", help="Show log statistics")
    parser.add_argument("--report", action="store_true", help="Generate comprehensive log report")
    parser.add_argument("--monitor", action="store_true", help="Monitor log growth")
    parser.add_argument("--size-threshold", type=int, default=100, help="Size threshold for rotation (MB)")
    
    args = parser.parse_args()
    
    log_manager = LogManager()
    
    if args.rotate:
        print("Rotating large log files...")
        results = log_manager.rotate_logs(args.size_threshold)
        print(f"Rotation completed: {results}")
        
    if args.cleanup:
        print(f"Cleaning up logs older than {args.cleanup} days...")
        results = log_manager.cleanup_old_logs(args.cleanup)
        print(f"Cleanup completed: {results}")
        
    if args.archive:
        print(f"Archiving logs older than {args.archive} days...")
        results = log_manager.archive_logs(args.archive)
        print(f"Archiving completed: {results}")
        
    if args.stats:
        print("Log Statistics:")
        stats = log_manager.get_log_statistics()
        for log_type, stat in stats.items():
            print(f"\n{log_type.upper()}:")
            print(f"  Files: {stat['file_count']}")
            print(f"  Size: {stat['total_size_mb']} MB")
            if stat['oldest_file']:
                print(f"  Oldest: {stat['oldest_file']['name']} ({stat['oldest_file']['modified']})")
            if stat['newest_file']:
                print(f"  Newest: {stat['newest_file']['name']} ({stat['newest_file']['modified']})")
                
    if args.monitor:
        print("Monitoring log growth...")
        warnings = log_manager.monitor_log_growth()
        if warnings:
            print("WARNINGS:")
            for warning in warnings:
                print(f"  {warning['log_type']}: {warning['size_mb']} MB (threshold: {warning['threshold_mb']} MB)")
        else:
            print("No log growth warnings")
            
    if args.report:
        print("Generating comprehensive log report...")
        report = log_manager.create_log_report()
        
        print(f"\nLOG REPORT - {report['timestamp']}")
        print("=" * 50)
        print(f"Total Files: {report['summary']['total_log_files']}")
        print(f"Total Size: {report['summary']['total_size_mb']} MB")
        print(f"Log Types: {report['summary']['log_types']}")
        print(f"Warnings: {report['summary']['warnings_count']}")
        
        if report['warnings']:
            print("\nWARNINGS:")
            for warning in report['warnings']:
                print(f"  - {warning['log_type']}: {warning['size_mb']} MB")
                
        print("\nRECOMMENDATIONS:")
        for rec in report['recommendations']:
            print(f"  - {rec}")
    
    if not any([args.rotate, args.cleanup, args.archive, args.stats, args.report, args.monitor]):
        parser.print_help()


if __name__ == "__main__":
    main()