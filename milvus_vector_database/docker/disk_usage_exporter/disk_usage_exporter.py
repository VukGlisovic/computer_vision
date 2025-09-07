"""
Custom Prometheus exporter for monitoring folder disk usage.
Provides detailed metrics about folder size, file count, and growth rate.
"""

import os
import time
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from prometheus_client import start_http_server, Gauge, Counter, Histogram
import argparse


class DiskUsageExporter:
    """Prometheus exporter for disk usage metrics."""
    
    def __init__(self, folders_to_monitor: List[str], port: int = 8000):
        """
        Initialize the disk usage exporter.
        
        Args:
            folders_to_monitor:
            port: Port to expose metrics on
        """
        self.folders_to_monitor = folders_to_monitor
        self.port = port
        
        # Prometheus metrics
        self.folder_size_bytes = Gauge(
            name='folder_size_bytes',
            documentation='Size of monitored folders in bytes',
            labelnames=['folder_path']
        )
        
        self.folder_file_count = Gauge(
            name='folder_file_count',
            documentation='Number of files in monitored folders',
            labelnames=['folder_path']
        )
        
        self.folder_subdirectory_count = Gauge(
            name='folder_subdirectory_count',
            documentation='Number of subdirectories in monitored folders',
            labelnames=['folder_path']
        )
        
        self.folder_largest_file_bytes = Gauge(
            name='folder_largest_file_bytes',
            documentation='Size of the largest file in monitored folders',
            labelnames=['folder_path', 'filename']
        )
        
        self.folder_scan_duration_seconds = Histogram(
            name='folder_scan_duration_seconds',
            documentation='Time taken to scan folder for metrics',
            labelnames=['folder_path']
        )
        
        self.folder_scan_errors = Counter(
            name='folder_scan_errors_total',
            documentation='Total number of errors while scanning folders',
            labelnames=['folder_path', 'error_type']
        )

    @staticmethod
    def get_folder_stats(folder_path: str) -> Tuple[int, int, int, Tuple[str, int]]:
        """
        Get comprehensive statistics for a folder.
        
        Returns:
            Tuple of (total_size, file_count, dir_count, (largest_file_name, largest_file_size))
        """
        total_size = 0
        file_count = 0
        dir_count = 0
        largest_file_size = 0
        largest_file_name = ""
        
        try:
            for root, dirs, files in os.walk(folder_path):
                dir_count += len(dirs)
                
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        file_size = os.path.getsize(file_path)
                        total_size += file_size
                        file_count += 1
                        
                        if file_size > largest_file_size:
                            largest_file_size = file_size
                            largest_file_name = os.path.relpath(file_path, folder_path)
                            
                    except (OSError, IOError) as e:
                        logger.warning(f"Could not get size for {file_path}: {e}")
                        
        except (OSError, IOError) as e:
            logger.error(f"Error scanning folder {folder_path}: {e}")
            raise
            
        return total_size, file_count, dir_count, (largest_file_name, largest_file_size)
    
    def update_metrics(self):
        """Update all metrics for monitored folders."""
        
        for folder_path in self.folders_to_monitor:
            if not os.path.exists(folder_path):
                logger.warning(f"Folder {folder_path} does not exist, skipping...")
                self.folder_scan_errors.labels(folder_path=folder_path, error_type="folder_not_found").inc()
                continue
            
            try:
                # Measure scan duration
                with self.folder_scan_duration_seconds.labels(folder_path=folder_path).time():
                    total_size, file_count, dir_count, (largest_file_name, largest_file_size) = self.get_folder_stats(folder_path)
                
                # Update metrics
                self.folder_size_bytes.labels(folder_path=folder_path).set(total_size)
                self.folder_file_count.labels(folder_path=folder_path).set(file_count)
                self.folder_subdirectory_count.labels(folder_path=folder_path).set(dir_count)
                
                if largest_file_name:
                    self.folder_largest_file_bytes.labels(folder_path=folder_path, filename=largest_file_name).set(largest_file_size)
                
                logger.info(f"Updated metrics for {folder_path}: {total_size:,} bytes, {file_count} files, {dir_count} dirs")
                
            except Exception as e:
                logger.error(f"Error updating metrics for {folder_path}: {e}")
                self.folder_scan_errors.labels(folder_path=folder_path, error_type="scan_error").inc()
    
    def run(self, update_interval: int = 15):
        """
        Start the exporter and continuously update metrics.
        
        Args:
            update_interval: Seconds between metric updates
        """
        logger.info(f"Starting disk usage exporter on port {self.port}")
        logger.info(f"Monitoring folders: {self.folders_to_monitor}")
        logger.info(f"Update interval: {update_interval} seconds")
        
        # Start Prometheus metrics server
        start_http_server(self.port)
        
        while True:
            try:
                self.update_metrics()
                time.sleep(update_interval)
            except KeyboardInterrupt:
                logger.info("Shutting down exporter...")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                time.sleep(update_interval)


def main():
    parser = argparse.ArgumentParser(description='Disk Usage Exporter')
    parser.add_argument('--port', type=int, default=8000, help='Port to expose metrics on')
    parser.add_argument('--interval', type=int, default=15, help='Update interval in seconds')
    parser.add_argument('--folders-to-monitor', nargs='*', default=[],
                       help='Configure which folders to monitor')
    
    args = parser.parse_args()

    logger.info(f"Checking the following folders for monitoring: {''.join(args.folders_to_monitor)}")

    # Add additional paths
    folders_to_monitor = []
    for i, path in enumerate(args.folders_to_monitor):
        if os.path.exists(path):
            folders_to_monitor.append(path)
        else:
            logger.warning(f"Folder '{path}' does not exist. It will not be included in the monitoring report.")
    
    if not folders_to_monitor:
        logger.error("No valid folders found to monitor!")
        return 1
    
    # Create and run exporter
    exporter = DiskUsageExporter(folders_to_monitor, args.port)
    exporter.run(args.interval)
    
    return 0


if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    exit(main())


