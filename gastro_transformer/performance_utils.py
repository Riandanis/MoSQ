"""
Performance monitoring utilities for Gastro-Transformer training.

Helps identify bottlenecks and optimize GPU utilization.
"""

import time
import torch
import logging
from contextlib import contextmanager
from typing import Dict, Optional
import psutil
import multiprocessing as mp

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """
    Monitor training performance to identify bottlenecks.

    Tracks:
    - GPU utilization and memory
    - Data loading time
    - Forward/backward pass time
    - Overall throughput
    """

    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.metrics = {
            'data_load_times': [],
            'forward_times': [],
            'backward_times': [],
            'step_times': [],
            'gpu_memory': [],
        }
        self.cpu_count = mp.cpu_count()
        self.total_memory = psutil.virtual_memory().total / (1024**3)  # GB

    def log_system_info(self):
        """Log system information."""
        logger.info("=" * 60)
        logger.info("SYSTEM INFORMATION")
        logger.info("=" * 60)
        logger.info(f"CPU cores: {self.cpu_count}")
        logger.info(f"Total RAM: {self.total_memory:.1f} GB")

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            logger.info(f"GPU count: {gpu_count}")
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                logger.info(f"  GPU {i}: {props.name}")
                logger.info(f"    Memory: {props.total_memory / (1024**3):.1f} GB")
                logger.info(f"    Compute capability: {props.major}.{props.minor}")
        else:
            logger.info("CUDA not available - using CPU")

        logger.info("=" * 60)

    def get_gpu_utilization(self) -> Dict[str, float]:
        """Get current GPU utilization."""
        if not torch.cuda.is_available():
            return {}

        utils = {}
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
            reserved = torch.cuda.memory_reserved(i) / (1024**3)  # GB
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                info = pynvml.nvmlDeviceGetUtilizationRates(handle)
                utils[f'gpu_{i}_util'] = info.gpu
                utils[f'gpu_{i}_mem_util'] = info.memory
            except ImportError:
                pass
            utils[f'gpu_{i}_allocated_gb'] = allocated
            utils[f'gpu_{i}_reserved_gb'] = reserved

        return utils

    def log_gpu_status(self, step: Optional[int] = None):
        """Log current GPU status."""
        if not torch.cuda.is_available():
            return

        prefix = f"Step {step}: " if step is not None else ""
        utils = self.get_gpu_utilization()

        for i in range(torch.cuda.device_count()):
            allocated = utils.get(f'gpu_{i}_allocated_gb', 0)
            reserved = utils.get(f'gpu_{i}_reserved_gb', 0)
            util = utils.get(f'gpu_{i}_util', 0)
            mem_util = utils.get(f'gpu_{i}_mem_util', 0)

            logger.info(
                f"{prefix}GPU {i}: "
                f"{allocated:.1f}GB allocated / {reserved:.1f}GB reserved | "
                f"Util: {util}% | Mem: {mem_util}%"
            )

    def analyze_bottlenecks(self) -> Dict[str, str]:
        """Analyze collected metrics to identify bottlenecks."""
        analysis = {}

        # Check if data loading is the bottleneck
        if self.metrics['data_load_times']:
            avg_data_time = sum(self.metrics['data_load_times']) / len(self.metrics['data_load_times'])
            avg_step_time = sum(self.metrics['step_times']) / len(self.metrics['step_times'])
            data_ratio = avg_data_time / avg_step_time

            if data_ratio > 0.3:
                analysis['data_loading'] = (
                    f"POTENTIAL BOTTLENECK: Data loading takes {data_ratio*100:.1f}% of step time. "
                    f"Consider increasing num_workers or prefetch_factor."
                )
            else:
                analysis['data_loading'] = f"OK: Data loading is {data_ratio*100:.1f}% of step time."

        # Check GPU utilization
        if torch.cuda.is_available():
            utils = self.get_gpu_utilization()
            for i in range(torch.cuda.device_count()):
                util = utils.get(f'gpu_{i}_util', 0)
                allocated = utils.get(f'gpu_{i}_allocated_gb', 0)

                # Try to get total memory
                try:
                    total_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    mem_ratio = allocated / total_mem

                    if util < 70:
                        analysis[f'gpu_{i}_utilization'] = (
                            f"LOW: GPU {i} utilization is only {util}%. "
                            f"Increase batch_size or check for bottlenecks."
                        )
                    elif mem_ratio < 0.5:
                        analysis[f'gpu_{i}_memory'] = (
                            f"LOW: GPU {i} memory usage is {mem_ratio*100:.1f}%. "
                            f"Increase batch_size to better utilize GPU."
                        )
                    else:
                        analysis[f'gpu_{i}'] = f"OK: Good utilization ({util}%, {mem_ratio*100:.1f}% memory)"
                except:
                    pass

        return analysis

    def log_analysis(self):
        """Log bottleneck analysis."""
        logger.info("=" * 60)
        logger.info("PERFORMANCE ANALYSIS")
        logger.info("=" * 60)

        analysis = self.analyze_bottlenecks()
        for key, value in analysis.items():
            logger.info(f"{key}: {value}")

        logger.info("=" * 60)

    @contextmanager
    def measure_time(self, metric_name: str):
        """Context manager to measure time."""
        start = time.time()
        yield
        elapsed = time.time() - start
        self.metrics[metric_name].append(elapsed)


def get_optimal_config(
    gpu_memory_gb: float,
    cpu_cores: int = None,
    data_size_mb: float = 1000
) -> Dict[str, any]:
    """
    Get optimal DataLoader configuration based on system specs.

    Args:
        gpu_memory_gb: GPU memory in GB
        cpu_cores: Number of CPU cores (auto-detect if None)
        data_size_mb: Approximate size of dataset in MB

    Returns:
        Dictionary with recommended config values
    """
    if cpu_cores is None:
        cpu_cores = mp.cpu_count()

    # Batch size scaling (rough estimate: ~2GB per 32 samples)
    recommended_batch_size = min(256, int(32 * gpu_memory_gb / 8))

    # Worker count (don't exceed CPU cores)
    recommended_workers = min(16, cpu_cores - 2, max(4, cpu_cores // 2))

    # Prefetch factor (more workers = more prefetch)
    recommended_prefetch = 4 if recommended_workers >= 8 else 2

    return {
        'batch_size': recommended_batch_size,
        'num_workers': recommended_workers,
        'prefetch_factor': recommended_prefetch,
        'persistent_workers': True,
        'pin_memory': True,
    }


def print_recommendations():
    """Print performance optimization recommendations."""
    logger.info("=" * 60)
    logger.info("PERFORMANCE OPTIMIZATION RECOMMENDATIONS")
    logger.info("=" * 60)
    logger.info("")
    logger.info("1. INCREASE BATCH SIZE:")
    logger.info("   - Current: 32 | Recommended: 128-256 (depends on GPU)")
    logger.info("   - Larger batches = better GPU utilization")
    logger.info("")
    logger.info("2. INCREASE num_workers:")
    logger.info("   - Current: 4 | Recommended: 8-16 (depends on CPU cores)")
    logger.info("   - More workers = faster data loading")
    logger.info("   - Rule of thumb: num_workers = CPU cores / 2")
    logger.info("")
    logger.info("3. ENABLE prefetch_factor:")
    logger.info("   - Each worker preloads this many batches")
    logger.info("   - Recommended: 4 (with 8+ workers) or 2 (with fewer workers)")
    logger.info("")
    logger.info("4. ENABLE persistent_workers:")
    logger.info("   - Workers stay alive between epochs")
    logger.info("   - Reduces startup time for each epoch")
    logger.info("")
    logger.info("5. USE GRADIENT ACCUMULATION (if batch_size is too large):")
    logger.info("   - Simulate larger batch sizes without more memory")
    logger.info("   - Set: accum_steps = desired_batch // actual_batch")
    logger.info("")
    logger.info("6. ENABLE MIXED PRECISION (already enabled by default):")
    logger.info("   - Uses FP16 for speed, FP32 for accuracy")
    logger.info("   - Reduces memory by ~40%, speeds up by ~2x")
    logger.info("")
    logger.info("=" * 60)


if __name__ == '__main__':
    # Test system detection
    print_recommendations()

    monitor = PerformanceMonitor()
    monitor.log_system_info()

    # Get recommendations for your system
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        rec = get_optimal_config(gpu_mem)
        logger.info("=" * 60)
        logger.info("RECOMMENDED CONFIG FOR YOUR SYSTEM:")
        logger.info("=" * 60)
        for key, val in rec.items():
            logger.info(f"  {key}: {val}")
        logger.info("=" * 60)
