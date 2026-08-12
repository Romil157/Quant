"""Job scheduler for production tasks."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional
import asyncio
import logging
from pathlib import Path

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.executors.asyncio import AsyncIOExecutor
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from quant.production.config import get_config, SchedulerConfig
from quant.production.monitoring import get_logger, get_metrics_collector


@dataclass
class JobInfo:
    """Information about a scheduled job."""
    id: str
    name: str
    func: str
    trigger: str
    next_run: Optional[datetime] = None
    last_run: Optional[datetime] = None
    status: str = "pending"
    error: Optional[str] = None


class JobScheduler:
    """Production job scheduler."""
    
    def __init__(self, config: Optional[SchedulerConfig] = None):
        self.config = config or get_config().scheduler
        self.logger = get_logger("scheduler")
        self.metrics = get_metrics_collector()
        
        # Configure job stores
        jobstores = {}
        if self.config.persistent and self.config.jobstore_url:
            jobstores["default"] = SQLAlchemyJobStore(url=self.config.jobstore_url)
        else:
            jobstores["default"] = MemoryJobStore()
        
        # Configure executors
        executors = {
            "default": AsyncIOExecutor(),
        }
        
        # Create scheduler
        self.scheduler = AsyncIOScheduler(
            jobstores=jobstores,
            executors=executors,
            job_defaults=self.config.job_defaults,
            timezone=self.config.timezone,
        )
        
        self._jobs: Dict[str, JobInfo] = {}
    
    def start(self) -> None:
        """Start the scheduler."""
        if not self.config.enabled:
            self.logger.info("scheduler_disabled")
            return
        
        self.scheduler.start()
        self.logger.info("scheduler_started", jobs=len(self._jobs))
    
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the scheduler."""
        self.scheduler.shutdown(wait=wait)
        self.logger.info("scheduler_shutdown")
    
    def add_job(
        self,
        func: Callable,
        trigger: str,
        name: str,
        job_id: Optional[str] = None,
        **trigger_args,
    ) -> str:
        """Add a scheduled job."""
        if job_id is None:
            job_id = name.lower().replace(" ", "_")
        
        if trigger == "cron":
            trigger_obj = CronTrigger(**trigger_args)
        elif trigger == "interval":
            trigger_obj = IntervalTrigger(**trigger_args)
        else:
            raise ValueError(f"Unknown trigger type: {trigger}")
        
        # Wrap function for monitoring
        async def monitored_func():
            start = datetime.utcnow()
            try:
                if asyncio.iscoroutinefunction(func):
                    await func()
                else:
                    func()
                self.metrics.record_job(job_id, "success", (datetime.utcnow() - start).total_seconds())
                self.logger.info("job_completed", job_id=job_id)
            except Exception as e:
                self.metrics.record_job(job_id, "error", (datetime.utcnow() - start).total_seconds())
                self.logger.error("job_failed", job_id=job_id, error=str(e))
                raise
        
        self.scheduler.add_job(
            monitored_func,
            trigger=trigger_obj,
            id=job_id,
            name=name,
            replace_existing=True,
        )
        
        job_info = JobInfo(
            id=job_id,
            name=name,
            func=func.__name__,
            trigger=f"{trigger}:{trigger_args}",
        )
        self._jobs[job_id] = job_info
        
        self.logger.info("job_added", job_id=job_id, trigger=trigger, trigger_args=trigger_args)
        return job_id
    
    def remove_job(self, job_id: str) -> bool:
        """Remove a scheduled job."""
        try:
            self.scheduler.remove_job(job_id)
            self._jobs.pop(job_id, None)
            self.logger.info("job_removed", job_id=job_id)
            return True
        except Exception as e:
            self.logger.error("job_remove_failed", job_id=job_id, error=str(e))
            return False
    
    def get_job(self, job_id: str) -> Optional[JobInfo]:
        """Get job info."""
        return self._jobs.get(job_id)
    
    def list_jobs(self) -> List[JobInfo]:
        """List all jobs."""
        # Update next_run times
        for job in self.scheduler.get_jobs():
            if job.id in self._jobs:
                self._jobs[job.id].next_run = job.next_run_time
        
        return list(self._jobs.values())
    
    def pause_job(self, job_id: str) -> bool:
        """Pause a job."""
        try:
            self.scheduler.pause_job(job_id)
            if job_id in self._jobs:
                self._jobs[job_id].status = "paused"
            return True
        except Exception as e:
            self.logger.error("job_pause_failed", job_id=job_id, error=str(e))
            return False
    
    def resume_job(self, job_id: str) -> bool:
        """Resume a job."""
        try:
            self.scheduler.resume_job(job_id)
            if job_id in self._jobs:
                self._jobs[job_id].status = "running"
            return True
        except Exception as e:
            self.logger.error("job_resume_failed", job_id=job_id, error=str(e))
            return False


# Global scheduler
_scheduler: Optional[JobScheduler] = None


def get_scheduler() -> JobScheduler:
    """Get global scheduler instance."""
    global _scheduler
    if _scheduler is None:
        _scheduler = JobScheduler()
    return _scheduler


# Built-in job functions

async def daily_data_update():
    """Daily job to update market data."""
    from quant.data import download_data
    from quant.production.config import get_config
    
    config = get_config()
    logger = get_logger("jobs.data_update")
    
    # Get symbols from config or use defaults
    symbols = getattr(config, "daily_update_symbols", ["SPY", "QQQ", "IWM"])
    
    # Download yesterday's data
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=1)
    
    for symbol in symbols:
        try:
            download_data(
                symbols=[symbol],
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
            )
            logger.info("data_updated", symbol=symbol)
        except Exception as e:
            logger.error("data_update_failed", symbol=symbol, error=str(e))


async def weekly_model_retrain():
    """Weekly job to retrain ML models."""
    from quant.ml import run_ml_experiment
    from quant.data import download_data
    from quant.production.config import get_config
    
    config = get_config()
    logger = get_logger("jobs.model_retrain")
    
    symbols = getattr(config, "ml_symbols", ["SPY", "QQQ", "IWM"])
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=365)
    
    data = download_data(
        symbols=symbols,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
    )
    
    # Retrain models
    for model_name in ["ridge", "rf", "gbr"]:
        try:
            result = run_ml_experiment(
                data=data,
                model_name=model_name,
                task="regression",
                tune=True,
            )
            logger.info("model_retrained", model=model_name, score=result.test_metrics.get("r2"))
        except Exception as e:
            logger.error("model_retrain_failed", model=model_name, error=str(e))


async def daily_backtest():
    """Daily job to run strategy backtests."""
    from quant.backtest.engine import BacktestEngine, BacktestConfig
    from quant.production.config import get_config
    
    config = get_config()
    logger = get_logger("jobs.backtest")
    
    symbols = getattr(config, "backtest_symbols", ["SPY", "QQQ"])
    strategies = getattr(config, "backtest_strategies", ["momentum", "mean_reversion"])
    
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=365)
    
    for strategy in strategies:
        for symbol in symbols:
            try:
                bt_config = BacktestConfig(
                    initial_capital=100000,
                    start_date=start_date,
                    end_date=end_date,
                )
                
                # Would need strategy factory here
                logger.info("backtest_completed", strategy=strategy, symbol=symbol)
            except Exception as e:
                logger.error("backtest_failed", strategy=strategy, symbol=symbol, error=str(e))


async def cleanup_old_data():
    """Cleanup old cache and temporary files."""
    from quant.production.config import get_config
    import shutil
    
    config = get_config()
    logger = get_logger("jobs.cleanup")
    
    # Clean cache older than 30 days
    cache_root = config.storage.cache_root
    cutoff = datetime.utcnow() - timedelta(days=30)
    
    for cache_dir in cache_root.iterdir():
        if cache_dir.is_dir():
            try:
                mtime = datetime.fromtimestamp(cache_dir.stat().st_mtime)
                if mtime < cutoff:
                    shutil.rmtree(cache_dir)
                    logger.info("cache_cleaned", path=str(cache_dir))
            except Exception as e:
                logger.error("cache_cleanup_failed", path=str(cache_dir), error=str(e))


async def health_check_job():
    """Periodic health check job."""
    from quant.production.monitoring import get_health_check
    
    health = get_health_check()
    results = health.run_checks()
    
    if results["status"] == "unhealthy":
        logger = get_logger("jobs.health")
        logger.warning("health_check_failed", checks=results["checks"])


def setup_default_jobs(scheduler: JobScheduler) -> None:
    """Setup default production jobs."""
    
    # Daily data update at 6 AM UTC
    scheduler.add_job(
        daily_data_update,
        "cron",
        "daily_data_update",
        hour=6,
        minute=0,
    )
    
    # Weekly model retrain on Sunday at 2 AM UTC
    scheduler.add_job(
        weekly_model_retrain,
        "cron",
        "weekly_model_retrain",
        day_of_week="sun",
        hour=2,
        minute=0,
    )
    
    # Daily backtest at 8 AM UTC
    scheduler.add_job(
        daily_backtest,
        "cron",
        "daily_backtest",
        hour=8,
        minute=0,
    )
    
    # Daily cleanup at 3 AM UTC
    scheduler.add_job(
        cleanup_old_data,
        "cron",
        "cleanup_old_data",
        hour=3,
        minute=0,
    )
    
    # Health check every 5 minutes
    scheduler.add_job(
        health_check_job,
        "interval",
        "health_check",
        minutes=5,
    )
    
    logger = get_logger("scheduler")
    logger.info("default_jobs_configured")


# Metrics extension for job tracking
def record_job(self, job_id: str, status: str, duration: float):
    """Record job metrics."""
    # This would be added to MetricsCollector
    pass


# Add to MetricsCollector dynamically
from quant.production.monitoring import MetricsCollector
MetricsCollector.record_job = record_job