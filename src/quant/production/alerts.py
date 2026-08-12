"""Alerting system for production monitoring."""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from quant.production.monitoring import get_logger, get_metrics_collector


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status."""
    FIRING = "firing"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"


@dataclass
class Alert:
    """Alert definition."""
    id: str
    name: str
    level: AlertLevel
    component: str
    message: str
    status: AlertStatus = AlertStatus.FIRING
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    resolved_at: datetime | None = None
    acknowledged_by: str | None = None
    acknowledged_at: datetime | None = None
    labels: dict[str, str] = field(default_factory=dict)
    annotations: dict[str, str] = field(default_factory=dict)
    value: float | None = None
    threshold: float | None = None


@dataclass
class AlertRule:
    """Alert rule definition."""
    name: str
    component: str
    query: str  # PromQL-like query
    condition: str  # e.g., "> 0.95"
    level: AlertLevel = AlertLevel.WARNING
    for_duration: timedelta = timedelta(minutes=5)
    labels: dict[str, str] = field(default_factory=dict)
    annotations: dict[str, str] = field(default_factory=dict)
    enabled: bool = True


class AlertManager:
    """Manage alerts and notifications."""

    def __init__(self):
        self.logger = get_logger("alerts")
        self.metrics = get_metrics_collector()

        self.rules: dict[str, AlertRule] = {}
        self.alerts: dict[str, Alert] = {}
        self.handlers: list[Callable[[Alert], Any]] = []

        # Evaluation state
        self._evaluating = False
        self._evaluation_task: asyncio.Task | None = None
        self._evaluation_interval = 60  # seconds

    def add_rule(self, rule: AlertRule) -> None:
        """Add an alert rule."""
        self.rules[rule.name] = rule
        self.logger.info("alert_rule_added", name=rule.name, component=rule.component)

    def remove_rule(self, name: str) -> bool:
        """Remove an alert rule."""
        if name in self.rules:
            del self.rules[name]
            self.logger.info("alert_rule_removed", name=name)
            return True
        return False

    def add_handler(self, handler: Callable[[Alert], Any]) -> None:
        """Add notification handler."""
        self.handlers.append(handler)


    def start(self) -> None:
        """Start alert evaluation loop."""
        if self._evaluation_task is None or self._evaluation_task.done():
            self._evaluation_task = asyncio.create_task(self._evaluate_loop())
            self.logger.info("alert_evaluation_started")

    def stop(self) -> None:
        """Stop alert evaluation loop."""
        if self._evaluation_task and not self._evaluation_task.done():
            self._evaluation_task.cancel()
            self.logger.info("alert_evaluation_stopped")

    async def _evaluate_loop(self) -> None:
        """Periodic alert evaluation."""
        while True:
            try:
                await self.evaluate()
            except Exception as e:
                self.logger.error("alert_evaluation_error", error=str(e))
            await asyncio.sleep(self._evaluation_interval)

    async def evaluate(self) -> None:
        """Evaluate all alert rules."""
        if self._evaluating:
            return

        self._evaluating = True

        try:
            for rule in self.rules.values():
                if not rule.enabled:
                    continue

                await self._evaluate_rule(rule)
        finally:
            self._evaluating = False

    async def _evaluate_rule(self, rule: AlertRule) -> None:
        """Evaluate a single alert rule."""
        # In a real implementation, this would query Prometheus
        # For now, we'll use the metrics collector
        pass

    def fire_alert(
        self,
        rule_name: str,
        message: str,
        value: float,
        threshold: float,
        labels: dict[str, str] | None = None,
    ) -> Alert:
        """Fire an alert."""
        rule = self.rules.get(rule_name)
        if not rule:
            raise ValueError(f"Rule not found: {rule_name}")

        alert_id = f"{rule_name}_{int(datetime.utcnow().timestamp())}"

        alert = Alert(
            id=alert_id,
            name=rule.name,
            level=rule.level,
            component=rule.component,
            message=message,
            value=value,
            threshold=threshold,
            labels={**rule.labels, **(labels or {})},
            annotations={**rule.annotations},
        )

        self.alerts[alert_id] = alert

        # Notify handlers
        for handler in self.handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error("alert_handler_error", handler=handler.__name__, error=str(e))

        self.logger.warning(
            "alert_fired",
            alert_id=alert_id,
            rule=rule_name,
            level=rule.level.value,
            message=message,
        )

        return alert

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        if alert_id not in self.alerts:
            return False

        alert = self.alerts[alert_id]
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.utcnow()
        alert.updated_at = datetime.utcnow()

        self.logger.info("alert_resolved", alert_id=alert_id)
        return True

    def acknowledge_alert(self, alert_id: str, user: str) -> bool:
        """Acknowledge an alert."""
        if alert_id not in self.alerts:
            return False

        alert = self.alerts[alert_id]
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_by = user
        alert.acknowledged_at = datetime.utcnow()
        alert.updated_at = datetime.utcnow()

        self.logger.info("alert_acknowledged", alert_id=alert_id, user=user)
        return True

    def get_active_alerts(self) -> list[Alert]:
        """Get all active (firing) alerts."""
        return [
            a for a in self.alerts.values()
            if a.status == AlertStatus.FIRING
        ]

    def get_alerts(
        self,
        status: AlertStatus | None = None,
        component: str | None = None,
        level: AlertLevel | None = None,
    ) -> list[Alert]:
        """Get alerts with filters."""
        alerts = list(self.alerts.values())

        if status:
            alerts = [a for a in alerts if a.status == status]
        if component:
            alerts = [a for a in alerts if a.component == component]
        if level:
            alerts = [a for a in alerts if a.level == level]

        return alerts

    def get_alert(self, alert_id: str) -> Alert | None:
        """Get a specific alert."""
        return self.alerts.get(alert_id)


# Built-in notification handlers

async def log_handler(alert: Alert) -> None:
    """Log alert handler."""
    logger = get_logger("alerts.notifications")

    level_map = {
        AlertLevel.INFO: logger.info,
        AlertLevel.WARNING: logger.warning,
        AlertLevel.CRITICAL: logger.error,
    }

    log_func = level_map.get(alert.level, logger.info)
    log_func(
        "alert_notification",
        alert_id=alert.id,
        name=alert.name,
        level=alert.level.value,
        component=alert.component,
        message=alert.message,
        value=alert.value,
        threshold=alert.threshold,
    )


async def webhook_handler(alert: Alert, webhook_url: str) -> None:
    """Webhook notification handler."""
    import aiohttp

    payload = {
        "alert_id": alert.id,
        "name": alert.name,
        "level": alert.level.value,
        "component": alert.component,
        "message": alert.message,
        "status": alert.status.value,
        "created_at": alert.created_at.isoformat(),
        "value": alert.value,
        "threshold": alert.threshold,
        "labels": alert.labels,
        "annotations": alert.annotations,
    }

    try:
        async with (
            aiohttp.ClientSession() as session,
            session.post(webhook_url, json=payload) as resp,
        ):
            if resp.status >= 400:
                logger = get_logger("alerts.webhook")
                logger.error("webhook_failed", status=resp.status)

    except Exception as e:
        logger = get_logger("alerts.webhook")
        logger.error("webhook_error", error=str(e))


async def email_handler(alert: Alert, recipients: list[str]) -> None:
    """Email notification handler (placeholder)."""
    logger = get_logger("alerts.email")
    logger.info("email_notification", alert_id=alert.id, recipients=recipients)
    # Would integrate with SMTP or email service


# Default alert rules

def create_default_rules() -> list[AlertRule]:
    """Create default production alert rules."""
    return [
        AlertRule(
            name="high_error_rate",
            component="api",
            query="rate(quant_errors_total[5m])",
            condition="> 0.1",
            level=AlertLevel.WARNING,
            for_duration=timedelta(minutes=5),
            annotations={"description": "High error rate detected"},
        ),
        AlertRule(
            name="backtest_failure_rate",
            component="backtest",
            query="rate(quant_backtest_runs_total{status=\"error\"}[15m])",
            condition="> 0.05",
            level=AlertLevel.WARNING,
            for_duration=timedelta(minutes=10),
            annotations={"description": "Backtest failure rate elevated"},
        ),
        AlertRule(
            name="ml_training_failure",
            component="ml",
            query="rate(quant_ml_training_runs_total{status=\"error\"}[15m])",
            condition="> 0.1",
            level=AlertLevel.CRITICAL,
            for_duration=timedelta(minutes=5),
            annotations={"description": "ML training failures detected"},
        ),
        AlertRule(
            name="data_download_failures",
            component="data",
            query="rate(quant_data_downloads_total{status=\"error\"}[15m])",
            condition="> 0.2",
            level=AlertLevel.WARNING,
            for_duration=timedelta(minutes=10),
            annotations={"description": "Data download failures elevated"},
        ),
        AlertRule(
            name="high_latency",
            component="api",
            query="histogram_quantile(0.95, rate(quant_request_duration_seconds_bucket[5m]))",
            condition="> 5",
            level=AlertLevel.WARNING,
            for_duration=timedelta(minutes=5),
            annotations={"description": "P95 latency above 5 seconds"},
        ),
        AlertRule(
            name="disk_space_low",
            component="storage",
            query="(disk_free_bytes / disk_total_bytes) * 100",
            condition="< 10",
            level=AlertLevel.CRITICAL,
            for_duration=timedelta(minutes=15),
            annotations={"description": "Disk space below 10%"},
        ),
    ]


# Global alert manager
_alert_manager: AlertManager | None = None


def get_alert_manager() -> AlertManager:
    """Get global alert manager."""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()

        # Add default rules
        for rule in create_default_rules():
            _alert_manager.add_rule(rule)

        # Add default handlers
        _alert_manager.add_handler(log_handler)

    return _alert_manager


async def check_metric_threshold(
    metric_name: str,
    value: float,
    threshold: float,
    condition: str,
    rule_name: str,
    component: str,
    level: AlertLevel = AlertLevel.WARNING,
    labels: dict[str, str] | None = None,
) -> Alert | None:
    """Check a metric against threshold and fire alert if needed."""
    manager = get_alert_manager()

    # Parse condition
    triggered = False
    if condition.startswith(">"):
        triggered = value > threshold
    elif condition.startswith("<"):
        triggered = value < threshold
    elif condition.startswith(">="):
        triggered = value >= threshold
    elif condition.startswith("<="):
        triggered = value <= threshold
    elif condition.startswith("=="):
        triggered = value == threshold
    elif condition.startswith("!="):
        triggered = value != threshold

    if triggered:
        return manager.fire_alert(
            rule_name=rule_name,
            message=f"{metric_name} {condition} {threshold} (current: {value})",
            value=value,
            threshold=threshold,
            labels=labels,
        )

    return None
