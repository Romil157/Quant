# Alerting System

This document describes the alerting system for the Quant platform production loop.

## Alert Types

### 1. StaleDataAlert
**Trigger**: Most recent successful data pull older than expected schedule
- **Condition**: `last_successful_pull < now() - expected_interval * 2`
- **Severity**: WARNING
- **Action**: Log + alert; investigate data provider connectivity

### 2. OrderRejectionSpikeAlert
**Trigger**: Order rejection rate exceeds threshold over rolling window
- **Condition**: `rejection_rate_5min > 0.1` (configurable)
- **Severity**: CRITICAL
- **Action**: Log + alert + auto-pause new orders; investigate data/connectivity

### 3. DrawdownBreachAlert
**Trigger**: Max drawdown check fires in live loop
- **Condition**: Portfolio drawdown > configured `max_drawdown` threshold
- **Severity**: CRITICAL
- **Action**: Log + alert + auto-reduce positions per risk config

### 4. SchedulerHeartbeatAlert
**Trigger**: Scheduled run fails to execute
- **Condition**: Expected run time passed without execution start
- **Severity**: CRITICAL
- **Action**: Log + alert; investigate scheduler process

### 5. ModelDriftAlert
**Trigger**: ML model drift detector fires
- **Condition**: `DriftDetector.drift_detected == True`
- **Severity**: WARNING
- **Action**: Log + alert; follow drift policy (see DRIFT_POLICY.md)

## Alert Configuration

All alerts are configured via `configs/paper.yaml` under `monitoring.alerts`:

```yaml
monitoring:
  alerts:
    stale_data_threshold_minutes: 30
    rejection_spike_threshold: 0.1
    rejection_window_minutes: 5
    drawdown_alert_enabled: true
    scheduler_heartbeat_enabled: true
    model_drift_alert_enabled: true
    webhook_url: ""  # Optional: Slack/webhook URL
```

## Alert Delivery

### Structured Logging
All alerts emit structured log entries:
```json
{
  "level": "WARNING",
  "event": "alert_triggered",
  "alert_type": "StaleDataAlert",
  "message": "Data pull stale for 45 minutes",
  "timestamp": "2026-08-18T10:30:00Z",
  "details": {
    "last_pull": "2026-08-18T09:45:00Z",
    "threshold_minutes": 30
  }
}
```

### Webhook Integration (Optional)
If `webhook_url` is configured, alerts POST JSON payload:
```json
{
  "alert_type": "StaleDataAlert",
  "severity": "WARNING",
  "message": "Data pull stale for 45 minutes",
  "timestamp": "2026-08-18T10:30:00Z",
  "details": {...}
}
```

## Integration Points

### In Paper Trading Loop
```python
# In paper/engine.py or scheduler job
async def paper_trading_job():
    try:
        # ... run paper trading step ...
    except DataStaleException as e:
        alert_manager.trigger(StaleDataAlert(details=str(e)))
    except OrderRejectionException as e:
        alert_manager.trigger(OrderRejectionSpikeAlert(details=str(e)))
```

### In Scheduler
```python
# In production/scheduler.py
async def monitored_func():
    try:
        await func()
    except Exception as e:
        if "stale" in str(e).lower():
            alert_manager.trigger(StaleDataAlert(details=str(e)))
        raise
```

### In Monitoring
```python
# In production/monitoring.py
class MetricsCollector:
    def record_rejection(self, strategy: str, rejected: bool):
        if rejected:
            self.rejection_rate.update(1)
        else:
            self.rejection_rate.update(0)
        
        if self.rejection_rate.rate() > config.rejection_spike_threshold:
            alert_manager.trigger(OrderRejectionSpikeAlert())
```

## Alert History

All alerts are persisted to SQLite (via `production/alerts.py`) with:
- Alert type
- Severity
- Timestamp
- Message
- Details (JSON)
- Resolution status (open/acknowledged/resolved)

## Testing Alerts

```bash
# Test stale data alert
uv run python -c "
from quant.production.alerts import AlertManager, StaleDataAlert
am = AlertManager()
am.trigger(StaleDataAlert(details='test'))
"

# Test rejection spike
uv run python -c "
from quant.production.alerts import AlertManager, OrderRejectionSpikeAlert
am = AlertManager()
am.trigger(OrderRejectionSpikeAlert(details={'rate': 0.15}))
"
```

## Escalation

| Alert Type | Auto-Action | Human Review |
|------------|-------------|--------------|
| StaleData | Log + retry | If > 2 consecutive |
| RejectionSpike | Pause orders | Immediate |
| DrawdownBreach | Reduce exposure | Immediate |
| SchedulerHeartbeat | Retry next cycle | If 2+ missed |
| ModelDrift | Alert only | Per drift policy |

---

*Last updated: 2026-08-18*