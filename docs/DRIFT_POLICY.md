# Model Drift Policy

This document defines the policy for handling model drift detection in production.

## Drift Detection Overview

The platform uses `DriftDetector` (in `src/quant/ml/online.py`) to monitor prediction errors in real-time:

- **Metric**: MSE (Mean Squared Error) or MAE
- **Window**: 100 predictions (configurable)
- **Threshold**: 5% relative increase in error rate (configurable)
- **Comparison**: Recent half-window vs. reference half-window

## Drift Response Policy

When `DriftDetector.drift_detected == True`:

### Level 1: Alert Only (Default)
- **Action**: Log alert + increment drift counter
- **Continue**: Normal operation, model continues to update
- **Review**: Human reviews at next scheduled review (daily)
- **Escalation**: If drift persists > 24 hours → Level 2

### Level 2: Auto-Fallback to Rule-Based
- **Trigger**: Drift persists > 24 hours OR error rate > 2x baseline
- **Action**: 
  1. Switch strategy selection to best rule-based strategy (from benchmark report)
  2. Continue ML model updates in background (shadow mode)
  3. Alert: "ML model drifted - fallen back to rule-based"
- **Recovery**: If ML model error rate returns to baseline for 48 hours → Level 1

### Level 3: Hold New Positions
- **Trigger**: Drift + risk limit breach OR error rate > 5x baseline
- **Action**:
  1. Halt new position entry for ML-driven strategies
  2. Allow existing positions to be managed (stop-loss, take-profit)
  2. Full manual review required before re-enabling
- **Recovery**: Manual approval required

## Drift Metric Exposure

Drift metrics are exposed via Prometheus `/metrics` endpoint:

```prometheus
# HELP quant_ml_drift_detected Drift detection status (1=drift, 0=no drift)
# TYPE quant_ml_drift_detected gauge
quant_ml_drift_detected{model="online_ensemble"} 0

# HELP quant_ml_drift_error_rate Current prediction error rate
# TYPE quant_ml_drift_error_rate gauge
quant_ml_drift_error_rate{model="online_ensemble"} 0.0023

# HELP quant_ml_drift_baseline_error Baseline error rate for comparison
# TYPE quant_ml_drift_baseline_error gauge
quant_ml_drift_baseline_error{model="online_ensemble"} 0.0021

# HELP quant_ml_drift_relative_change Relative change in error rate
# TYPE quant_ml_drift_relative_change gauge
quant_ml_drift_relative_change{model="online_ensemble"} 0.095
```

## Configuration

Drift detection is configured via `configs/paper.yaml`:

```yaml
ml:
  drift_detection:
    enabled: true
    window_size: 100
    threshold: 0.05  # 5% relative increase
    metric: "mse"    # or "mae"
    response_level: 1  # 1=alert, 2=fallback, 3=hold
    shadow_mode: true  # Continue ML updates during fallback
```

## Decision Matrix

| Drift Duration | Error Rate vs Baseline | Response Level |
|----------------|------------------------|----------------|
| < 1 hour | < 2x | Level 1 (Alert) |
| 1-24 hours | < 2x | Level 1 (Alert + monitor) |
| > 24 hours | < 2x | Level 2 (Fallback) |
| Any | 2x - 5x | Level 2 (Fallback) |
| Any | > 5x | Level 3 (Hold) |
| Any + risk breach | Any | Level 3 (Hold) |

## Recovery Procedures

### Level 1 → Normal
- Drift resolves (error rate returns to baseline)
- Alert auto-clears
- No action needed

### Level 2 → Level 1
- ML model error rate returns to baseline for 48 continuous hours
- Auto-switch back to ML model
- Log: "ML model recovered - resumed primary strategy"

### Level 3 → Level 2
- Manual review confirms:
  - Root cause identified and fixed
  - ML model retrained on fresh data
  - Error rate < 1.5x baseline on validation set
- Manual approval required to re-enable

## Monitoring Dashboard

Drift metrics are visible in:
1. **Streamlit Dashboard** (`dashboard/app.py`) - ML tab
2. **Prometheus/Grafana** - `quant_ml_drift_*` metrics
3. **Paper Trading Logs** - Structured log entries

## Incident Response Playbook

### On Drift Alert (Level 1)
1. Check dashboard: Is error rate actually elevated?
2. Check data quality: Any stale/missing data?
3. Check model: Has regime changed?
4. Document findings in incident log

### On Fallback Trigger (Level 2)
1. Verify fallback executed successfully
2. Confirm rule-based strategy is active
3. Begin shadow-mode ML monitoring
4. Schedule root cause analysis within 4 hours

### On Position Hold (Level 3)
1. Confirm all new ML orders blocked
2. Verify existing positions have stops
3. Escalate to quant lead / PM
4. Do not re-enable without written approval

## Testing Drift Detection

```bash
# Unit test
uv run pytest tests/unit/ml/test_drift_detection.py -v

# Integration test (simulates drift)
uv run pytest tests/integration/test_drift_detection.py -v
```

## Version

v1.0 — 2026-08-18