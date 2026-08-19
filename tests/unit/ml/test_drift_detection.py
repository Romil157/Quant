"""Unit tests for ML concept drift detection."""
import pytest

from quant.ml.online import DriftDetector


def test_drift_detection_normal_regime():
    """Test that stationary errors do not trigger drift."""
    detector = DriftDetector(window_size=20, threshold=0.5)

    # 30 updates with constant small prediction error
    for _ in range(30):
        drift = detector.update(10.0, 10.1)
        assert not drift
    assert not detector.drift_detected


def test_drift_detection_sudden_shock():
    """Test that sudden large error surge triggers drift detection."""
    detector = DriftDetector(window_size=20, threshold=0.5)

    # 10 initial baseline low-error observations
    for _ in range(10):
        detector.update(10.0, 10.1)

    # 10 high-error observations (regime shift)
    for _ in range(10):
        drift = detector.update(10.0, 30.0)

    # Window now contains 10 baseline errors (0.01) and 10 large errors (400.0)
    assert detector.drift_detected is True


def test_drift_detector_error_buffer_and_reset():
    """Test error buffer bounds and reset lifecycle."""
    detector = DriftDetector(window_size=15, threshold=0.5)

    for i in range(30):
        detector.update(float(i), float(i + 10))

    # Error buffer should not exceed window_size
    assert len(detector.errors) <= 15

    # Test reset
    detector.reset()
    assert len(detector.errors) == 0
    assert not detector.drift_detected
