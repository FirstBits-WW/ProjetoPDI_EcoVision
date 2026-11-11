import sys
from types import ModuleType

import pytest

from app.services.detector import DetectorUnavailableError, YoloDetector


def test_yolo_download_failure_becomes_actionable_error(monkeypatch) -> None:
    ultralytics_module = ModuleType("ultralytics")

    def fail_to_load(model_path: str):
        raise ConnectionError("TLS certificate verification failed")

    ultralytics_module.YOLO = fail_to_load
    monkeypatch.setitem(sys.modules, "ultralytics", ultralytics_module)

    with pytest.raises(DetectorUnavailableError, match="YOLO_MODEL_PATH") as error:
        YoloDetector("yolo11n.pt", confidence=0.25)

    assert isinstance(error.value.__cause__, ConnectionError)
