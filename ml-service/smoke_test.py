"""Run with the deployed ML environment to verify the real V4 pipeline primitives."""
import os
os.environ["DEMO_MODE"] = "false"
import numpy as np
import app

def main():
    app.init_models()
    assert app.status == {"classifierLoaded": True, "segmentationLoaded": True, "calibrationLoaded": True}
    sample = np.zeros((1, 300, 300, 3), dtype="float32")
    grade, binary = app.normalize_classifier_outputs(app.classifier.predict(sample, verbose=0))
    result = app.build_inference_result(grade, binary)
    assert result["modelVersion"] == "V4 Multi-Domain" and len(result["gradeProbabilities"]) == 5
    assert 0 <= result["referableProbability"] <= 1 and result["decision"] in ("REFER", "NON_REFER")
    heatmap = app.generate_gradcam(sample)
    assert heatmap.dtype == np.float32 and heatmap.flags["C_CONTIGUOUS"] and np.isfinite(heatmap).all()
    seg = app.segmenter.predict(np.zeros((1, 384, 384, 3), dtype="float32"), verbose=0)
    assert seg.shape == (1, 384, 384, 1) and np.isfinite(seg).all()
    print("REAL V4 SMOKE TEST PASSED")

if __name__ == "__main__": main()
