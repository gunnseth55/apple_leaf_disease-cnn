"""Exercise both segmenters through the running local Gradio upload API."""

import hashlib
import json
from pathlib import Path

from gradio_client import Client, handle_file
import numpy as np
from PIL import Image

from appleleaf.demo import (
    DEFAULT_CLASSIFIER, DEFAULT_SEGMENTER, DEFAULT_COMPARISON, PRIMARY, COMPARISON,
)


def main():
    paths = [DEFAULT_CLASSIFIER, DEFAULT_SEGMENTER, DEFAULT_COMPARISON]
    before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    source = Path("data/lesion_zenodo/restricted/Restricted_Plant_Village_Dataset/images/test/black_rot_1.jpg")
    output = Path("artifacts/demo_verification")
    output.mkdir(parents=True, exist_ok=True)
    client = Client("http://127.0.0.1:7860", verbose=False)
    results = []
    for index, model in enumerate([PRIMARY, COMPARISON]):
        result = client.predict(handle_file(str(source)), model, api_name="/analyze")
        assert len(result) == 5
        assert abs(sum(item["confidence"] for item in result[0]["confidences"]) - 1) < 1e-5
        images = [Image.open(path) for path in result[1:4]]
        with Image.open(source) as original:
            assert all(image.size == original.size for image in images)
            np.testing.assert_array_equal(np.array(images[0]), np.array(original.convert("RGB")))
        assert set(np.unique(np.asarray(images[2]))).issubset({0, 255})
        images[1].save(output / f"overlay_{index}.png")
        results.append({"segmenter": model, "scores": result[0],
                        "image_size": images[1].size, "details": result[4]})
        for image in images:
            image.close()
    assert before == [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    report = {"results": results, "checkpoint_sha256": dict(zip(map(str, paths), before)),
              "checkpoints_unchanged": True}
    (output / "smoke_test.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("PASS: both upload/inference paths, normalized class scores, original-image parity,")
    print("overlay dimensions, binary mask export, and unchanged checkpoint hashes.")
    print(f"Report: {output / 'smoke_test.json'}")


if __name__ == "__main__":
    main()
