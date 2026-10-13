"""Launch the local Gradio end-to-end demo without training or downloading weights."""

import argparse
from pathlib import Path

from appleleaf.demo import (
    DEFAULT_CLASSIFIER, DEFAULT_SEGMENTER, DEFAULT_COMPARISON, DemoPipeline, build_demo,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classifier-checkpoint", type=Path, default=DEFAULT_CLASSIFIER)
    parser.add_argument("--segmenter-checkpoint", type=Path, default=DEFAULT_SEGMENTER)
    parser.add_argument("--comparison-checkpoint", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--no-comparison", action="store_true", help="Load only the reference segmenter")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535")
    return args


def main():
    args = parse_args()
    try:
        pipeline = DemoPipeline(
            args.classifier_checkpoint, args.segmenter_checkpoint,
            None if args.no_comparison else args.comparison_checkpoint, args.device,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise SystemExit(str(error)) from error
    print(f"Frozen checkpoints loaded on {pipeline.device}; no training is performed.")
    for name, (_, threshold) in pipeline.segmenters.items():
        print(f"{name}: saved threshold {threshold:.2f}")
    build_demo(pipeline).queue(max_size=8).launch(
        server_name="127.0.0.1", server_port=args.port, share=False, inbrowser=False,
    )


if __name__ == "__main__":
    main()
