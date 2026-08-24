#!/usr/bin/env python
"""Viewer server.

    python app/backend/server.py                       # SAT and Mask R-CNN
    python app/backend/server.py --segmenters sat      # SAT only
    python app/backend/server.py --check               # validate weights and exit

Start-up validates every weight the enabled segmenters need and exits non-zero
naming the missing file and the environment variable that relocates it.  A viewer
that boots and then fails on the clinician's first image is worse than one that
refuses to boot.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "third_party"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from flask import Flask, jsonify  # noqa: E402
from flask_cors import CORS  # noqa: E402

from api import api  # noqa: E402
from cases import CaseStore  # noqa: E402
from config import Settings  # noqa: E402
from pipeline import SEGMENTERS, Pipeline  # noqa: E402

log = logging.getLogger("iopc.viewer")


def create_app(settings: Settings, segmenters: tuple[str, ...]) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = settings.max_upload_bytes * 32

    CORS(app, resources={r"/api/*": {"origins": settings.cors_origins}})

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    app.extensions["iopc_settings"] = settings
    app.extensions["iopc_pipeline"] = Pipeline(settings, segmenters)
    app.extensions["iopc_cases"] = CaseStore()
    app.register_blueprint(api)

    @app.get("/")
    def index():
        return jsonify(
            {
                "name": "IOP-Compass viewer",
                "api": "/api/health",
                "segmenters": list(segmenters),
            }
        )

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument(
        "--segmenters",
        default=",".join(SEGMENTERS),
        help=f"comma-separated subset of {SEGMENTERS}",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate the configuration and exit without serving",
    )
    parser.add_argument(
        "--no-warm-up",
        action="store_true",
        help="load models on first request instead of at start-up",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    segmenters = tuple(s.strip() for s in args.segmenters.split(",") if s.strip())
    unknown = [s for s in segmenters if s not in SEGMENTERS]
    if unknown:
        print(f"unknown segmenter(s): {unknown}; expected {SEGMENTERS}", file=sys.stderr)
        return 2
    if not segmenters:
        print("at least one segmenter must be enabled", file=sys.stderr)
        return 2

    settings = Settings.from_env()
    problems = settings.missing(segmenters)
    if problems:
        print("viewer cannot start:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        print(
            "\nSee the weights section of README.md for what to download and where "
            "to put it.",
            file=sys.stderr,
        )
        return 1

    app = create_app(settings, segmenters)
    if args.check:
        print(f"configuration ok; segmenters: {', '.join(segmenters)}")
        return 0

    if not args.no_warm_up:
        pipeline = app.extensions["iopc_pipeline"]
        log.info("warming up on %s (%s)", pipeline.device, pipeline.device_name())
        pipeline.warm_up()

    log.info("serving on http://%s:%d", args.host, args.port)
    # threaded=True keeps uploads and image fetches responsive; every model call is
    # serialised inside Pipeline.lock, so the GPU still sees one request at a time.
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
