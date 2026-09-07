#!/usr/bin/env python3
"""trt_add_ultralytics_meta.py -- Prepend ultralytics metadata header to TensorRT engine.

Python 3.8 stdlib only. Reads an engine built by trtexec and prepends the metadata
header that ultralytics' own TensorRT exporter writes.

The header format (from ultralytics/utils/export/engine.py:452-455):
    1. 4-byte little-endian signed integer: length of metadata JSON
    2. UTF-8 encoded JSON metadata string
    3. TensorRT engine binary data

Ultralytics metadata keys (from ultralytics/engine/exporter.py ~line 905-921):
    - task: 'detect' | 'segment' | 'pose' | 'obb' | 'classify'
    - model: model stem (e.g., 'yoloe-26n')
    - imgsz: [height, width]
    - names: {0: 'bottle', 1: 'chair', ...}
    - stride: model stride (typically 32)
    - etc.

Usage:
    trt_add_ultralytics_meta.py \
        --engine /path/to/raw.engine \
        --names bottle chair cup ... \
        --output /path/to/final.engine \
        [--task segment] [--imgsz 640 640] [--stride 32]

The output engine can be loaded by ultralytics YOLO(engine_path) and will have
the correct class names accessible.
"""
import argparse
import json
import struct
import sys


def add_ultralytics_metadata(input_engine, output_engine, names, task='segment',
                             imgsz=(640, 640), stride=32):
    """Prepend ultralytics metadata to a TensorRT engine.

    Args:
        input_engine: Path to raw TensorRT engine (no metadata)
        output_engine: Path to save engine with metadata
        names: List of class names in order
        task: Task type ('detect', 'segment', 'pose', etc.)
        imgsz: (height, width) tuple
        stride: Model stride

    Returns:
        dict with 'output_path', 'metadata_len', 'engine_len'

    Raises:
        RuntimeError if input engine already contains a metadata header
    """
    # same shape ultralytics writes
    metadata = {
        'task': task,
        'model': 'exported',  # box tool fills this in
        'imgsz': list(imgsz),
        'stride': stride,
        'names': {i: name for i, name in enumerate(names)},
    }

    # json -> bytes
    meta_json = json.dumps(metadata, separators=(',', ':'))
    meta_bytes = meta_json.encode('utf-8')
    meta_len = len(meta_bytes)

    # read engine, check for an existing header
    try:
        with open(input_engine, 'rb') as f:
            first_bytes = f.read(4)
            f.seek(0)
            engine_data = f.read()
    except IOError as e:
        raise RuntimeError(f"Failed to read engine from {input_engine}: {e}")

    # header = 4-byte LE int + UTF-8 JSON
    if len(first_bytes) == 4:
        try:
            existing_meta_len = int.from_bytes(first_bytes, byteorder='little')
            # sane len: 0 < n < 102400
            if 0 < existing_meta_len < 102400:
                # decode as JSON
                if len(engine_data) > 4 + existing_meta_len:
                    supposed_json = engine_data[4:4 + existing_meta_len]
                    try:
                        json.loads(supposed_json.decode('utf-8'))
                        # it parsed: really a header
                        raise RuntimeError(
                            f"Input engine {input_engine} already contains a metadata header. "
                            "Use a raw engine file (no metadata) as input."
                        )
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        # not JSON: not a header
                        pass
        except (ValueError, IndexError):
            # not an int: safe to go on
            pass

    # out: len + meta + engine
    try:
        with open(output_engine, 'wb') as f:
            # 4-byte LE signed len
            f.write(struct.pack('<i', meta_len))
            # the JSON
            f.write(meta_bytes)
            # then the engine
            f.write(engine_data)
    except IOError as e:
        raise RuntimeError(f"Failed to write engine to {output_engine}: {e}")

    return {
        'output_path': output_engine,
        'metadata_len': meta_len,
        'engine_len': len(engine_data),
        'total_len': 4 + meta_len + len(engine_data),
        'metadata': metadata,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', required=True,
                        help='Path to raw TensorRT engine (no metadata)')
    parser.add_argument('--names', nargs='+', required=True,
                        help='Class names in order')
    parser.add_argument('--output', required=True,
                        help='Path to save engine with metadata')
    parser.add_argument('--task', default='segment',
                        help='Task type (default: segment)')
    parser.add_argument('--imgsz', type=int, nargs=2, default=[640, 640],
                        help='Image size [height width] (default: 640 640)')
    parser.add_argument('--stride', type=int, default=32,
                        help='Model stride (default: 32)')
    args = parser.parse_args()

    # inputs first
    import os
    if not os.path.isfile(args.engine):
        print(f"ERROR: {args.engine} not found", file=sys.stderr)
        sys.exit(1)

    # mkdir -p the output dir
    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir and not os.path.isdir(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            print(f"ERROR: Failed to create {output_dir}: {e}", file=sys.stderr)
            sys.exit(1)

    try:
        result = add_ultralytics_metadata(
            args.engine, args.output, args.names,
            task=args.task, imgsz=tuple(args.imgsz), stride=args.stride
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # evidence
    print(f"OUTPUT_PATH: {result['output_path']}")
    print(f"METADATA_LEN: {result['metadata_len']}")
    print(f"ENGINE_LEN: {result['engine_len']}")
    print(f"TOTAL_LEN: {result['total_len']}")
    print(f"METADATA: {json.dumps(result['metadata'])}")
