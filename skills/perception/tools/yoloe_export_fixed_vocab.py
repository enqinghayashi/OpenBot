#!/usr/bin/env python3
"""yoloe_export_fixed_vocab.py -- Export YOLOE with fixed vocabulary to ONNX.

Creates an ONNX model with a fixed set of class names baked in (no dynamic vocabulary).
The exported ONNX has opset <= 17 for TensorRT 8.5 compatibility.

Usage:
    yoloe_export_fixed_vocab.py \
        --pt /path/to/yoloe-26n-seg.pt \
        --bank /path/to/vocab_bank_yoloe-26n-seg.npz \
        --vocab bottle chair cup ... \
        --output /path/to/model.onnx \
        [--imgsz 640]

Outputs:
    - model.onnx: ONNX model with fixed vocabulary
    - model.onnx.json: Sidecar with class names in order
    - Prints md5 and line counts to stdout
"""
import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np


def compute_md5(filepath):
    """Compute MD5 hash of a file."""
    md5 = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            md5.update(chunk)
    return md5.hexdigest()


def count_lines(filepath):
    """Count lines in a text file; return 0 for binary."""
    try:
        with open(filepath, 'r') as f:
            return sum(1 for _ in f)
    except (UnicodeDecodeError, OSError):
        return 0


def export_fixed_vocabulary(pt_path, bank_path, vocab_names, output_onnx, imgsz=(384, 640)):
    """Export YOLOE model with fixed vocabulary to ONNX.

    Args:
        pt_path: Path to the YOLOE .pt weights
        bank_path: Path to the vocabulary bank NPZ
        vocab_names: List of class names in order
        output_onnx: Path to save the ONNX model
        imgsz: Image size (default (384, 640) as tuple for static size)

    Returns:
        dict with keys: 'onnx_path', 'json_path', 'onnx_md5', 'vocab_names'
    """
    try:
        import torch
        from ultralytics import YOLOE
    except ImportError as e:
        raise RuntimeError(f"Failed to import torch/ultralytics: {e}")

    # vocab bank
    try:
        with np.load(bank_path, allow_pickle=False) as z:
            bank_prompts = [str(p) for p in z['prompts'].tolist()]
            bank_embeds = np.asarray(z['embeds'], dtype=np.float32)
            bank_model = str(z['model']) if 'model' in z.files else None
    except Exception as e:
        raise RuntimeError(f"Failed to load bank {bank_path}: {e}")

    # embeddings for our vocab
    embeddings_list = []
    for name in vocab_names:
        if name not in bank_prompts:
            raise ValueError(f"Vocabulary name '{name}' not in bank (has {len(bank_prompts)} prompts)")
        idx = bank_prompts.index(name)
        embeddings_list.append(bank_embeds[idx])

    # -> [len(vocab_names), 512]
    fixed_embeddings = np.stack(embeddings_list, axis=0).astype(np.float32)

    try:
        model = YOLOE(pt_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load YOLOE model from {pt_path}: {e}")

    # needs set_classes()
    if not callable(getattr(model, 'set_classes', None)):
        raise RuntimeError("Model has no set_classes() method")

    tensor = torch.from_numpy(fixed_embeddings)[None]  # shape [1, N, 512]
    model.set_classes(vocab_names, tensor)

    # opset 17: TensorRT 8.5 reads it
    try:
        # list -> tuple
        imgsz_tuple = tuple(imgsz) if isinstance(imgsz, list) else imgsz
        output_path = model.export(
            format='onnx',
            imgsz=imgsz_tuple,
            opset=17,
            simplify=True,
            device=0,
        )
    except Exception as e:
        raise RuntimeError(f"ONNX export failed: {e}")

    # move to the requested path
    output_onnx_path = Path(output_onnx)
    if output_path != str(output_onnx_path):
        import shutil
        shutil.move(str(output_path), str(output_onnx_path))

    # sidecar JSON, same keys as ultralytics
    # tuple or scalar imgsz (old callers)
    if isinstance(imgsz, (tuple, list)):
        imgsz_list = list(imgsz)
    else:
        imgsz_list = [imgsz, imgsz]

    metadata = {
        'task': 'segment',
        'model': 'yoloe-26n',
        'imgsz': imgsz_list,  # height, then width
        'stride': 32,
        'names': {i: name for i, name in enumerate(vocab_names)},
        'class_names': vocab_names,
        'num_classes': len(vocab_names),
        'opset': 17,
    }
    json_path = Path(str(output_onnx_path) + '.json')
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # md5 the onnx
    onnx_md5 = compute_md5(str(output_onnx_path))

    return {
        'onnx_path': str(output_onnx_path),
        'json_path': str(json_path),
        'onnx_md5': onnx_md5,
        'vocab_names': vocab_names,
        'num_classes': len(vocab_names),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pt', required=True, help='Path to YOLOE .pt weights')
    parser.add_argument('--bank', required=True, help='Path to vocabulary bank NPZ')
    parser.add_argument('--vocab', nargs='+', required=True, help='Class names in order')
    parser.add_argument('--output', required=True, help='Path to save ONNX model')
    parser.add_argument('--imgsz', type=int, nargs=2, default=[384, 640], help='Image size [height width]')
    args = parser.parse_args()

    # inputs first
    if not os.path.isfile(args.pt):
        print(f"ERROR: {args.pt} not found", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.bank):
        print(f"ERROR: {args.bank} not found", file=sys.stderr)
        sys.exit(1)

    # mkdir -p the output dir
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = export_fixed_vocabulary(
            args.pt, args.bank, args.vocab, args.output, args.imgsz
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # evidence
    print(f"ONNX_PATH: {result['onnx_path']}")
    print(f"ONNX_MD5: {result['onnx_md5']}")
    print(f"JSON_PATH: {result['json_path']}")
    print(f"NUM_CLASSES: {result['num_classes']}")

    # file stats
    onnx_lines = count_lines(result['onnx_path'])
    json_lines = count_lines(result['json_path'])
    print(f"ONNX_LINES: {onnx_lines}")
    print(f"JSON_LINES: {json_lines}")


if __name__ == '__main__':
    main()
