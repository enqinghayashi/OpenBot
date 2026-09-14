#!/usr/bin/env python3
"""yoloe_box_probe.py -- READ-ONLY S1 measurement probe for the YOLOE batch-4 lane.

Batch 4 (docs/YOLOE_BATCH4_ARCHITECTURE.md section 11).  Operator-run on the box;
never run by an agent against the robot.  py3.8, stdlib + OPTIONAL imports
(numpy / cv2 / torch / ultralytics / rclpy are each probed and reported, never
required).

What it does (each step is independent; a failing step is recorded, never fatal):
  0. provenance  md5 of the DEPLOYED node / projector vs the staging base md5s,
                 `match_staging_base: {"node": bool, "projector": bool}` and a
                 unified diff `deployed_vs_base_<file>.diff` in --out-dir when false.
  1. env         interpreter, ultralytics / torch / CUDA / TensorRT, YOLOE API,
                 clip importability, free RAM, nproc, 1-min load, file presence,
                 and `torch.load(<each .pt>, weights_only=False)` per checkpoint.
  2. --bench     the engine, then each .pt, on CPU and CUDA, at threads 1/2/4:
                 median/p90 ms, process CPU %, RSS; for YOLOE: bank_swap_ms.
  3. --record N  saves N frames of /camera/color/image_raw as PNGs (subscribe only).
     --agree DIR engine (or --baseline-pt) vs YOLOE per PNG: base-20 label-set
                 agreement, |dconf| median, extra-row count per seeded prompt; TWO
                 tiers (architecture section 17 M7): raw (GATE_CONF 0.20, reported)
                 and pick (--tier-conf, default 0.35, the G-AGREE gate tier), plus
                 <out-dir>/agree_frames.json with both detectors' box lists per frame.
  4. output      ONE JSON line on stdout + --out <file>; exit 0 ALWAYS.

Writes ONLY inside --out-dir (the ultralytics settings dir is redirected there via
YOLO_CONFIG_DIR, and YOLO_OFFLINE=1 stops network probes).  Publishes nothing,
calls no service, moves nothing.  The text encoder is NEVER invoked: a vocabulary
is set only as set_classes(names, embeddings) from a laptop-built bank NPZ.
"""
import argparse
import difflib
import glob
import hashlib
import json
import os
import re
import resource
import statistics
import struct
import sys
import time
import traceback
import zlib

PROBE_VERSION = '1.0'

# base md5s (BASE_MD5.json)
# hand-placed copies, no repo source:
# md5 = is it base?
BASE_NODE_MD5 = '8b1619e50724cecdd40aa759cd950b6e'        # md5 of yolo_node_seg.py
BASE_PROJECTOR_MD5 = '40e1757677a1b754f093e49358fd7e42'   # md5 of depth_projector_node.py
BASE_NODE_NAME = 'yolo_node_seg.py'
BASE_PROJECTOR_NAME = 'depth_projector_node.py'

DEFAULT_LEON_DIR = '/home/unitree/leon/openbot'           # from launch_pick_stack.sh:84
DEFAULT_ENGINE = '/home/unitree/yolo11n-seg.engine'      # see yolo_node_seg.py:236
DEFAULT_ASSETS_DIR = '/home/unitree/leon/yoloe'          # INFERRED
IMAGE_TOPIC = '/camera/color/image_raw'                  # see yolo_node_seg.py:258
FRAME_W, FRAME_H = 424, 240
GATE_CONF = 0.20                                         # see yolo_node_seg.py:96
IOU = 0.45
EXTRA_MIN_CONF = 0.35                                    # config.yaml det_conf_floor
PICK_TIER_CONF = 0.35                                    # < 0.35 never binds
AGREE_FRAMES_FILE = 'agree_frames.json'
EXTRAS_PER_CHUNK = 16                                    # cap: 16/chunk

# Base-20 = TRACKED_CLASSES verbatim
# spaces kept exactly as the detector emits them
BASE_20 = ('bottle', 'chair', 'cup', 'laptop', 'cell phone', 'book', 'keyboard',
           'mouse', 'remote', 'clock', 'vase', 'potted plant', 'person', 'apple',
           'backpack', 'sports ball', 'bowl', 'suitcase', 'handbag', 'fork')

# top-level keys, test pinned
# tests/unit/perception/test_yoloe_probe_schema.py
REQUIRED_KEYS = ('probe_version', 'interpreter', 'args', 'out_dir', 'provenance',
                 'match_staging_base', 'env', 'torch_load', 'banks', 'bench',
                 'record', 'agree', 'errors', 'elapsed_s')


# ------------------------------------------------------------ the stdout guard
# stdout: one line
# ultralytics/TensorRT -> stderr at FD
# + sys.stdout
_REAL_STDOUT_FD = None


def _guard_stdout():
    global _REAL_STDOUT_FD
    try:
        sys.stdout.flush()
    except Exception:
        pass
    _REAL_STDOUT_FD = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr


def _emit_line(line):
    data = (line + '\n').encode('utf-8')
    fd = _REAL_STDOUT_FD if _REAL_STDOUT_FD is not None else 1
    while data:
        n = os.write(fd, data)
        data = data[n:]


def _log(msg):
    sys.stderr.write('[probe] %s\n' % (msg,))
    sys.stderr.flush()


# ---------------------------------------------------------- small helpers
def md5_file(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _exc(e):
    return '%s: %s' % (type(e).__name__, e)


def _rss_mb():
    """Current RSS (VmRSS) in MB; falls back to peak ru_maxrss."""
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return round(int(line.split()[1]) / 1024.0, 1)
    except Exception:
        pass
    return None


def _maxrss_mb():
    """This process's own peak (VmHWM).  ru_maxrss is inherited across execve on Linux
    (it can report the launching shell's/parent's peak), so it is only the fallback."""
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmHWM:'):
                    return round(int(line.split()[1]) / 1024.0, 1)
    except Exception:
        pass
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)


def _cpu_s():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def _pct(sorted_vals, q):
    if not sorted_vals:
        return None
    i = min(len(sorted_vals) - 1, int(q * len(sorted_vals)))
    return round(sorted_vals[i], 2)


def _parse_list(s, cast=str):
    return [cast(x.strip()) for x in str(s).split(',') if x.strip()]


# ---------------------------------------------------------- 0. provenance check
def _find_base_src(explicit, name, want_md5, probe_dir):
    cands = []
    if explicit:
        cands.append(explicit)
    cands.append(os.path.join(probe_dir, 'base', name))
    cands.append(os.path.join(os.path.dirname(probe_dir), name))
    for c in cands:
        try:
            if os.path.isfile(c) and md5_file(c) == want_md5:
                return c
        except Exception:
            pass
    return None


def provenance(args, out_dir, errors):
    probe_dir = os.path.dirname(os.path.abspath(__file__))
    prov, match = {}, {}
    for key, deployed, want, base_src, name in (
            ('node', args.deployed_node, args.base_node_md5, args.base_node_src, BASE_NODE_NAME),
            ('projector', args.deployed_projector, args.base_projector_md5,
             args.base_projector_src, BASE_PROJECTOR_NAME)):
        ent = {'deployed_path': deployed, 'expected_md5': want, 'deployed_md5': None,
               'match': False, 'diff_file': None, 'base_src': None}
        try:
            if os.path.isfile(deployed):
                ent['deployed_md5'] = md5_file(deployed)
                ent['match'] = ent['deployed_md5'] == want
            else:
                ent['error'] = 'deployed file missing'
        except Exception as e:
            ent['error'] = _exc(e)
            errors.append('provenance.%s: %s' % (key, _exc(e)))
        match[key] = bool(ent['match'])
        if not ent['match']:
            diff_path = os.path.join(out_dir, 'deployed_vs_base_%s.diff'
                                     % os.path.basename(deployed or key))
            src = _find_base_src(base_src, name, want, probe_dir)
            ent['base_src'] = src
            try:
                if src is None:
                    body = ('# base source with md5 %s not found (pass --base-%s-src <file>);\n'
                            '# deployed md5 = %s\n' % (want, key, ent['deployed_md5']))
                elif not os.path.isfile(deployed):
                    body = '# deployed file missing: %s\n' % (deployed,)
                else:
                    with open(src, 'r', errors='replace') as f:
                        a = f.readlines()
                    with open(deployed, 'r', errors='replace') as f:
                        b = f.readlines()
                    body = ''.join(difflib.unified_diff(
                        a, b, fromfile='base/%s (%s)' % (name, want),
                        tofile='deployed/%s (%s)' % (os.path.basename(deployed),
                                                     ent['deployed_md5'])))
                    if not body:
                        body = '# no textual difference (md5 differs: line endings/bytes)\n'
                with open(diff_path, 'w') as f:
                    f.write(body)
                ent['diff_file'] = diff_path
            except Exception as e:
                errors.append('provenance.%s.diff: %s' % (key, _exc(e)))
        prov[key] = ent
    return prov, match


# --------------------------------------------------------- 1. env report
def _try_import(name):
    try:
        mod = __import__(name)
        return mod, None
    except Exception as e:  # noqa: BLE001 - every failure to import is data
        return None, _exc(e)


def _meminfo():
    out = {}
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                k, v = line.split(':', 1)
                if k in ('MemTotal', 'MemFree', 'MemAvailable', 'SwapFree'):
                    out[k + '_mb'] = round(int(v.split()[0]) / 1024.0, 1)
    except Exception as e:
        out['error'] = _exc(e)
    return out


def environment(args):
    env = {'python': sys.version.split()[0], 'executable': sys.executable,
           'prefix': sys.prefix, 'base_prefix': getattr(sys, 'base_prefix', None),
           'sys_path': list(sys.path), 'nproc': os.cpu_count()}
    try:
        env['loadavg_1min'] = os.getloadavg()[0]
    except Exception as e:
        env['loadavg_1min'] = None
        env['loadavg_error'] = _exc(e)
    env['mem'] = _meminfo()
    # import deps
    # (WHEELS.txt 'NEEDED' rows)
    for name in ('numpy', 'cv2', 'torch', 'torchvision', 'ultralytics', 'tensorrt', 'clip', 'rclpy',
                 'matplotlib', 'yaml', 'PIL', 'filelock', 'requests'):
        mod, err = _try_import(name)
        env[name] = {'importable': mod is not None,
                     'version': getattr(mod, '__version__', None) if mod is not None else None,
                     'file': getattr(mod, '__file__', None) if mod is not None else None,
                     'error': err}
    env['clip_importable'] = env['clip']['importable']
    torch = sys.modules.get('torch')
    if torch is not None:
        try:
            env['torch']['cuda_available'] = bool(torch.cuda.is_available())
            env['torch']['cuda_device'] = (torch.cuda.get_device_name(0)
                                           if env['torch']['cuda_available'] else None)
            env['torch']['default_threads'] = torch.get_num_threads()
        except Exception as e:
            env['torch']['cuda_error'] = _exc(e)
    ul = sys.modules.get('ultralytics')
    env['yoloe_class'] = bool(ul is not None and hasattr(ul, 'YOLOE'))
    env['load_prompt_embeddings'] = bool(env['yoloe_class'] and
                                         hasattr(ul.YOLOE, 'load_prompt_embeddings'))
    env['files'] = {
        'engine': {'path': args.engine, 'exists': os.path.isfile(args.engine)},
        'OPENBOT_YOLOE_WEIGHTS': {'path': os.environ.get('OPENBOT_YOLOE_WEIGHTS'),
                                  'exists': bool(os.environ.get('OPENBOT_YOLOE_WEIGHTS')) and
                                  os.path.isfile(os.environ['OPENBOT_YOLOE_WEIGHTS'])},
        'OPENBOT_YOLOE_BANK': {'path': os.environ.get('OPENBOT_YOLOE_BANK'),
                               'exists': bool(os.environ.get('OPENBOT_YOLOE_BANK')) and
                               os.path.isfile(os.environ['OPENBOT_YOLOE_BANK'])},
        'assets_dir': {'path': args.assets_dir, 'exists': os.path.isdir(args.assets_dir)},
    }
    env['env_vars'] = dict((k, os.environ.get(k)) for k in (
        'LEON_DIR', 'PYTHONPATH', 'OPENBOT_YOLOE', 'OPENBOT_YOLOE_WEIGHTS', 'OPENBOT_YOLOE_BANK',
        'OPENBOT_YOLOE_THREADS', 'OPENBOT_YOLOE_EXTRA_CONF', 'OPENBOT_YOLO_CLASSES',
        'OPENBOT_INSTANCE_IDS', 'ROS_DISTRO', 'CUDA_VISIBLE_DEVICES'))
    return env


def discover_pts(args):
    pts = []
    if args.pt:
        for p in args.pt:
            pts.append(p if os.path.isabs(p) or os.path.exists(p)
                       else os.path.join(args.assets_dir, p))
    else:
        w = os.environ.get('OPENBOT_YOLOE_WEIGHTS')
        if w:
            pts.append(w)
        pts.extend(sorted(glob.glob(os.path.join(args.assets_dir, '*.pt'))))
    seen, out = set(), []
    for p in pts:
        ap = os.path.abspath(p)
        if ap not in seen:
            seen.add(ap)
            out.append(ap)
    return out


def torch_load_report(pts):
    rep = {}
    torch = sys.modules.get('torch')
    for p in pts:
        ent = {'path': p, 'exists': os.path.isfile(p), 'ok': False}
        if not ent['exists']:
            ent['error'] = 'file missing'
            rep[os.path.basename(p)] = ent
            continue
        if torch is None:
            ent['error'] = 'torch not importable'
            rep[os.path.basename(p)] = ent
            continue
        try:
            try:
                ck = torch.load(p, map_location='cpu', weights_only=False)
                ent['weights_only_kwarg'] = True
            except TypeError as e:          # torch <1.13: no weights_only
                if 'weights_only' not in str(e):
                    raise
                ck = torch.load(p, map_location='cpu')
                ent['weights_only_kwarg'] = False
            ent['ok'] = True
            if isinstance(ck, dict):
                ent['keys'] = sorted(str(k) for k in ck.keys())[:20]
                m = ck.get('model') or ck.get('ema')
                ent['model_class'] = type(m).__name__ if m is not None else None
                ent['ckpt_version'] = ck.get('version')
            del ck
        except Exception as e:  # which class is missing
            ent['error'] = _exc(e)
            if isinstance(e, ModuleNotFoundError):
                ent['missing_module'] = getattr(e, 'name', None)
            elif isinstance(e, AttributeError):
                mm = re.search(r"attribute '([^']+)'", str(e))
                ent['missing_attr'] = mm.group(1) if mm else None
        rep[os.path.basename(p)] = ent
    return rep


# -------------------------------------------------------- bank handling
# unused banks
# (--bank / OPENBOT_YOLOE_BANK)
# abspath -> {stem: reason}
# flag only if no checkpoint used it
# (BANK_USED); healthy multi-bank
# tries other model first (:385)
# not a veto; not just vocab
BANK_REJECTED = {}
BANK_USED = set()


def bank_rejected_rows():
    """'<path> (for <stem>): <reason>' for every named bank no checkpoint used."""
    rows = []
    for path in sorted(BANK_REJECTED):
        if path in BANK_USED:
            continue
        for stem in sorted(BANK_REJECTED[path]):
            rows.append('%s (for %s): %s' % (path, stem, BANK_REJECTED[path][stem]))
    return rows


def bank_path_for(pt, args):
    stem = os.path.splitext(os.path.basename(pt))[0]
    explicit = []
    if args.bank:
        explicit.extend(args.bank)
    env_b = os.environ.get('OPENBOT_YOLOE_BANK')
    if env_b:
        explicit.append(env_b)
    cands = explicit + [os.path.join(os.path.dirname(pt), 'vocab_bank_%s.npz' % stem),
                        os.path.join(args.assets_dir, 'vocab_bank_%s.npz' % stem)]
    for i, c in enumerate(cands):
        why = None
        if not os.path.isfile(c):
            why = 'missing'
        else:
            try:
                b = load_bank(c)
                if b['model'] == stem:
                    BANK_USED.add(os.path.abspath(c))
                    return c
                why = 'model mismatch: bank %r vs %r' % (b['model'], stem)
            except Exception as e:
                why = 'load error: %s' % _exc(e)
        # YOLOE text only
        # -pf: no bank
        if i < len(explicit) and stem.startswith('yoloe') and not stem.endswith('-pf'):
            BANK_REJECTED.setdefault(os.path.abspath(c), {})[stem] = why
    return None


def load_bank(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as d:
        prompts = [str(x) for x in d['prompts'].tolist()]
        embeds = np.asarray(d['embeds'], dtype=np.float32)
        model = str(d['model'].item()) if 'model' in d.files else None
        ver = str(d['ultralytics'].item()) if 'ultralytics' in d.files else None
        built = str(d['built'].item()) if 'built' in d.files else None
    if embeds.ndim != 2 or embeds.shape[0] != len(prompts):
        raise ValueError('bank shape %r does not match %d prompts' % (embeds.shape, len(prompts)))
    return {'path': path, 'prompts': prompts, 'embeds': embeds, 'model': model,
            'ultralytics': ver, 'built': built,
            'index': dict((p, i) for i, p in enumerate(prompts))}


def bank_report(pts, args):
    rep = {}
    paths = set()
    for p in pts:
        bp = bank_path_for(p, args)
        if bp:
            paths.add(bp)
    paths.update(glob.glob(os.path.join(args.assets_dir, 'vocab_bank_*.npz')))
    for bp in sorted(paths):
        ent = {'path': bp}
        try:
            b = load_bank(bp)
            ent.update({'model': b['model'], 'n_prompts': len(b['prompts']),
                        'dim': int(b['embeds'].shape[1]), 'ultralytics': b['ultralytics'],
                        'built': b['built'],
                        'base20_present': all(x in b['index'] for x in BASE_20),
                        'missing_base': [x for x in BASE_20 if x not in b['index']],
                        'md5': md5_file(bp)})
        except Exception as e:
            ent['error'] = _exc(e)
        rep[os.path.basename(bp)] = ent
    return rep


def _vocab_tensor(bank, prompts, device):
    import numpy as np
    import torch
    rows = [bank['index'][p] for p in prompts]
    arr = np.ascontiguousarray(bank['embeds'][rows][None, :, :], dtype=np.float32)
    return torch.from_numpy(arr).to(device)


def set_bank_vocab(model, bank, prompts, device):
    """ONLY the two-argument form: set_classes(names, embeddings). Never the encoder."""
    model.set_classes(list(prompts), _vocab_tensor(bank, prompts, device))


# --------------------------------------------------------- image helpers
def load_frame(path):
    """BGR uint8 HxWx3 at 424x240 (cv2), or a synthetic frame (flagged)."""
    import numpy as np
    if path:
        try:
            import cv2
            img = cv2.imread(path)
            if img is None:
                raise IOError('cv2.imread returned None for %s' % path)
            if img.shape[1] != FRAME_W or img.shape[0] != FRAME_H:
                img = cv2.resize(img, (FRAME_W, FRAME_H))
            return img, False
        except Exception as e:
            _log('frame %s unusable (%s) - synthetic frame' % (path, _exc(e)))
    rng = np.random.RandomState(0)
    return rng.randint(0, 255, size=(FRAME_H, FRAME_W, 3)).astype(np.uint8), True


def write_png(path, arr):
    """Save an HxWx3 uint8 array so that cv2.imread(path) returns it byte-exact.

    cv2 when available; else a stdlib (zlib) PNG writer storing channels reversed
    (cv2 decodes PNG RGB into BGR order)."""
    if os.environ.get('OPENBOT_PROBE_NO_CV2') != '1':
        try:
            import cv2
            if cv2.imwrite(path, arr):
                return 'cv2'
        except Exception:
            pass
    h, w = arr.shape[0], arr.shape[1]
    rgb = arr[:, :, ::-1]
    raw = b''.join(b'\x00' + rgb[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return (struct.pack('>I', len(data)) + tag + data +
                struct.pack('>I', zlib.crc32(tag + data) & 0xffffffff))
    png = (b'\x89PNG\r\n\x1a\n' +
           chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)) +
           chunk(b'IDAT', zlib.compress(raw, 6)) + chunk(b'IEND', b''))
    with open(path, 'wb') as f:
        f.write(png)
    return 'zlib'


def image_msg_to_array(msg):
    """sensor_msgs/Image -> HxWx3 uint8 exactly as yolo_node_seg.py:281 sees it
    (raw bytes reshaped, no colour conversion), honouring row padding."""
    import numpy as np
    h, w = int(msg.height), int(msg.width)
    enc = str(getattr(msg, 'encoding', '') or '')
    ch = {'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4, 'mono8': 1}.get(enc, 3)
    step = int(getattr(msg, 'step', 0) or w * ch)
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    arr = buf[:h * step].reshape(h, step)[:, :w * ch].reshape(h, w, ch)
    if ch == 4:
        arr = arr[:, :, :3]
    elif ch == 1:
        arr = np.repeat(arr, 3, axis=2)
    return np.ascontiguousarray(arr), enc


# ------------------------------------------------------------ 2. bench run
def _make_model(kind, path):
    from ultralytics import YOLO
    if kind == 'yoloe':
        from ultralytics import YOLOE
        return YOLOE(path)
    return YOLO(path, task='segment')


def _time_infer(model, img, device, n, warmup, imgsz):
    import torch
    kw = {'conf': GATE_CONF, 'iou': IOU, 'verbose': False, 'device': device}
    if imgsz:
        kw['imgsz'] = imgsz
    for _ in range(warmup):
        model(img, **kw)
    cuda = str(device).startswith('cuda')
    ts = []
    c0, w0 = _cpu_s(), time.time()
    for _ in range(n):
        if cuda:
            torch.cuda.synchronize()
        t = time.perf_counter()
        model(img, **kw)
        if cuda:
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1000.0)
    c1, w1 = _cpu_s(), time.time()
    ts.sort()
    return {'median_ms': round(statistics.median(ts), 2), 'p90_ms': _pct(ts, 0.9),
            'min_ms': round(ts[0], 2), 'n': n, 'warmup': warmup,
            'cpu_pct': round(100.0 * (c1 - c0) / max(1e-6, (w1 - w0)), 1)}


def _bank_swap(model, bank, device, reps):
    base = list(BASE_20)
    extras = [p for p in bank['prompts'] if p not in BASE_20][:4]
    add, restore, first = [], [], None
    for _ in range(reps):
        t = time.perf_counter()
        set_bank_vocab(model, bank, base + extras, device)
        add.append(round((time.perf_counter() - t) * 1000.0, 3))
        t = time.perf_counter()
        set_bank_vocab(model, bank, base, device)
        restore.append(round((time.perf_counter() - t) * 1000.0, 3))
    return {'add_extras': add, 'restore_base': restore, 'extras_used': extras}


def bench(args, pts, frame_img, synthetic, errors):
    import torch
    devices = _parse_list(args.device)
    if 'auto' in devices:
        devices = ['cpu'] + (['cuda:0'] if torch.cuda.is_available() else [])
    threads = _parse_list(args.threads, int)
    out = {'frame': args.frame, 'synthetic': synthetic, 'frame_shape': list(frame_img.shape),
           'devices': devices, 'threads': threads, 'imgsz': args.imgsz or 'default (as node)',
           'conf': GATE_CONF, 'iou': IOU, 'models': {}}
    specs = []
    if os.path.isfile(args.engine):
        specs.append(('engine', 'yolo', args.engine))
        out['engine'] = 'present'
    else:
        out['engine'] = 'missing'
    for p in pts:
        if os.path.isfile(p):
            kind = 'yoloe' if os.path.basename(p).startswith('yoloe') else 'yolo'
            specs.append((os.path.basename(p), kind, p))
    for name, kind, path in specs:
        ment = {'path': path, 'kind': kind, 'per_device': {}}
        out['models'][name] = ment
        for dev in devices:
            dent = {}
            ment['per_device'][dev] = dent
            if dev.startswith('cuda') and not torch.cuda.is_available():
                dent['skipped'] = 'cuda not available'
                continue
            if name == 'engine' and not dev.startswith('cuda'):
                dent['skipped'] = 'TensorRT engine is GPU-only'
                continue
            try:
                rss0 = _rss_mb()
                t0 = time.perf_counter()
                model = _make_model(kind, path)
                dent['load_ms'] = round((time.perf_counter() - t0) * 1000.0, 1)
                bank = None
                if kind == 'yoloe' and not os.path.basename(path).endswith('-pf.pt'):
                    bp = bank_path_for(path, args)
                    if bp:
                        bank = load_bank(bp)
                        missing = [x for x in BASE_20 if x not in bank['index']]
                        if missing:
                            raise ValueError('bank %s lacks base prompts %r' % (bp, missing))
                        set_bank_vocab(model, bank, BASE_20, dev)
                        dent['vocab'] = 'bank base-20 (%s)' % os.path.basename(bp)
                    else:
                        dent['vocab'] = 'checkpoint default (no bank found)'
                dent['rss_mb_before'] = rss0
                # select_device resets threads
                # predictor first
                # under sentinel
                # then set counts
                torch.set_num_threads(1)
                model(frame_img, conf=GATE_CONF, iou=IOU, verbose=False, device=dev)
                dent['torch_threads_after_predictor_setup'] = torch.get_num_threads()
                dent['threads'] = {}
                for th in threads:
                    torch.set_num_threads(th)
                    r = _time_infer(model, frame_img, dev, args.n, args.warmup, args.imgsz)
                    r['threads_effective'] = torch.get_num_threads()
                    dent['threads'][str(th)] = r
                dent['rss_mb_after'] = _rss_mb()
                dent['maxrss_mb'] = _maxrss_mb()
                if bank is not None:
                    dent['bank_swap_ms'] = _bank_swap(model, bank, dev, args.swap_reps)
                    t = time.perf_counter()
                    model(frame_img, conf=GATE_CONF, iou=IOU, verbose=False, device=dev)
                    dent['first_infer_after_swap_ms'] = round((time.perf_counter() - t) * 1000.0, 2)
                elif kind == 'yoloe':
                    dent['bank_swap_ms'] = None
                del model
            except Exception as e:
                dent['error'] = _exc(e)
                errors.append('bench.%s.%s: %s' % (name, dev, _exc(e)))
            finally:
                try:
                    if dev.startswith('cuda'):
                        torch.cuda.empty_cache()
                except Exception:
                    pass
    return out
