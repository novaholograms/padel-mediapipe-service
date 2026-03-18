import os
import gc
import json
import tempfile
import urllib.request

import cv2
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# ─────────────────────────────────────────────
# MODELO YOLOV8 POSE (entrenado en pádel)
# ─────────────────────────────────────────────

MODEL_PATH = "players_keypoints.pt"
MODEL_URL  = "https://huggingface.co/calvos/padel-pose-model/resolve/main/best.pt"

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Descargando modelo YOLOv8 pose pádel...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Modelo descargado.")

ensure_model()

from ultralytics import YOLO
pose_model = YOLO(MODEL_PATH)
print("Modelo YOLOv8 pose cargado.")

# ─────────────────────────────────────────────
# KEYPOINTS (13 puntos formato pádel)
# ─────────────────────────────────────────────

KP = {
    'left_foot':      0,
    'right_foot':     1,
    'torso':          2,
    'right_shoulder': 3,
    'left_shoulder':  4,
    'head':           5,
    'neck':           6,
    'left_hand':      7,
    'right_hand':     8,
    'right_knee':     9,
    'left_knee':      10,
    'right_elbow':    11,
    'left_elbow':     12,
}

def moving_average(series, window=5):
    if len(series) < window:
        return series
    kernel = np.ones(window) / window
    return np.convolve(series, kernel, mode='same').tolist()

def angle_between(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc  = a - b, c - b
    cos_a   = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))

def resize_frame(frame, max_width=640):
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    return cv2.resize(frame, (max_width, int(h * scale)))

# ─────────────────────────────────────────────
# EXTRAER LANDMARKS DE UN FRAME
# ─────────────────────────────────────────────

def extract_landmarks(frame):
    results = pose_model(frame, verbose=False)
    if not results or results[0].keypoints is None:
        return None
    kps = results[0].keypoints
    if kps.xy is None or len(kps.xy) == 0:
        return None

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None

    best_idx = 0
    if len(boxes) > 1:
        areas = []
        for box in boxes.xyxy:
            w = float(box[2] - box[0])
            h = float(box[3] - box[1])
            areas.append(w * h)
        best_idx = int(np.argmax(areas))

    h, w = frame.shape[:2]
    xy   = kps.xy[best_idx].cpu().numpy()
    conf = kps.conf[best_idx].cpu().numpy() if kps.conf is not None else np.ones(len(xy))

    landmarks = {}
    for name, idx in KP.items():
        if idx < len(xy):
            landmarks[name] = {
                'x': float(xy[idx][0]) / w,
                'y': float(xy[idx][1]) / h,
                'v': float(conf[idx]),
            }
    return landmarks

# ─────────────────────────────────────────────
# SMOOTH
# ─────────────────────────────────────────────

def smooth_frames(frames):
    for coord in ('x', 'y'):
        for key in KP:
            series = [f[key][coord] for f in frames if key in f]
            if len(series) < 2:
                continue
            sm = moving_average(series)
            j = 0
            for f in frames:
                if key in f:
                    f[key][coord] = sm[j]
                    j += 1
    return frames

# ─────────────────────────────────────────────
# DETECCIÓN DE FASES
# ─────────────────────────────────────────────

def dominant_arm(frames, handedness='right'):
    return handedness

def find_impact_frame(frames, arm):
    total = len(frames)
    start = max(1, int(total * 0.15))
    end   = max(start + 1, int(total * 0.90))
    hand_key = f'{arm}_hand'
    best_idx = start
    best_y   = 1.0
    for i in range(start, end):
        f = frames[i]
        if hand_key not in f:
            continue
        hand = f[hand_key]
        if hand.get('v', 1.0) < 0.3:
            continue
        if hand['y'] < best_y:
            best_y   = hand['y']
            best_idx = i
    return best_idx

def find_prep_frame(frames, impact_idx, effective_fps):
    offset = max(1, round(effective_fps * 0.4))
    return max(0, impact_idx - offset)

def find_followthrough_frame(frames, impact_idx, effective_fps):
    offset = max(1, round(effective_fps * 0.35))
    return min(len(frames) - 1, impact_idx + offset)

# ─────────────────────────────────────────────
# MÉTRICAS
# ─────────────────────────────────────────────

def calc_metrics_remate(frames, impact_idx, prep_idx, follow_idx, arm):
    imp    = frames[impact_idx]
    prep   = frames[prep_idx]
    follow = frames[follow_idx]

    arm_angle = angle_between(
        [imp[f'{arm}_shoulder']['x'], imp[f'{arm}_shoulder']['y']],
        [imp[f'{arm}_elbow']['x'],    imp[f'{arm}_elbow']['y']],
        [imp[f'{arm}_hand']['x'],     imp[f'{arm}_hand']['y']],
    )

    knee_angle = angle_between(
        [prep['torso']['x'],       prep['torso']['y']],
        [prep[f'{arm}_knee']['x'], prep[f'{arm}_knee']['y']],
        [prep[f'{arm}_foot']['x'], prep[f'{arm}_foot']['y']],
    )

    torso_x_impact  = imp['torso']['x']
    torso_x_follow  = follow['torso']['x']
    weight_transfer = abs(torso_x_follow - torso_x_impact)

    hand_key = f'{arm}_hand'
    wrist_y_segment = [f[hand_key]['y'] for f in frames[prep_idx:impact_idx+1] if hand_key in f]
    if len(wrist_y_segment) > 2:
        diffs = np.diff(wrist_y_segment)
        direction_changes = int(np.sum(np.diff(np.sign(diffs)) != 0))
        fluidity_score = direction_changes / max(len(wrist_y_segment), 1)
    else:
        fluidity_score = 0.0

    print(f"[YOLOv8] Análisis completado ✅")
    print(f"[Metrics] arm_extension={arm_angle:.1f}° knee_flexion={knee_angle:.1f}° weight_transfer={weight_transfer:.3f} fluidity={fluidity_score:.3f}")

    def frame_landmarks(f):
        return {k: {'x': round(f[k]['x'], 4), 'y': round(f[k]['y'], 4)} for k in KP if k in f}

    return {
        'arm_extension_angle': round(arm_angle, 1),
        'knee_flexion_angle':  round(knee_angle, 1),
        'hip_displacement':    round(weight_transfer, 3),
        'fluidity_score':      round(fluidity_score, 3),
        '_phases': {
            'prep_frame':          prep_idx,
            'impact_frame':        impact_idx,
            'followthrough_frame': follow_idx,
        },
        '_landmarks': {
            'prep':          frame_landmarks(prep),
            'impact':        frame_landmarks(imp),
            'followthrough': frame_landmarks(follow),
        }
    }

# ─────────────────────────────────────────────
# SCORING
# ─────────────────────────────────────────────

METRIC_MAP = {
    'arm_extension':   ('arm_extension_angle', 'asc'),
    'knee_flexion':    ('knee_flexion_angle',  'desc'),
    'weight_transfer': ('hip_displacement',    'asc'),
    'fluidity':        ('fluidity_score',      'desc'),
}

def score_asc(value, ranges, weight):
    for r in ranges:
        if value >= r['min']:
            return weight * r['multiplier']
    return 0.0

def score_desc(value, ranges, weight):
    for r in ranges:
        if value <= r['max']:
            return weight * r['multiplier']
    return 0.0

def compute_score(metrics, shot_config):
    scoring = shot_config['scoring']
    total, details = 0.0, {}
    for key, (metric_key, direction) in METRIC_MAP.items():
        if key not in scoring:
            continue
        cfg   = scoring[key]
        value = metrics.get(metric_key, 0)
        pts   = score_asc(value, cfg['ranges'], cfg['weight']) \
                if direction == 'asc' \
                else score_desc(value, cfg['ranges'], cfg['weight'])
        total += pts
        details[key] = {
            'value': value,
            'score': round(pts, 2),
            'max':   cfg['weight'],
            'label': cfg.get('label', key),
        }
    return round(total), details

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'service': 'padel-yolo-pose'})

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files.get('video') or request.files.get('file')
    if not file:
        return jsonify({'error': 'No video file provided'}), 400

    shot_type  = request.form.get('shotType', 'Remate')
    handedness = request.form.get('handedness', 'right')

    with open('shots_config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)

    shot_config = config.get(shot_type)
    if not shot_config:
        return jsonify({'error': f'Shot type not configured: {shot_type}'}), 400

    suffix = os.path.splitext(file.filename or '.mp4')[1] or '.mp4'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        cap         = cv2.VideoCapture(tmp_path)
        fps         = cap.get(cv2.CAP_PROP_FPS) or 30.0
        raw         = []
        frame_count = 0
        FRAME_SKIP  = 3

        print(f"[YOLOv8] Iniciando análisis: shotType={shot_type} handedness={handedness}")

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_count += 1
            if frame_count % FRAME_SKIP != 0:
                continue
            frame = resize_frame(frame, max_width=640)
            lm    = extract_landmarks(frame)
            if lm is not None:
                raw.append(lm)
            del frame

        cap.release()
        gc.collect()

        print(f"[YOLOv8] Frames procesados: {frame_count} total, {len(raw)} con pose detectada")

        frames = [f for f in raw if f is not None]
        del raw
        gc.collect()

        if len(frames) < 5:
            return jsonify({
                'score': 0, 'metrics': {}, 'score_details': {},
                'error': 'Could not detect body pose in video',
            }), 200

        frames        = smooth_frames(frames)
        arm           = dominant_arm(frames, handedness)
        effective_fps = fps / FRAME_SKIP
        impact_idx    = find_impact_frame(frames, arm)
        prep_idx      = find_prep_frame(frames, impact_idx, effective_fps)
        follow_idx    = find_followthrough_frame(frames, impact_idx, effective_fps)

        print(f"[Phases] prep={prep_idx} impact={impact_idx} follow={follow_idx} total={len(frames)} arm={arm}")

        metrics        = calc_metrics_remate(frames, impact_idx, prep_idx, follow_idx, arm)
        phases         = metrics.pop('_phases')
        landmarks      = metrics.pop('_landmarks')
        score, details = compute_score(metrics, shot_config)

        print(f"[Score] final={score}/100")

        return jsonify({
            'score':         score,
            'metrics':       metrics,
            'score_details': details,
            'dominant_arm':  arm,
            'phases': {
                'prep_frame':           phases['prep_frame'],
                'prep_second':          round(phases['prep_frame'] / effective_fps, 2),
                'impact_frame':         phases['impact_frame'],
                'impact_second':        round(phases['impact_frame'] / effective_fps, 2),
                'followthrough_frame':  phases['followthrough_frame'],
                'followthrough_second': round(phases['followthrough_frame'] / effective_fps, 2),
            },
            'landmarks':     landmarks,
            'total_frames':  len(frames),
            'effective_fps': round(effective_fps, 1),
        })

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(exc)}), 500

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        gc.collect()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
