import os
import gc
import json
import tempfile
import urllib.request

import cv2
import numpy as np
from flask import Flask, request, jsonify

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

app = Flask(__name__)

MODEL_PATH = "pose_landmarker_lite.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/"
    "pose_landmarker_lite.task"
)

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Descargando modelo MediaPipe lite...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Modelo descargado.")

ensure_model()

def angle_between(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc  = a - b, c - b
    cos_a   = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))

def moving_average(series, window=5):
    if len(series) < window:
        return series
    kernel = np.ones(window) / window
    return np.convolve(series, kernel, mode='same').tolist()

IDX = {
    'left_shoulder':  11, 'right_shoulder': 12,
    'left_elbow':     13, 'right_elbow':    14,
    'left_wrist':     15, 'right_wrist':    16,
    'left_hip':       23, 'right_hip':      24,
    'left_knee':      25, 'right_knee':     26,
    'left_ankle':     27, 'right_ankle':    28,
    'nose':            0,
}

def lm_to_dict(landmarks):
    if not landmarks:
        return None
    return {
        k: {'x': landmarks[i].x, 'y': landmarks[i].y, 'z': landmarks[i].z}
        for k, i in IDX.items()
    }

def smooth_frames(frames):
    for coord in ('x', 'y', 'z'):
        for key in IDX:
            series = [f[key][coord] for f in frames]
            sm = moving_average(series)
            for i, f in enumerate(frames):
                f[key][coord] = sm[i]
    return frames

def resize_frame(frame, max_width=480):
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    return cv2.resize(frame, (max_width, int(h * scale)))

def dominant_arm(frames, handedness=None):
    if handedness in ('right', 'left'):
        return handedness
    min_r = min(f['right_wrist']['y'] for f in frames)
    min_l = min(f['left_wrist']['y']  for f in frames)
    return 'right' if min_r < min_l else 'left'

def find_impact_frame(frames, arm):
    ys = [f[f'{arm}_wrist']['y'] for f in frames]
    return int(np.argmin(ys))

def find_prep_frame(frames, impact_idx, arm):
    search_end = max(1, impact_idx)
    best_idx = 0
    best_angle = 999.0
    for i in range(search_end):
        f = frames[i]
        for side in ('left', 'right'):
            try:
                angle = angle_between(
                    [f[f'{side}_hip']['x'],   f[f'{side}_hip']['y']],
                    [f[f'{side}_knee']['x'],  f[f'{side}_knee']['y']],
                    [f[f'{side}_ankle']['x'], f[f'{side}_ankle']['y']],
                )
                if angle < best_angle:
                    best_angle = angle
                    best_idx = i
            except Exception:
                continue
    print(f"[PrepFrame] idx={best_idx} knee_angle={round(best_angle,1)}")
    return best_idx

def find_followthrough_frame(frames, impact_idx):
    if impact_idx >= len(frames) - 1:
        return len(frames) - 1
    hip_x_at_impact = (
        frames[impact_idx]['left_hip']['x'] +
        frames[impact_idx]['right_hip']['x']
    ) / 2
    best_idx = impact_idx
    best_displacement = 0.0
    for i in range(impact_idx + 1, len(frames)):
        f = frames[i]
        hip_x = (f['left_hip']['x'] + f['right_hip']['x']) / 2
        displacement = abs(hip_x - hip_x_at_impact)
        if displacement > best_displacement:
            best_displacement = displacement
            best_idx = i
    return best_idx

def calc_metrics_remate(frames, impact_idx, prep_idx, follow_idx, arm):
    imp    = frames[impact_idx]
    prep   = frames[prep_idx]
    follow = frames[follow_idx]

    # Extensión del brazo en el codo (frame de impacto)
    arm_angle = angle_between(
        [imp[f'{arm}_shoulder']['x'], imp[f'{arm}_shoulder']['y']],
        [imp[f'{arm}_elbow']['x'],    imp[f'{arm}_elbow']['y']],
        [imp[f'{arm}_wrist']['x'],    imp[f'{arm}_wrist']['y']],
    )

    # Altura de la muñeca sobre la cabeza (frame de impacto)
    # Negativo = muñeca por encima de la nariz (bueno en un remate)
    # y=0 es arriba en MediaPipe, por eso restamos al revés
    wrist_above_head = imp['nose']['y'] - imp[f'{arm}_wrist']['y']

    # Flexión de rodillas (frame de preparación)
    knee_angle = angle_between(
        [prep[f'{arm}_hip']['x'],   prep[f'{arm}_hip']['y']],
        [prep[f'{arm}_knee']['x'],  prep[f'{arm}_knee']['y']],
        [prep[f'{arm}_ankle']['x'], prep[f'{arm}_ankle']['y']],
    )

    # Transferencia de peso (frame de seguimiento)
    hip_x_impact = (imp['left_hip']['x']    + imp['right_hip']['x'])    / 2
    hip_x_follow = (follow['left_hip']['x'] + follow['right_hip']['x']) / 2
    weight_transfer = abs(hip_x_follow - hip_x_impact)

    # Fluidez — cambios de dirección de la muñeca entre prep e impacto
    wrist_y_segment = [f[f'{arm}_wrist']['y'] for f in frames[prep_idx:impact_idx+1]]
    if len(wrist_y_segment) > 2:
        diffs = np.diff(wrist_y_segment)
        direction_changes = int(np.sum(np.diff(np.sign(diffs)) != 0))
        fluidity_score = direction_changes / max(len(wrist_y_segment), 1)
    else:
        fluidity_score = 0.0

    print(f"[Metrics] arm={arm_angle:.1f} wrist_above_head={wrist_above_head:.3f} knee={knee_angle:.1f} weight={weight_transfer:.3f} fluidity={fluidity_score:.3f}")

    return {
        'arm_extension_angle':  round(arm_angle, 1),
        'wrist_above_head':     round(wrist_above_head, 3),
        'knee_flexion_angle':   round(knee_angle, 1),
        'hip_displacement':     round(weight_transfer, 3),
        'fluidity_score':       round(fluidity_score, 3),
        '_phases': {
            'prep_frame':           prep_idx,
            'impact_frame':         impact_idx,
            'followthrough_frame':  follow_idx,
        }
    }

METRIC_MAP = {
    'arm_extension':   ('arm_extension_angle', 'asc'),
    'contact_height':  ('wrist_above_head',    'asc'),
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

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'service': 'padel-mediapipe'})

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

        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=1,
        )

        with mp_vision.PoseLandmarker.create_from_options(options) as detector:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_count += 1
                if frame_count % FRAME_SKIP != 0:
                    continue
                frame  = resize_frame(frame, max_width=480)
                rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = detector.detect(mp_img)
                if result.pose_landmarks:
                    raw.append(lm_to_dict(result.pose_landmarks[0]))
                del frame, rgb, mp_img, result

        cap.release()
        gc.collect()

        frames = [f for f in raw if f is not None]
        del raw
        gc.collect()

        if len(frames) < 5:
            return jsonify({
                'score': 0, 'metrics': {}, 'score_details': {},
                'error': 'Could not detect body pose in video',
            }), 200

        frames     = smooth_frames(frames)
        arm        = dominant_arm(frames, handedness)
        impact_idx = find_impact_frame(frames, arm)
        prep_idx   = find_prep_frame(frames, impact_idx, arm)
        follow_idx = find_followthrough_frame(frames, impact_idx)

        print(f"[Phases] prep={prep_idx} impact={impact_idx} follow={follow_idx} total={len(frames)} arm={arm}")

        metrics        = calc_metrics_remate(frames, impact_idx, prep_idx, follow_idx, arm)
        phases         = metrics.pop('_phases')
        score, details = compute_score(metrics, shot_config)

        effective_fps = fps / FRAME_SKIP

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
