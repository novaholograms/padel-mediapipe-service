import os
import json
import tempfile

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)
mp_pose = mp.solutions.pose


# ─────────────────────────────────────────────
# GEOMETRY
# ─────────────────────────────────────────────

def angle_between(a, b, c):
    """Angle in degrees at vertex b, formed by points a-b-c."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))


def moving_average(series, window=5):
    if len(series) < window:
        return series
    kernel = np.ones(window) / window
    return np.convolve(series, kernel, mode='same').tolist()


# ─────────────────────────────────────────────
# LANDMARK EXTRACTION
# ─────────────────────────────────────────────

KEYS = {
    'nose': 0,
    'left_shoulder': 11,  'right_shoulder': 12,
    'left_elbow': 13,     'right_elbow': 14,
    'left_wrist': 15,     'right_wrist': 16,
    'left_hip': 23,       'right_hip': 24,
    'left_knee': 25,      'right_knee': 26,
    'left_ankle': 27,     'right_ankle': 28,
}


def extract(pose_landmarks):
    if not pose_landmarks:
        return None
    lm = pose_landmarks.landmark
    return {k: {'x': lm[i].x, 'y': lm[i].y, 'z': lm[i].z} for k, i in KEYS.items()}


def smooth_frames(frames):
    """Apply moving-average to every coordinate of every landmark."""
    for coord in ('x', 'y', 'z'):
        for key in KEYS:
            series = [f[key][coord] for f in frames]
            sm = moving_average(series)
            for i, f in enumerate(frames):
                f[key][coord] = sm[i]
    return frames


# ─────────────────────────────────────────────
# SHOT LOGIC
# ─────────────────────────────────────────────

def dominant_arm(frames):
    min_r = min(f['right_wrist']['y'] for f in frames)
    min_l = min(f['left_wrist']['y'] for f in frames)
    return 'right' if min_r < min_l else 'left'


def impact_frame_highest_wrist(frames, arm):
    ys = [f[f'{arm}_wrist']['y'] for f in frames]
    return int(np.argmin(ys))


def calc_metrics_remate(frames, impact_idx, arm, fps):
    imp = frames[impact_idx]

    # 1. Arm extension (shoulder → elbow → wrist angle)
    arm_angle = angle_between(
        [imp[f'{arm}_shoulder']['x'], imp[f'{arm}_shoulder']['y']],
        [imp[f'{arm}_elbow']['x'],    imp[f'{arm}_elbow']['y']],
        [imp[f'{arm}_wrist']['x'],    imp[f'{arm}_wrist']['y']],
    )

    # 2. Contact height (wrist above shoulder, positive = higher)
    contact_height = imp[f'{arm}_shoulder']['y'] - imp[f'{arm}_wrist']['y']

    # 3. Knee flexion
    knee_angle = angle_between(
        [imp[f'{arm}_hip']['x'],   imp[f'{arm}_hip']['y']],
        [imp[f'{arm}_knee']['x'],  imp[f'{arm}_knee']['y']],
        [imp[f'{arm}_ankle']['x'], imp[f'{arm}_ankle']['y']],
    )

    # 4. Weight transfer (hip-centre X displacement: prep → impact)
    prep_idx = max(0, impact_idx - int(fps * 0.8))
    prep = frames[prep_idx]
    hip_prep   = (prep['left_hip']['x']  + prep['right_hip']['x'])  / 2
    hip_impact = (imp['left_hip']['x']   + imp['right_hip']['x'])   / 2
    weight_transfer = abs(hip_impact - hip_prep)

    # 5. Fluidity (std-dev of wrist-Y acceleration — lower = smoother)
    wrist_y = [f[f'{arm}_wrist']['y'] for f in frames]
    accel   = np.diff(np.diff(wrist_y))
    fluidity_variance = float(np.std(accel)) if len(accel) > 1 else 0.1

    return {
        'arm_extension_angle':  round(arm_angle, 1),
        'contact_height_ratio': round(contact_height, 3),
        'knee_flexion_angle':   round(knee_angle, 1),
        'hip_displacement':     round(weight_transfer, 3),
        'fluidity_variance':    round(fluidity_variance, 4),
    }


# ─────────────────────────────────────────────
# SCORING ENGINE
# ─────────────────────────────────────────────

METRIC_MAP = {
    'arm_extension':   ('arm_extension_angle',  'asc'),
    'contact_height':  ('contact_height_ratio', 'asc'),
    'knee_flexion':    ('knee_flexion_angle',    'desc'),
    'weight_transfer': ('hip_displacement',      'asc'),
    'fluidity':        ('fluidity_variance',     'desc'),
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
    return jsonify({'status': 'ok', 'service': 'padel-mediapipe'})


@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files.get('video') or request.files.get('file')
    if not file:
        return jsonify({'error': 'No video file provided'}), 400

    shot_type = request.form.get('shotType', 'Remate')

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
        cap = cv2.VideoCapture(tmp_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        raw = []

        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = pose.process(rgb)
                raw.append(extract(result.pose_landmarks))

        cap.release()

        frames = [f for f in raw if f is not None]
        if len(frames) < 10:
            return jsonify({
                'score': 0, 'metrics': {}, 'score_details': {},
                'error': 'Could not detect body pose in video',
            }), 200

        frames     = smooth_frames(frames)
        arm        = dominant_arm(frames)
        impact_idx = impact_frame_highest_wrist(frames, arm)
        metrics    = calc_metrics_remate(frames, impact_idx, arm, fps)
        score, details = compute_score(metrics, shot_config)

        return jsonify({
            'score':         score,
            'metrics':       metrics,
            'score_details': details,
            'dominant_arm':  arm,
            'impact_frame':  impact_idx,
            'total_frames':  len(frames),
            'fps':           fps,
        })

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(exc)}), 500

    finally:
        os.unlink(tmp_path)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
