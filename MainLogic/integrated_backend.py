from __future__ import annotations
import os
import io
import uuid
import time
import threading
import queue
import traceback
import subprocess
import json
import secrets
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from collections import deque
from io import BytesIO

# Flask & Extensions
from flask import Flask, Blueprint, request, jsonify, abort, send_file, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from dotenv import load_dotenv

# Drowsiness Detection Specifics
import cv2
import numpy as np
import mediapipe as mp
import telegram
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Groq Specifics
from groq import Groq

# Load env
load_dotenv()

# --- App Initialization ----------------------------------------------------
app = Flask(__name__, static_folder='../static')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///smart_locker.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Enable CORS broadly to cover both systems
CORS(app, resources={r"/*": {"origins": "*"}})

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
    return response

app.url_map.strict_slashes = False

db = SQLAlchemy(app)
lockers_bp = Blueprint('lockers', __name__, url_prefix='/api/lockers')

# ---------------------------------------------------------------------------
# GLOBAL STATE & CONFIG (Drowsiness Detection)
# ---------------------------------------------------------------------------
BOT_TOKEN = "8579934462:AAG7ItuNpjkuQ8lqflntPEUATZL4HdhYk5g"
CHAT_ID = "@EARdrowsines_alert"

class AppState:
    def __init__(self):
        self.current_ear = 0.0
        self.threshold = 0.20
        self.drowsy_time_sec = 3.0
        self.consecutive_frames = 0
        self.max_buffer = 15
        self.is_drowsy = False
        self.events = deque(maxlen=100)
        self.start_time = time.time()
        self.telegram_configured = True
        self.last_telegram_time = 0
        self.telegram_cooldown = 60
        
state = AppState()

# Mediapipe FaceLandmarker setup
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'face_landmarker.task')
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1,
    running_mode=vision.RunningMode.IMAGE)
landmarker = vision.FaceLandmarker.create_from_options(options)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# ---------------------------------------------------------------------------
# GLOBAL STATE & CONFIG (Transcribe & LLM)
# ---------------------------------------------------------------------------
BASE_DIR = app.root_path
RAW_UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads', 'raw')
CONVERTED_DIR = os.path.join(BASE_DIR, 'uploads', 'converted')
EXPORT_DIR = os.path.join(BASE_DIR, 'uploads', 'exports')
os.makedirs(RAW_UPLOAD_DIR, exist_ok=True)
os.makedirs(CONVERTED_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

DIRECT_EXTS = {'.mp3', '.wav', '.ogg', '.flac', '.webm'}
jobs: Dict[str, dict] = {}
jobs_lock = threading.Lock()

# GroqChat helper
@dataclass
class GroqChat:
    system_prompt: str
    api_key: Optional[str] = None
    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.0
    max_tokens: int = 1024
    max_history_messages: int = 24
    client: Groq = field(init=False)
    context: List[Dict[str, str]] = field(default_factory=list, init=False)

    def __post_init__(self):
        self.api_key = self.api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            # We don't raise error here to allow the rest of the app to run even if Groq is missing
            self.client = None
            print("WARNING: GROQ_API_KEY not found. LLM features will be disabled.")
        else:
            self.client = Groq(api_key=self.api_key)
            self.reset_context(system_prompt=self.system_prompt)

    def reset_context(self, system_prompt: Optional[str] = None) -> None:
        if system_prompt is not None:
            self.system_prompt = system_prompt
        self.context = [{"role": "system", "content": self.system_prompt}]

    def summarize(self, text: str, style: str = None, model: str = None, temperature: float = None) -> str:
        if not self.client: return "Groq client not initialized."
        prompt = f"Summarize the following text in {style} style:\n\n{text}" if style else f"Summarize short:\n\n{text}"
        try:
            resp = self.client.chat.completions.create(
                model=model or self.model,
                messages=[{"role": "system", "content": self.system_prompt}, {"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=self.temperature if temperature is None else temperature,
            )
            return self._extract_text_from_response(resp)
        except Exception as e: return f"(Summarization failed: {e})"

    def answer_with_context(self, context: str, question: str, model: str = None, temperature: float = None) -> str:
        if not self.client: return "Groq client not initialized."
        combined = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        try:
            resp = self.client.chat.completions.create(
                model=model or self.model,
                messages=[{"role": "system", "content": self.system_prompt}, {"role": "user", "content": combined}],
                max_tokens=512,
                temperature=self.temperature if temperature is None else temperature,
            )
            return self._extract_text_from_response(resp)
        except Exception as e: return f"(QA failed: {e})"

    def _extract_text_from_response(self, resp: Any) -> str:
        try:
            if hasattr(resp, "choices"):
                return "".join([c.message.get("content", "") if isinstance(c.message, dict) else getattr(c.message, "content", "") for c in resp.choices]).strip()
            if hasattr(resp, "content"): return getattr(resp, "content")
        except Exception: pass
        return str(resp)

DEFAULT_SYSTEM_PROMPT = """You are an expert call review assistant...""" # Truncated for brevity but should be full in implementation
try:
    groq_chat = GroqChat(system_prompt=DEFAULT_SYSTEM_PROMPT)
except Exception:
    groq_chat = None

# ---------------------------------------------------------------------------
# HELPERS (Combined)
# ---------------------------------------------------------------------------
def euclidean(p1, p2): return np.linalg.norm(np.array(p1) - np.array(p2))

def compute_ear(landmarks, eye_indices, width, height):
    points = [(int(landmarks[i].x * width), int(landmarks[i].y * height)) for i in eye_indices]
    A, B, C = euclidean(points[1], points[5]), euclidean(points[2], points[4]), euclidean(points[0], points[3])
    return (A + B) / (2.0 * C)

async def send_telegram_alert(message, image_bytes=None):
    try:
        async with telegram.Bot(token=BOT_TOKEN) as bot_client:
            if image_bytes: await bot_client.send_photo(chat_id=CHAT_ID, photo=image_bytes, caption=message)
            else: await bot_client.send_message(chat_id=CHAT_ID, text=message)
        return True
    except Exception as e:
        traceback.print_exc()
        return False

def iso_now(): return datetime.utcnow().isoformat() + "Z"

def format_job_for_frontend(job_entry: dict) -> dict:
    meta = job_entry.get('meta', {})
    audio_name = os.path.basename(meta.get('orig_path', '')) or meta.get('orig_name') or 'unknown'
    return {
        "id": job_entry["job_id"],
        "audioFile": {"id": meta.get("audio_id") or job_entry["job_id"], "name": audio_name, "size": meta.get("orig_size") or 0, "type": meta.get("orig_type") or "audio/*", "file": None},
        "status": job_entry.get("status", "queued"), "progress": meta.get("progress", None), "error": job_entry.get("error"),
        "createdAt": job_entry.get("created_at"), "completedAt": job_entry.get("updated_at") if job_entry.get("status") == "ready" else None
    }

def format_transcription_for_frontend(job_id: str, transcription_text: str, segments_raw: Optional[list] = None, language: str = "en", duration: float = 0.0) -> dict:
    segments = []
    if segments_raw:
        for i, s in enumerate(segments_raw):
            segments.append({"id": s.get("id") or f"seg_{i}", "start": float(s.get("start", 0.0)), "end": float(s.get("end", s.get("start", 0.0))), "text": s.get("text", ""), "speaker": s.get("speaker"), "confidence": float(s.get("confidence", 1.0)) if s.get("confidence") is not None else 1.0})
    else:
        segments = [{"id": f"seg_0", "start": 0.0, "end": float(duration or 0.0), "text": transcription_text or "", "speaker": None, "confidence": 1.0}]
    return {"id": f"trans_{job_id}", "jobId": job_id, "segments": segments, "fullText": transcription_text or "", "language": language}

def transcribe_with_groq(audio_path, model='whisper-large-v3', timeout=120):
    with open(audio_path,'rb') as f: data = f.read()
    client = Groq()
    transcript = client.audio.transcriptions.create(file=(os.path.basename(audio_path), data), model=model, timeout=timeout)
    text = getattr(transcript,'text', '') if not isinstance(transcript, dict) else transcript.get('text')
    segments = getattr(transcript,'segments',None) if not isinstance(transcript, dict) else transcript.get('segments')
    return text, segments

def summarize_text_with_groq(text, style='short', model=None, temperature=0.0):
    global groq_chat
    if groq_chat is None: groq_chat = GroqChat(system_prompt=DEFAULT_SYSTEM_PROMPT)
    return groq_chat.summarize(text, style=style, model=model, temperature=temperature)

def answer_question_with_groq(context_text, question, requirement_text=None, model=None, temperature=0.0):
    global groq_chat
    if groq_chat is None: groq_chat = GroqChat(system_prompt=DEFAULT_SYSTEM_PROMPT)
    if requirement_text: context_text = f"{context_text}\n\nRequirements:\n{requirement_text}"
    return groq_chat.answer_with_context(context_text, question, model=model, temperature=temperature)

# ---------------------------------------------------------------------------
# ENDPOINTS (Drowsiness)
# ---------------------------------------------------------------------------
@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "status": "integrated_server_online",
        "systems": ["Drowsiness Monitoring", "Audio Transcription & QA", "Locker Management"],
        "drowsiness_status": "ok_deployed"
    })

@app.route('/status', methods=['GET'])
def get_status_drowsy():
    uptime = int(time.time() - state.start_time)
    return jsonify({"status": "online", "uptime": uptime, "camera": "connected", "telegram": "configured" if state.telegram_configured else "not_configured"})

@app.route('/ear', methods=['GET'])
def get_ear():
    return jsonify({"ear": round(state.current_ear, 3), "threshold": state.threshold, "timestamp": int(time.time())})

@app.route('/alert', methods=['GET'])
def get_alert():
    return jsonify({"drowsy": state.is_drowsy, "consecutive_frames": state.consecutive_frames, "max_buffer": state.max_buffer})

@app.route('/events', methods=['GET'])
def get_events():
    return jsonify({"events": list(state.events)})

@app.route('/settings', methods=['POST'])
def update_settings():
    try:
        data = request.json
        if 'ear_threshold' in data: state.threshold = float(data['ear_threshold'])
        if 'drowsy_time' in data:
            state.drowsy_time_sec = float(data['drowsy_time'])
            state.max_buffer = max(2, int(state.drowsy_time_sec * 5))
        if 'telegram_cooldown' in data: state.telegram_cooldown = int(data['telegram_cooldown'])
        return jsonify({"success": True, "message": "Settings updated"})
    except Exception as e: return jsonify({"success": False, "error": str(e)}), 400

@app.route('/detect', methods=['POST'])
def detect():
    try:
        img_bytes = BytesIO(request.data).read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None: return jsonify({"error": "invalid image"}), 400
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)
        if not result.face_landmarks: return jsonify({"error": "no face detected"}), 200
        landmarks = result.face_landmarks[0]
        avg_ear = (compute_ear(landmarks, LEFT_EYE, w, h) + compute_ear(landmarks, RIGHT_EYE, w, h)) / 2.0
        state.current_ear = avg_ear
        drowsy = bool(avg_ear < state.threshold)
        state.consecutive_frames = state.consecutive_frames + 1 if drowsy else 0
        was_drowsy, state.is_drowsy = state.is_drowsy, state.consecutive_frames >= state.max_buffer
        now = time.time()
        if state.is_drowsy and (not was_drowsy or (now - state.last_telegram_time >= state.telegram_cooldown)):
            msg = f"🚨 Drowsiness Detected!\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nEAR: {avg_ear:.3f}\nThreshold: {state.threshold:.2f}"
            _, img_encoded = cv2.imencode('.jpg', frame)
            asyncio.run(send_telegram_alert(msg, img_encoded.tobytes()))
            state.last_telegram_time = now
        if state.is_drowsy:
            if not len(state.events) or (datetime.now() - datetime.fromisoformat(state.events[0]['timestamp'])).total_seconds() > 30:
                state.events.appendleft({"timestamp": datetime.now().isoformat(), "ear": round(avg_ear, 3), "alert_sent": True, "type": "critical"})
        return jsonify({"drowsy": drowsy, "confidence": round(avg_ear, 3), "consecutive_frames": state.consecutive_frames, "max_buffer": state.max_buffer})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ---------------------------------------------------------------------------
# ENDPOINTS (Audio & LLM)
# ---------------------------------------------------------------------------
@app.route('/api/v1/audio/upload', methods=['POST'])
def api_upload_audio():
    if 'file' not in request.files: return jsonify({'error':'No file provided'}),400
    f = request.files['file']
    fn = f.filename
    ext = os.path.splitext(fn)[1].lower()
    raw_path = os.path.join(RAW_UPLOAD_DIR, f"{uuid.uuid4().hex}{ext}")
    f.save(raw_path)
    job_id = uuid.uuid4().hex
    jobs[job_id] = {'job_id': job_id, 'type':'transcription', 'status':'queued', 'created_at':iso_now(), 'updated_at':None, 'meta':{'audio_path': raw_path, 'orig_path': raw_path, 'orig_name': fn, 'orig_size': os.path.getsize(raw_path), 'duration':0.0}, 'result': None, 'error': None}
    try:
        text, segments = transcribe_with_groq(raw_path)
        jobs[job_id].update({'result': {'transcription': text, 'segments': segments}, 'status': 'ready', 'updated_at': iso_now()})
    except Exception as e:
        jobs[job_id].update({'status': 'failed', 'error': str(e), 'updated_at': iso_now()})
    finally:
        if os.path.exists(raw_path): os.remove(raw_path)
    return jsonify({'job_id': job_id, 'status': jobs[job_id]['status'], 'job': format_job_for_frontend(jobs[job_id])}), 201

@app.route('/api/v1/jobs/<job_id>/transcription', methods=['GET'])
def api_get_transcription(job_id):
    job = jobs.get(job_id)
    if not job or job['type'] != 'transcription': return jsonify({'error':'Not found or incorrect job type'}), (404 if not job else 400)
    if job['status'] != 'ready': return jsonify({'status': job['status']}), 202
    return jsonify(format_transcription_for_frontend(job_id, job['result'].get('transcription',''), job['result'].get('segments'), duration=job['meta'].get('duration',0.0))), 200

@app.route('/api/v1/jobs/<job_id>/summary', methods=['POST'])
def api_create_summary(job_id):
    parent = jobs.get(job_id)
    if not parent: return jsonify({'error': 'Not found'}), 404
    data = request.get_json() or {}
    text = parent.get('result', {}).get('transcription') or data.get('text')
    if not text: return jsonify({'error': 'No text available'}), 400
    sub_id = uuid.uuid4().hex
    jobs[sub_id] = {'job_id': sub_id, 'type': 'summary', 'status': 'queued', 'created_at': iso_now(), 'updated_at': None, 'meta': {'parent_job_id': job_id, 'text': text, 'style': data.get('style', 'short')}, 'result': None, 'error': None}
    try:
        summary = summarize_text_with_groq(text, style=data.get('style', 'short'))
        jobs[sub_id].update({'result': {'summary': summary}, 'status': 'ready', 'updated_at': iso_now()})
    except Exception as e: jobs[sub_id].update({'status': 'failed', 'error': str(e), 'updated_at': iso_now()})
    return jsonify({'summary_job_id': sub_id, 'status': jobs[sub_id]['status'], 'summary': jobs[sub_id].get('result', {}).get('summary'), 'job': format_job_for_frontend(jobs[sub_id])}), 201

@app.route('/api/v1/jobs/<job_id>/status', methods=['GET'])
def api_job_status(job_id):
    job = jobs.get(job_id)
    if not job: return jsonify({'error': 'Not found'}), 404
    return jsonify(format_job_for_frontend(job)), 200

@app.route('/api/v1/jobs/<job_id>/qa', methods=['POST'])
def api_create_qa(job_id):
    job = jobs.get(job_id)
    if not job: return jsonify({'error': 'Not found'}), 404
    data = request.get_json() or {}
    question = data.get('question')
    context = data.get('context') or job.get('result', {}).get('transcription', '')
    if not question: return jsonify({'error': 'No question provided'}), 400
    qa_id = uuid.uuid4().hex
    jobs[qa_id] = {'job_id': qa_id, 'type': 'qa', 'status': 'queued', 'created_at': iso_now(), 'updated_at': None, 'meta': {'context_text': context, 'question': question}, 'result': None, 'error': None}
    try:
        answer = answer_question_with_groq(context, question)
        jobs[qa_id].update({'result': {'answer': answer}, 'status': 'ready', 'updated_at': iso_now()})
    except Exception as e: jobs[qa_id].update({'status': 'failed', 'error': str(e), 'updated_at': iso_now()})
    return jsonify({'qa_job_id': qa_id, 'status': jobs[qa_id]['status'], 'answer': jobs[qa_id].get('result', {}).get('answer'), 'job': format_job_for_frontend(jobs[qa_id])}), 201

@app.route('/api/v1/jobs/<job_id>/download', methods=['GET'])
def api_download(job_id):
    job = jobs.get(job_id)
    if not job or job['status'] != 'ready': return jsonify({'error': 'Not found or not ready'}), 404
    text = job['result'].get('transcription') or job['result'].get('summary') or job['result'].get('answer') or ""
    return send_file(io.BytesIO(text.encode('utf-8')), as_attachment=True, download_name=f"{job_id}.txt")

@app.route('/api/v1/jobs', methods=['GET'])
def api_list_jobs():
    return jsonify([format_job_for_frontend(jobs[k]) for k in jobs]), 200

# ---------------------------------------------------------------------------
# MODELS & LOCKER BLUEPRINT
# ---------------------------------------------------------------------------
class Locker(db.Model):
    __tablename__ = 'lockers'
    id = db.Column(db.String, primary_key=True)
    status = db.Column(db.Enum('open', 'closed', name='status_enum'), default='closed')
    otp = db.Column(db.String(6), nullable=True)
    otp_expires = db.Column(db.DateTime, nullable=True)
    last_activity = db.Column(db.DateTime, default=datetime.utcnow)

class Activity(db.Model):
    __tablename__ = 'activities'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    locker_id = db.Column(db.String, db.ForeignKey('lockers.id'), nullable=False)
    type = db.Column(db.String, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    detail = db.Column(db.JSON, nullable=True)

@lockers_bp.route('/', methods=['POST'])
def register_locker():
    data = request.get_json() or {}
    locker_id = data.get('locker_id')
    if not locker_id: return jsonify({'error': 'locker_id required'}), 400
    if Locker.query.get(locker_id): return jsonify({'error': 'Locker already exists'}), 409
    l = Locker(id=locker_id, status='closed', last_activity=datetime.utcnow())
    db.session.add(l); db.session.commit()
    return jsonify({'locker_id': locker_id, 'status': 'closed'}), 201

@lockers_bp.route('/', methods=['GET'])
def list_lockers():
    lockers = Locker.query.order_by(Locker.id).all()
    return jsonify([{'locker_id': l.id, 'status': l.status, 'current_password': l.otp, 'last_activity': l.last_activity.isoformat() + 'Z'} for l in lockers])

@lockers_bp.route('/<locker_id>/status', methods=['GET'])
def get_locker_status(locker_id):
    l = Locker.query.get_or_404(locker_id)
    return jsonify({'locker_id': l.id, 'status': l.status, 'current_password': l.otp, 'last_activity': l.last_activity.isoformat() + 'Z'})

@lockers_bp.route('/<locker_id>/status', methods=['POST'])
def post_locker_status(locker_id):
    data = request.get_json() or {}
    l = Locker.query.get_or_404(locker_id)
    l.status = data.get('status')
    l.last_activity = datetime.utcnow()
    db.session.add(Activity(locker_id=locker_id, type='status_update', detail={'status': l.status}))
    db.session.commit()
    return ('', 204)

@lockers_bp.route('/<locker_id>/otp', methods=['GET'])
def get_otp(locker_id):
    l = Locker.query.get_or_404(locker_id)
    return jsonify({'locker_id': l.id, 'otp': l.otp, 'expires_at': l.otp_expires.isoformat() + 'Z' if l.otp_expires else None})

@lockers_bp.route('/<locker_id>/otp', methods=['POST'])
def generate_otp(locker_id):
    l = Locker.query.get_or_404(locker_id)
    otp = f"{secrets.randbelow(10**4):04}"
    l.otp = otp
    l.otp_expires = datetime.utcnow() + timedelta(minutes=15)
    db.session.add(Activity(locker_id=locker_id, type='otp_generated'))
    db.session.commit()
    return jsonify({'locker_id': l.id, 'otp': otp, 'expires_at': l.otp_expires.isoformat() + 'Z'}), 201

@lockers_bp.route('/<locker_id>/activity', methods=['POST'])
def post_activity(locker_id):
    data = request.get_json() or {}
    db.session.add(Activity(locker_id=locker_id, type=data.get('type'), detail=data.get('detail', {})))
    db.session.commit()
    return ('', 204)

@lockers_bp.route('/<locker_id>/verify-otp', methods=['POST'])
def verify_otp(locker_id):
    data = request.get_json() or {}; entered = data.get('otp'); l = Locker.query.get_or_404(locker_id)
    if l.otp and entered == l.otp and datetime.utcnow() < l.otp_expires:
        db.session.add(Activity(locker_id=locker_id, type='otp_used', detail={'success': True})); db.session.commit()
        return jsonify({'status': 'success', 'message': 'OTP verified'})
    return jsonify({'status': 'fail', 'message': 'Invalid or expired OTP'}), 400

app.register_blueprint(lockers_bp)

with app.app_context():
    db.create_all()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

