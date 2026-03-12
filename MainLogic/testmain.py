from __future__ import annotations
"""
Integrated Flask backend for audio transcription + summary + QA using Groq.

This file includes:
- A refined `GroqChat` helper class (sync + streaming) for LLM interactions.
- An in-memory job queue + worker to process transcription, summary, and QA jobs.
- Audio upload and conversion (moviepy / ffmpeg fallback).
- Transcription using Groq audio.transcriptions (template call).
- Summary and QA wired to use the `GroqChat` helper.
- Helpers to format responses to match frontend TypeScript interfaces (Job, Transcription, etc.).

Notes:
- Replace model names, SDK call shapes, and any Groq-specific options to match your installed Groq SDK.
- This is a development skeleton. For production, persist jobs/results to a DB, secure uploads, and add authentication.
"""
from flask import Flask, Blueprint, request, jsonify, abort, send_file, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
import secrets
import os
import io
import uuid
import time
import threading
import queue
import traceback
import subprocess
import json
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from flask_cors import CORS
from dotenv import load_dotenv

# Optional PDF parsing
try:
    import PyPDF2
except Exception:
    PyPDF2 = None

# Groq SDK
from groq import Groq

# Load env
if "GROQ_API_KEY" not in os.environ:
    load_dotenv()


# --- App & Extensions Initialization --------------------------------------
app = Flask(__name__)
# Use SQLite for local testing; switch to PostgreSQL in production
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///smart_locker.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
CORS(app, resources={r"/api/*": {"origins": "*"}})
db = SQLAlchemy(app)
lockers_bp = Blueprint('lockers', __name__, url_prefix='/api/lockers')

# -------------------------
# GroqChat helper (refined)
# -------------------------
@dataclass
class GroqChat:
    """Groq chat helper with context management and streaming support."""
    system_prompt: str
    api_key: Optional[str] = None
    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.0
    max_tokens: int = 1024
    max_history_messages: int = 24
    client: Groq = field(init=False)
    context: List[Dict[str, str]] = field(default_factory=list, init=False)

    def __post_init__(self):
        # prefer provided api_key, otherwise rely on env
        self.api_key = self.api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY not found in environment or passed to constructor")
        # instantiate client (SDK may auto-read env)
        self.client = Groq()
        self.reset_context(system_prompt=self.system_prompt)

    def reset_context(self, system_prompt: Optional[str] = None) -> None:
        if system_prompt is not None:
            self.system_prompt = system_prompt
        self.context = [{"role": "system", "content": self.system_prompt}]

    def summarize(self, text: str, style: str = None, model: str = None, temperature: float = None) -> str:
        """Summarize given text using the LLM, with optional style control."""

        if style:
            prompt = f"Summarize the following text in {style} style:\n\n{text}"
        else:
            prompt = f"Summarize the following text concisly and very short:\n\n{text}"

        try:
            resp = self.client.chat.completions.create(
                model=model or self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=512,
                temperature=self.temperature if temperature is None else temperature,
            )
            return self._extract_text_from_response(resp)

        except Exception as e:
            print(f"Summarization failed: {e}")
            return f"(Summarization failed: {e})"

    def answer_with_context(self, context: str, question: str, model: str = None, temperature: float = None) -> str:
        """Answer a question given supporting context using the instance's system prompt."""
        combined = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        try:
            # Use the GroqChat's own system prompt so QA follows your rules.
            resp = self.client.chat.completions.create(
                model=model or self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": combined}
                ],
                max_tokens=512,
                temperature=self.temperature if temperature is None else temperature,
            )
            return self._extract_text_from_response(resp)
        except Exception as e:
            print(f"Context QA failed: {e}")
            return f"(QA failed: {e})"

    def _extract_text_from_response(self, resp: Any) -> str:
        try:
            # Handle SDK objects with 'choices'
            if hasattr(resp, "choices"):
                parts = []
                for c in resp.choices:
                    if getattr(c, "message", None):
                        parts.append(
                            c.message.get("content", "")
                            if isinstance(c.message, dict)
                            else getattr(c.message, "content", str(c.message))
                        )
                    elif getattr(c, "delta", None):
                        parts.append(getattr(c.delta, "content", "") or "")
                    else:
                        parts.append(str(c))
                out = "".join(parts).strip()
                if out:
                    return out

            # Handle plain dicts
            if isinstance(resp, dict):
                out = (
                        resp.get("output_text")
                        or resp.get("text")
                        or resp.get("output", {}).get("text")
                )
                if out:
                    return out

            # Handle message objects like ChatCompletionMessage
            if hasattr(resp, "content"):
                return getattr(resp, "content")

            # Fallback: objects with .text
            if hasattr(resp, "text"):
                return getattr(resp, "text") or str(resp)

        except Exception:
            pass
        return str(resp)

# -------------------------
# Directories
# -------------------------
BASE_DIR = app.root_path
RAW_UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads', 'raw')
CONVERTED_DIR = os.path.join(BASE_DIR, 'uploads', 'converted')
EXPORT_DIR = os.path.join(BASE_DIR, 'uploads', 'exports')
os.makedirs(RAW_UPLOAD_DIR, exist_ok=True)
os.makedirs(CONVERTED_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

DIRECT_EXTS = {'.mp3', '.wav', '.ogg', '.flac', '.webm'}

# -------------------------
# In-memory job store
# -------------------------
jobs: Dict[str, dict] = {}
jobs_lock = threading.Lock()
job_queue = queue.Queue()

def create_job_entry(job_type, meta=None):
    job_id = uuid.uuid4().hex
    entry = {
        'job_id': job_id,
        'type': job_type,
        'status': 'queued',
        'created_at': datetime.utcnow().isoformat() + 'Z',
        'updated_at': None,
        'meta': meta or {},
        'result': None,
        'error': None
    }
    with jobs_lock:
        jobs[job_id] = entry
    return job_id


# Instantiate a shared GroqChat used by summary/QA jobs
DEFAULT_SYSTEM_PROMPT = (
    """
    You are an expert call review assistant for dealership call analysis.
    You work with some predefined category systems, and you will also answer questions based on this categories selected by the user
    that are specific to one chosen system make sure to always answer based on a catergory a user would specify, if possibe before any
    conversation ask for the category, then proceed to answeering the questions based on the category rules.

    ---

    ### 1. Service Setter Booked Categories
    - Booked Appointment – Specific time within 1 hour
    - Booked Appointment – Loose time over 1 hour
    - Appointment Inquiry – No appointment booked
    - Service Inquiry – No appointment discussion
    - Already Scheduled Appointment
    - Vehicle Already in Service
    - Not an Appointment Opportunity
    - Never Connected to a Qualified Agent
    - Unfamiliar Language

    ---

    ### 2. Dealership Discussion Categories
    - Opportunity – Buy/Sell/Lease/Trade
    - Service / Parts / Collision / Body Shop
    - General
    - Correction – Never Connected to a Qualified Agent
    - Unfamiliar Language

    ---

    ### How You Work
    1. **Always keep both category systems in memory.**
    2. If asked to *classify a transcript*, use the requested system’s categories.
       - Example: “Classify with Service Setter Booked” → only use that list.
       - Example: “Classify with Dealership Discussion” → only use that list.
       - If no system is specified → return results for both systems.
    3. If asked a **question about a specific system**, use ONLY the categories from that system
       to answer.
       - Example: “In Service Setter Booked, how do we handle callers who only ask about service pricing?”
       - Example: “In Dealership Discussion, what category fits a caller asking about trading in their car?”
    4. Always respond with:
       - **Direct Answer**: the classification or the specific answer to the question.
       - **Reason**: a brief explanation of how you derived it.

    ---

    ### Examples

    Transcript Example:
    Caller: “Hi, I’d like to book an oil change tomorrow.”
    Agent: “We can do 10 AM.”
    Caller: “Yes, please.”

    Answer:
    - Service Setter Booked: Booked Appointment – Specific time within 1 hour
    - Dealership Discussion: Service / Parts / Collision / Body Shop
    - Reason: Caller scheduled a service appointment for a specific time.

    ---

    Q&A Example:
    User: use the Service Setter Booked
    User: what if the customer asks about parts but doesn’t book?”
    Answer:  No appointment discussion note answer based on the category specified
    Reason: Caller only inquired without booking an appointment.

    ---

    Remember:
    - You classify transcripts into categories and hold that in memory.
    - You answer category-specific questions when requested.
    - You default to BOTH systems if no system is specified.
    """
)
try:
    groq_chat = GroqChat(system_prompt=DEFAULT_SYSTEM_PROMPT)
except Exception:
    groq_chat = None


def iso_now():
    return datetime.utcnow().isoformat() + "Z"


def format_job_for_frontend(job_entry: dict) -> dict:
    """Return Job object matching frontend `Job` interface."""
    meta = job_entry.get('meta', {})
    audio_name = os.path.basename(meta.get('orig_path', '')) or meta.get('orig_name') or 'unknown'
    return {
        "id": job_entry["job_id"],
        "audioFile": {
            "id": meta.get("audio_id") or job_entry["job_id"],
            "name": audio_name,
            "size": meta.get("orig_size") or 0,
            "type": meta.get("orig_type") or "audio/*",
            "file": None,
        },
        "status": job_entry.get("status", "queued"),
        "progress": job_entry.get("meta", {}).get("progress", None),
        "error": job_entry.get("error"),
        "createdAt": job_entry.get("created_at"),
        "completedAt": job_entry.get("updated_at") if job_entry.get("status") == "ready" else None
    }

def format_transcription_for_frontend(job_id: str, transcription_text: str, segments_raw: Optional[list] = None,
                                      language: str = "en", duration: float = 0.0) -> dict:
    """
    Return a Transcription dict matching the TS interface.
    """
    segments = []
    if segments_raw:
        for i, s in enumerate(segments_raw):
            segments.append({
                "id": s.get("id") or f"seg_{i}",
                "start": float(s.get("start", 0.0)),
                "end": float(s.get("end", s.get("start", 0.0))),
                "text": s.get("text", ""),
                "speaker": s.get("speaker"),
                "confidence": float(s.get("confidence", 1.0)) if s.get("confidence") is not None else 1.0
            })
    else:
        segments = [{
            "id": f"seg_0",
            "start": 0.0,
            "end": float(duration or 0.0),
            "text": transcription_text or "",
            "speaker": None,
            "confidence": 1.0
        }]

    return {
        "id": f"trans_{job_id}",
        "jobId": job_id,
        "segments": segments,
        "fullText": transcription_text or "",
        "language": language
    }


# -------------------------
# File handling
# -------------------------
def save_uploaded_file(f, dest_dir, allowed_exts=None):
    fn = f.filename
    ext = os.path.splitext(fn)[1].lower()
    if allowed_exts and ext not in allowed_exts:
        raise ValueError('Unsupported file type')
    unique = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex}{ext}"
    path = os.path.join(dest_dir, unique)
    f.save(path)
    return path

def convert_to_mp3(src_path):
    ts = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    out = os.path.join(CONVERTED_DIR, f"{ts}_{uuid.uuid4().hex}.mp3")
    try:
        audio_clip = AudioFileClip(src_path)
        duration = float(audio_clip.duration)
        audio_clip.write_audiofile(out, logger=None)
        audio_clip.close()
        return out, duration
    except Exception:
        import subprocess
        cmd = ['ffmpeg','-y','-i',src_path,'-vn','-acodec','libmp3lame',out]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            stderr = result.stderr.decode('utf-8', errors='ignore')
            raise RuntimeError(f"ffmpeg conversion failed: {stderr}")
        return out, 0.0

def transcribe_with_groq(audio_path, model='whisper-large-v3', timeout=120):
    with open(audio_path,'rb') as f:
        data=f.read()
    client=Groq()
    transcript=client.audio.transcriptions.create(file=(os.path.basename(audio_path),data),model=model,timeout=timeout)
    text = getattr(transcript,'text', '') if not isinstance(transcript, dict) else transcript.get('text')
    segments = getattr(transcript,'segments',None) if not isinstance(transcript, dict) else transcript.get('segments')
    return text, segments


def summarize_text_with_groq(text, style='short', model=None, temperature=0.0):
    global groq_chat
    if groq_chat is None:
        groq_chat = GroqChat(system_prompt=DEFAULT_SYSTEM_PROMPT)
    return groq_chat.summarize(text, style=style, model=model, temperature=temperature)


def answer_question_with_groq(context_text, question, requirement_text=None, model=None, temperature=0.0):
    global groq_chat
    if groq_chat is None:
        groq_chat = GroqChat(system_prompt=DEFAULT_SYSTEM_PROMPT)

    # If there's an uploaded requirements file or extra instructions, append it to the context
    if requirement_text:
        context_text = f"{context_text}\n\nRequirements:\n{requirement_text}"

    # Correct parameter order: context first, question second
    return groq_chat.answer_with_context(
        context_text,
        question,
        model=model,
        temperature=temperature)


# -------------------------
# Routes
# -------------------------
@app.route('/api/v1/audio/upload', methods=['POST'])
def api_upload_audio():
    if 'file' not in request.files:
        return jsonify({'error':'No file provided'}),400
    f = request.files['file']
    try:
        raw_path = save_uploaded_file(f, RAW_UPLOAD_DIR)
    except Exception as e:
        return jsonify({'error': str(e)}),400

    ext = os.path.splitext(raw_path)[1].lower()
    duration = 0.0
    if ext not in DIRECT_EXTS:
        try:
            send_path, duration = convert_to_mp3(raw_path)
        except Exception as e:
            if os.path.exists(raw_path):
                os.remove(raw_path)
            return jsonify({'error': f'Conversion failed: {str(e)}'}),500
    else:
        send_path = raw_path
        try:
            clip = AudioFileClip(send_path)
            duration = float(clip.duration)
            clip.close()
        except Exception:
            duration = 0.0

    job_id = uuid.uuid4().hex
    jobs[job_id] = {
        'job_id': job_id,
        'type':'transcription',
        'status':'queued',
        'created_at':iso_now(),
        'updated_at':None,
        'meta':{
            'audio_path': send_path,
            'orig_path': raw_path,
            'orig_name': f.filename,
            'orig_size': os.path.getsize(raw_path) if os.path.exists(raw_path) else 0,
            'orig_type': getattr(f,'content_type',''),
            'duration': duration
        },
        'result': None,
        'error': None
    }

    # process transcription synchronously
    try:
        text, segments = transcribe_with_groq(send_path)
        jobs[job_id]['result'] = {'transcription': text, 'segments': segments}
        jobs[job_id]['status'] = 'ready'
        jobs[job_id]['updated_at'] = iso_now()
    except Exception as e:
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['error'] = str(e)
        jobs[job_id]['updated_at'] = iso_now()
    finally:
        # --- CLEAN UP RAW AND CONVERTED FILES ---
        for path in [raw_path, send_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass

    return jsonify({'job_id': job_id, 'status': jobs[job_id]['status'], 'job': format_job_for_frontend(jobs[job_id])}), 201

@app.route('/api/v1/jobs/<job_id>/transcription', methods=['GET'])
def api_get_transcription(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({'error':'Not found'}),404
    if job['type'] != 'transcription':
        return jsonify({'error':'Job is not a transcription job'}),400
    if job['status'] != 'ready':
        return jsonify({'status': job['status']}),202

    text = job['result'].get('transcription','')
    segments = job['result'].get('segments')
    duration = job['meta'].get('duration',0.0)
    return jsonify(format_transcription_for_frontend(job_id,text,segments,duration=duration)),200

@app.route('/api/v1/jobs/<job_id>/summary', methods=['POST'])
def api_create_summary(job_id):
    parent_job = jobs.get(job_id)
    if not parent_job:
        return jsonify({'error': 'Not found'}), 404

    data = request.get_json() or {}
    style = data.get('style', 'short')
    text = parent_job.get('result', {}).get('transcription') or data.get('text')
    if not text:
        return jsonify({'error': 'No text available for summarization'}), 400

    # Create a new summary job entry
    summary_job_id = uuid.uuid4().hex
    jobs[summary_job_id] = {
        'job_id': summary_job_id,
        'type': 'summary',
        'status': 'queued',
        'created_at': iso_now(),
        'updated_at': None,
        'meta': {
            'parent_job_id': job_id,
            'text': text,
            'style': style
        },
        'result': None,
        'error': None
    }

    try:
        summary_text = summarize_text_with_groq(text, style=style)
        jobs[summary_job_id]['result'] = {'summary': summary_text}
        jobs[summary_job_id]['status'] = 'ready'
        jobs[summary_job_id]['updated_at'] = iso_now()
    except Exception as e:
        jobs[summary_job_id]['status'] = 'failed'
        jobs[summary_job_id]['error'] = str(e)
        jobs[summary_job_id]['updated_at'] = iso_now()
        summary_text = None

    return jsonify({
        'summary_job_id': summary_job_id,
        'status': jobs[summary_job_id]['status'],
        'summary': summary_text,
        'job': format_job_for_frontend(jobs[summary_job_id])
    }), 201


@app.route('/api/v1/jobs/<job_id>/status', methods=['GET'])
def api_job_status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Not found'}), 404
    return jsonify(format_job_for_frontend(job)), 200


@app.route('/api/v1/jobs/<job_id>/qa', methods=['POST'])
def api_create_qa(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Not found'}), 404

    question = None
    requirement_text = None
    context_text = None

    if request.content_type and 'multipart/form-data' in request.content_type:
        question = request.form.get('question')
        context_text = request.form.get('context')
        req_file = request.files.get('requirement_file')
        if req_file:
            r_ext = os.path.splitext(req_file.filename)[1].lower()
            if r_ext in {'.txt', '.md'}:
                requirement_text = req_file.read().decode('utf-8', errors='ignore')
            elif r_ext == '.pdf' and PyPDF2:
                buf = io.BytesIO(req_file.read())
                try:
                    reader = PyPDF2.PdfReader(buf)
                    pages = [p.extract_text() or '' for p in reader.pages]
                    requirement_text = ''.join(pages)
                except Exception:
                    requirement_text = None
    else:
        data = request.get_json() or {}
        question = data.get('question')
        requirement_text = data.get('requirement_text')
        context_text = data.get('context')

    if not question:
        return jsonify({'error': 'No question provided'}), 400

    # prefer transcription result as context if available
    context_text = context_text or job.get('result', {}).get('transcription', '')

    qa_job = create_job_entry('qa', meta={'context_text': context_text, 'question': question,
                                          'requirement_text': requirement_text})

    try:
        # Process immediately
        answer = answer_question_with_groq(context_text, question, requirement_text)
        jobs[qa_job]['result'] = {'answer': answer}
        jobs[qa_job]['status'] = 'ready'
        jobs[qa_job]['updated_at'] = iso_now()
    except Exception as e:
        jobs[qa_job]['status'] = 'failed'
        jobs[qa_job]['error'] = str(e)
        jobs[qa_job]['updated_at'] = iso_now()

    return jsonify({
        'qa_job_id': qa_job,
        'status': jobs[qa_job]['status'],
        'answer': jobs[qa_job].get('result', {}).get('answer'),
        'job': format_job_for_frontend(jobs[qa_job])
    }), 201




@app.route('/api/v1/jobs/<job_id>/download', methods=['GET'])
def api_download(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({'error':'Not found'}),404
    fmt = request.args.get('format','txt')
    if job['status'] != 'ready':
        return jsonify({'status':job['status']}),202
    if job['type']=='transcription':
        text = job['result'].get('transcription','')
    elif job['type']=='summary':
        text = job['result'].get('summary','')
    elif job['type']=='qa':
        text = job['result'].get('answer','')
    else:
        return jsonify({'error':'Unsupported job type'}),400
    out = io.BytesIO(text.encode('utf-8'))
    out.seek(0)
    return send_file(out, as_attachment=True, download_name=f"{job_id}.txt")

@app.route('/api/v1/jobs', methods=['GET'])
def api_list_jobs():
    return jsonify([format_job_for_frontend(jobs[k]) for k in jobs]),200

@app.route("/",methods=['GET'])
def startServer():
    return jsonify({"message":"All endpoints functional"}),200

# --- Models ---------------------------------------------------------------
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

with app.app_context():
    db.create_all()

# --- Helpers --------------------------------------------------------------
def now_utc():
    return datetime.utcnow()

def generate_4_digit_otp():
    return f"{secrets.randbelow(10**4):04}"

# --- Endpoints ------------------------------------------------------------
# Register new locker
@lockers_bp.route('/', methods=['POST'])
def register_locker():
    data = request.get_json() or {}
    locker_id = data.get('locker_id')
    if not locker_id:
        return jsonify({'error': 'locker_id required'}), 400
    if Locker.query.get(locker_id):
        return jsonify({'error': 'Locker already exists'}), 409
    l = Locker(id=locker_id, status='closed', last_activity=now_utc())
    db.session.add(l)
    db.session.commit()
    return jsonify({'locker_id': locker_id, 'status': 'closed'}), 201

# List all lockers
@lockers_bp.route('/', methods=['GET'])
def list_lockers():
    lockers = Locker.query.order_by(Locker.id).all()
    return jsonify([
        {
            'locker_id': l.id,
            'status': l.status,
            'current_password': l.otp,
            'last_activity': l.last_activity.isoformat() + 'Z',
            'expires_at': l.otp_expires.isoformat() + 'Z' if l.otp_expires else None
        } for l in lockers
    ])

@lockers_bp.route('/<locker_id>/status', methods=['GET'])
def get_status(locker_id):
    l = Locker.query.get_or_404(locker_id)
    return jsonify({
        'locker_id': l.id,
        'status': l.status,
        'current_password': l.otp,
        'last_activity': l.last_activity.isoformat() + 'Z'
    })

@lockers_bp.route('/<locker_id>/status', methods=['POST'])
def post_status(locker_id):
    data = request.get_json() or {}
    status = data.get('status')
    timestamp = data.get('timestamp')
    if status not in ('open', 'closed'):
        return jsonify({'error': 'Invalid status value'}), 400
    l = Locker.query.get_or_404(locker_id)
    l.status = status
    l.last_activity = datetime.fromisoformat(timestamp.replace('Z','')) if timestamp else now_utc()
    db.session.add(Activity(
        locker_id=locker_id,
        type='status_update',
        timestamp=l.last_activity,
        detail={'status': status}
    ))
    db.session.commit()
    return ('', 204)

@lockers_bp.route('/<locker_id>/otp', methods=['GET'])
def get_otp(locker_id):
    l = Locker.query.get_or_404(locker_id)
    return jsonify({
        'locker_id': l.id,
        'otp': l.otp,
        'expires_at': l.otp_expires.isoformat() + 'Z' if l.otp_expires else None
    })

@lockers_bp.route('/<locker_id>/otp', methods=['POST'])
def post_otp(locker_id):
    l = Locker.query.get_or_404(locker_id)
    otp = generate_4_digit_otp()
    expires = now_utc() + timedelta(minutes=15)
    l.otp = otp
    l.otp_expires = expires
    l.last_activity = now_utc()
    db.session.add(Activity(
        locker_id=locker_id,
        type='otp_generated',
        detail={}
    ))
    db.session.commit()
    return jsonify({
        'locker_id': l.id,
        'otp': otp,
        'expires_at': expires.isoformat() + 'Z'
    }), 201

@lockers_bp.route('/<locker_id>/activity', methods=['POST'])
def post_activity(locker_id):
    data = request.get_json() or {}
    activity_type = data.get('type')
    timestamp = data.get('timestamp')
    detail = data.get('detail', {})
    if activity_type not in ('opened','closed','otp_used','otp_failed','status_update','otp_generated'):
        return jsonify({'error': 'Invalid activity type'}), 400
    ts = datetime.fromisoformat(timestamp.replace('Z','')) if timestamp else now_utc()
    db.session.add(Activity(
        locker_id=locker_id,
        type=activity_type,
        timestamp=ts,
        detail=detail
    ))
    db.session.commit()
    return ('', 204)

@lockers_bp.route('/<locker_id>/verify-otp', methods=['POST'])
def verify_otp(locker_id):
    data = request.get_json() or {}
    entered = data.get('otp')
    l = Locker.query.get_or_404(locker_id)
    now = now_utc()
    if not l.otp or not l.otp_expires:
        return jsonify({'status': 'fail', 'message': 'No OTP set'}), 400
    if entered == l.otp and now < l.otp_expires:
        db.session.add(Activity(
            locker_id=locker_id,
            type='otp_used',
            detail={'entered_otp': entered, 'success': True}
        ))
        # Optionally clear OTP
        # l.otp = None; l.otp_expires = None
        db.session.commit()
        return jsonify({'status': 'success', 'message': 'OTP verified, unlock allowed'})
    else:
        db.session.add(Activity(
            locker_id=locker_id,
            type='otp_failed',
            detail={'entered_otp': entered, 'success': False}
        ))
        db.session.commit()
        return jsonify({'status': 'fail', 'message': 'Invalid or expired OTP'})

# Register blueprint and run app
app.register_blueprint(lockers_bp)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
