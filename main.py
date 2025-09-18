import os
import io
import re
import json
import sqlite3
import requests
from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, send_file
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from PIL import Image
from gtts import gTTS
import speech_recognition as sr
import torch
import logging

from job import EnhancedRealTimeJobSearch, RealTimeJobListing

try:
    import google.generativeai as genai
except Exception:
    genai = None

try:
    from transformers import BlipProcessor, BlipForConditionalGeneration

    BLIP_AVAILABLE = True
except Exception:
    BLIP_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("app")

app = Flask(__name__)
app.secret_key="Arun"

GEMINI_API_KEY = ''
HUGGINGFACE_API_KEY = ''
RAPIDAPI_KEY = ''
GEMINI_MODEL_NAME = 'gemini-2.0-flash'

job_searcher = EnhancedRealTimeJobSearch(rapidapi_key=RAPIDAPI_KEY)

for directory in ['static/generated', 'static/uploads', 'static/audio', 'static/temp', 'exports', 'templates']:
    os.makedirs(directory, exist_ok=True)

class GeminiClient:
    def __init__(self, api_key, model_name=GEMINI_MODEL_NAME):
        self.api_key = api_key
        self.model_name = model_name
        self.available = False
        self.model = None

        if not api_key or genai is None:
            return

        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
            self.available = True
        except Exception as e:
            logger.warning(f"[Gemini] Error during initialization: {e}")

    def generate_text(self, prompt, conversation_history=None):
        if not self.available:
            return "I'm currently not available. Please check if Gemini API is configured."

        if conversation_history:
            ctx = []
            for msg in conversation_history[-10:]:
                who = 'User' if msg.get('role') == 'user' else 'Assistant'
                ctx.append(f"{who}: {msg.get('content')}\n")
            prompt = f"{''.join(ctx)}\nUser: {prompt}\nAssistant:"

        try:
            resp = self.model.generate_content(prompt)
            return getattr(resp, 'text', str(resp))
        except Exception as e:
            return f"I encountered an error: {e}"

    def generate_title(self, first_message):
        try:
            prompt = f"Generate a concise 3-5 word title for a conversation that starts with: '{first_message[:100]}...'. Only return the title, nothing else."
            title = self.generate_text(prompt)
            title = re.sub(r'[^\w\s-]', '', title).strip()
            return title[:50] if title else "New Conversation"
        except:
            return "New Conversation"

gemini_client = GeminiClient(GEMINI_API_KEY) if GEMINI_API_KEY else None

class BLIPModel:
    def __init__(self):
        self.processor = None
        self.model = None
        self.device = 'cpu'
        self.available = False
        self.load_model()

    def load_model(self):
        if not BLIP_AVAILABLE:
            return

        try:
            os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
            cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'huggingface')
            os.makedirs(cache_dir, exist_ok=True)

            self.processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-base",
                cache_dir=cache_dir,
                use_fast=True
            )
            self.model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base",
                cache_dir=cache_dir
            )

            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = self.model.to(self.device)
            self.available = True
        except Exception as e:
            logger.warning(f"[BLIP] Error while loading: {e}")

    def analyze_image(self, image_path, prompt=None):
        if not self.available:
            return "BLIP image analysis not available."

        try:
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(image, prompt, return_tensors="pt") if prompt else self.processor(image,
                                                                                                      return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                out = self.model.generate(**inputs, max_length=50, num_beams=5)

            return self.processor.decode(out[0], skip_special_tokens=True)
        except Exception as e:
            return f"BLIP analysis error: {e}"

blip_model = BLIPModel()

class VoiceRecognizer:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.available = True

voice_recognizer = VoiceRecognizer()

def get_db_connection():
    """Get a database connection with proper error handling"""
    try:
        conn = sqlite3.connect('users.db', timeout=20.0)
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA foreign_keys = ON')
        return conn
    except sqlite3.Error as e:
        logger.error(f"Database connection failed: {e}")
        raise

def init_db():
    """Initialize database with enhanced real-time job search fields"""
    logger.info("Initializing database with real-time job search support...")

    conn = get_db_connection()
    c = conn.cursor()

    try:

        c.execute("SELECT name FROM sqlite_master WHERE type='table'")
        existing_tables = [row[0] for row in c.fetchall()]

        if 'users' not in existing_tables:
            logger.info("Creating users table...")
            c.execute('''
                      CREATE TABLE users
                      (
                          id         INTEGER PRIMARY KEY AUTOINCREMENT,
                          first_name TEXT        NOT NULL,
                          last_name  TEXT        NOT NULL,
                          email      TEXT UNIQUE NOT NULL,
                          password   TEXT        NOT NULL,
                          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                      )
                      ''')

        if 'conversations' not in existing_tables:
            logger.info("Creating conversations table...")
            c.execute('''
                      CREATE TABLE conversations
                      (
                          id          INTEGER PRIMARY KEY AUTOINCREMENT,
                          user_id     INTEGER NOT NULL,
                          title       TEXT    NOT NULL DEFAULT 'New Conversation',
                          is_favorite BOOLEAN          DEFAULT FALSE,
                          created_at  TIMESTAMP        DEFAULT CURRENT_TIMESTAMP,
                          updated_at  TIMESTAMP        DEFAULT CURRENT_TIMESTAMP,
                          FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                      )
                      ''')

        if 'messages' not in existing_tables:
            logger.info("Creating messages table...")
            c.execute('''
                      CREATE TABLE messages
                      (
                          id              INTEGER PRIMARY KEY AUTOINCREMENT,
                          conversation_id INTEGER NOT NULL,
                          role            TEXT    NOT NULL,
                          content         TEXT    NOT NULL,
                          timestamp       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                          image_path      TEXT,
                          audio_path      TEXT,
                          message_type    TEXT      DEFAULT 'text',
                          metadata        TEXT,
                          is_favorite     BOOLEAN   DEFAULT FALSE,
                          reaction        TEXT,
                          FOREIGN KEY (conversation_id) REFERENCES conversations (id) ON DELETE CASCADE
                      )
                      ''')

        if 'job_searches' not in existing_tables:
            logger.info("Creating job_searches table...")
            c.execute('''
                      CREATE TABLE job_searches
                      (
                          id              INTEGER PRIMARY KEY AUTOINCREMENT,
                          user_id         INTEGER NOT NULL,
                          query           TEXT    NOT NULL,
                          location        TEXT,
                          results_count   INTEGER   DEFAULT 0,
                          search_time     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                          search_metadata TEXT,
                          max_age_hours   INTEGER   DEFAULT 168,
                          FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                      )
                      ''')

        if 'saved_jobs' not in existing_tables:
            logger.info("Creating enhanced saved_jobs table...")
            c.execute('''
                      CREATE TABLE saved_jobs
                      (
                          id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                          user_id            INTEGER NOT NULL,
                          job_id             TEXT    NOT NULL,
                          title              TEXT    NOT NULL,
                          company            TEXT    NOT NULL,
                          location           TEXT,
                          description        TEXT,
                          url                TEXT,
                          company_url        TEXT      DEFAULT '',
                          source             TEXT,
                          salary             TEXT,
                          job_type           TEXT,
                          posted_date        TEXT,
                          posted_timestamp   TEXT,
                          freshness_score    INTEGER   DEFAULT 0,
                          is_fresh           BOOLEAN   DEFAULT FALSE,
                          saved_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                          application_status TEXT      DEFAULT 'saved',
                          FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                      )
                      ''')
        else:

            c.execute("PRAGMA table_info(saved_jobs)")
            columns = [column[1] for column in c.fetchall()]

            new_columns = [
                ('job_id', 'TEXT DEFAULT ""'),
                ('company_url', 'TEXT DEFAULT ""'),
                ('posted_timestamp', 'TEXT'),
                ('freshness_score', 'INTEGER DEFAULT 0'),
                ('is_fresh', 'BOOLEAN DEFAULT FALSE')
            ]

            for col_name, col_def in new_columns:
                if col_name not in columns:
                    logger.info(f"Adding {col_name} column to saved_jobs table...")
                    c.execute(f'ALTER TABLE saved_jobs ADD COLUMN {col_name} {col_def}')

        indexes = [
            'CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id)',
            'CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id)',
            'CREATE INDEX IF NOT EXISTS idx_job_searches_user_id ON job_searches(user_id)',
            'CREATE INDEX IF NOT EXISTS idx_saved_jobs_user_id ON saved_jobs(user_id)',
            'CREATE INDEX IF NOT EXISTS idx_saved_jobs_job_id ON saved_jobs(job_id)',
            'CREATE INDEX IF NOT EXISTS idx_saved_jobs_freshness ON saved_jobs(freshness_score)',
            'CREATE INDEX IF NOT EXISTS idx_saved_jobs_is_fresh ON saved_jobs(is_fresh)'
        ]

        for idx_sql in indexes:
            c.execute(idx_sql)

        conn.commit()
        logger.info("Database initialized successfully with real-time job search support!")

    except Exception as e:
        logger.error(f"Database initialization error: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()

def save_message(conversation_id, role, content, image_path=None, audio_path=None, message_type='text', metadata=None):
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute('''
                  INSERT INTO messages (conversation_id, role, content, image_path, audio_path, message_type, metadata)
                  VALUES (?, ?, ?, ?, ?, ?, ?)
                  ''', (conversation_id, role, content, image_path, audio_path, message_type,
                        json.dumps(metadata) if metadata else None))

        c.execute('UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?', (conversation_id,))
        conn.commit()
        message_id = c.lastrowid
        return message_id
    except Exception as e:
        logger.error(f"Error saving message: {e}")
        conn.rollback()
        return None
    finally:
        conn.close()

def get_or_create_conversation(user_id, title='New Conversation'):
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute('INSERT INTO conversations (user_id, title) VALUES (?, ?)', (user_id, title))
        conn.commit()
        conv_id = c.lastrowid
        return conv_id
    except Exception as e:
        logger.error(f"Error creating conversation: {e}")
        conn.rollback()
        return None
    finally:
        conn.close()

def get_user_conversations(user_id, limit=50):
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute('''
                  SELECT c.id,
                         c.title,
                         c.is_favorite,
                         c.created_at,
                         c.updated_at,
                         (SELECT content FROM messages WHERE conversation_id = c.id ORDER BY timestamp ASC LIMIT 1) as first_message, (
                  SELECT COUNT (*)
                  FROM messages
                  WHERE conversation_id = c.id) as message_count
                  FROM conversations c
                  WHERE c.user_id = ?
                  ORDER BY c.updated_at DESC
                      LIMIT ?
                  ''', (user_id, limit))

        conversations = []
        for row in c.fetchall():
            conversations.append({
                'id': row['id'],
                'title': row['title'],
                'is_favorite': bool(row['is_favorite']) if row['is_favorite'] is not None else False,
                'created_at': row['created_at'],
                'updated_at': row['updated_at'] if row['updated_at'] else row['created_at'],
                'first_message': row['first_message'] or '',
                'message_count': row['message_count']
            })
        return conversations
    except Exception as e:
        logger.error(f"Error getting conversations: {e}")
        return []
    finally:
        conn.close()

def get_conversation_messages(conversation_id):
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute('''
                  SELECT id,
                         role,
                         content, timestamp, image_path, audio_path, message_type, metadata, is_favorite, reaction
                  FROM messages
                  WHERE conversation_id = ?
                  ORDER BY timestamp ASC
                  ''', (conversation_id,))

        messages = []
        for row in c.fetchall():
            messages.append({
                'id': row['id'],
                'role': row['role'],
                'content': row['content'],
                'timestamp': row['timestamp'],
                'image_path': row['image_path'],
                'audio_path': row['audio_path'],
                'message_type': row['message_type'],
                'metadata': json.loads(row['metadata']) if row['metadata'] else None,
                'is_favorite': bool(row['is_favorite']),
                'reaction': row['reaction']
            })
        return messages
    except Exception as e:
        logger.error(f"Error getting messages: {e}")
        return []
    finally:
        conn.close()

def save_image_bytes(image_bytes, ext='.png'):
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
    filename = f"generated_{timestamp}{ext}"
    path = os.path.join('static/generated', filename)
    with open(path, 'wb') as f:
        f.write(image_bytes)
    return '/' + path.replace('\\', '/')

def generate_image(prompt):
    if HUGGINGFACE_API_KEY:
        try:
            hf_url = 'https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1'
            headers = {'Authorization': f'Bearer {HUGGINGFACE_API_KEY}', 'Accept': 'image/png'}
            payload = {
                'inputs': prompt,
                'options': {'wait_for_model': True},
                'parameters': {'num_inference_steps': 20, 'guidance_scale': 7.5, 'width': 512, 'height': 512}
            }
            resp = requests.post(hf_url, headers=headers, json=payload, timeout=60)

            if resp.status_code == 200 and 'image' in resp.headers.get('Content-Type', ''):
                return save_image_bytes(resp.content, ext='.png')
        except Exception as e:
            logger.warning(f"[HuggingFace] Exception: {e}")

    try:
        encoded = requests.utils.quote(prompt)
        pollinations_url = f'https://image.pollinations.ai/prompt/{encoded}?width=512&height=512&nologo=true'
        resp = requests.get(pollinations_url, timeout=30)
        if resp.status_code == 200 and resp.headers.get('Content-Type', '').startswith('image'):
            return save_image_bytes(resp.content)
    except Exception as e:
        logger.warning(f"[Pollinations] Exception: {e}")

    return None

def get_gemini_response(message, conversation_history=None, image_data=None):
    image_creation_patterns = [
        r'(?:create|generate|make|draw|show me)\s+(?:an?\s+)?image\s+(?:of\s+)?(.+)',
        r'(?:paint|sketch|illustrate)\s+(.+)',
    ]

    for pattern in image_creation_patterns:
        m = re.search(pattern, message.lower())
        if m:
            image_prompt = m.group(1).strip()
            img_url = generate_image(image_prompt)
            if img_url:
                return f"I've created an image of '{image_prompt}' for you. [IMAGE_GENERATED:{img_url}]"
            else:
                return f"I tried to create an image of '{image_prompt}', but the image generation services are currently unavailable."

    if image_data:
        try:
            analysis = blip_model.analyze_image(image_data, prompt=f"User asked: {message}")
            return f"I can see the image! {analysis}"
        except Exception as e:
            return f"I had trouble analyzing the image: {e}"

    if gemini_client and gemini_client.available:
        try:
            return gemini_client.generate_text(message, conversation_history=conversation_history)
        except Exception as e:
            return f"I encountered an error: {e}"

    return "Hello! I'm an AI assistant with real-time job search capabilities. I can help you find the latest job postings directly from company websites. How can I help you today?"

def check_gemini_connection():
    if not gemini_client or not gemini_client.available:
        return False, 'Gemini not configured or unavailable.'

    try:
        test_resp = gemini_client.generate_text('Hello')
        return True, 'Gemini API is working.'
    except Exception as e:
        return False, f'Gemini error: {e}'

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('chat'))
    return render_template('home.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')

        if not email or not password:
            flash('Please enter both email and password')
            return render_template('login.html')

        conn = get_db_connection()
        c = conn.cursor()
        try:
            c.execute('SELECT id, first_name, last_name, email, password FROM users WHERE email = ?', (email,))
            user = c.fetchone()

            if user and check_password_hash(user['password'], password):
                session['user_id'] = user['id']
                session['first_name'] = user['first_name']
                session['last_name'] = user['last_name']
                session['email'] = user['email']
                flash('Login successful!')
                return redirect(url_for('chat'))
            else:
                flash('Invalid email or password')
        except Exception as e:
            logger.error(f"Login error: {e}")
            flash('Login failed. Please try again.')
        finally:
            conn.close()

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        first_name = request.form.get('first_name', '').strip()
        last_name = request.form.get('last_name', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        if not first_name or not last_name or not email or not password:
            flash('All fields are required')
            return render_template('register.html')

        if password != confirm_password:
            flash('Passwords do not match')
            return render_template('register.html')

        if len(password) < 6:
            flash('Password must be at least 6 characters long')
            return render_template('register.html')

        if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[A-Za-z]{2,}$', email):
            flash('Please enter a valid email address')
            return render_template('register.html')

        hashed = generate_password_hash(password)

        conn = get_db_connection()
        c = conn.cursor()
        try:
            c.execute('INSERT INTO users (first_name, last_name, email, password) VALUES (?, ?, ?, ?)',
                      (first_name, last_name, email, hashed))
            conn.commit()
            flash('Registration successful! Please log in.')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Email already exists. Please use a different email.')
        except Exception as e:
            logger.error(f"Registration error: {e}")
            flash('Registration failed. Please try again.')
            conn.rollback()
        finally:
            conn.close()

    return render_template('register.html')

@app.route('/chat')
@app.route('/chat/<int:conversation_id>')
def chat(conversation_id=None):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    try:
        user_conversations = get_user_conversations(session['user_id'])

        if not conversation_id and user_conversations:
            conversation_id = user_conversations[0]['id']
        elif not conversation_id:
            conversation_id = get_or_create_conversation(session['user_id'])
            user_conversations = get_user_conversations(session['user_id'])

        current_messages = get_conversation_messages(conversation_id) if conversation_id else []
        gemini_status, status_message = check_gemini_connection()

        first_name = session.get('first_name', 'User')
        last_name = session.get('last_name', '')

        current_user_obj = {
            'id': session.get('user_id'),
            'first_name': first_name,
            'last_name': last_name,
            'name': f"{first_name} {last_name}".strip(),
            'email': session.get('email', ''),
            'created_at': None,
            'last_active': datetime.now()
        }

        total_conversations = len(user_conversations)
        total_messages = sum(conv.get('message_count', 0) for conv in user_conversations)

        user_stats = {
            'total_conversations': total_conversations,
            'total_messages': total_messages,
            'days_active': 30
        }

        recent_conversations = []
        for conv in user_conversations:
            updated_at = conv.get('updated_at', conv.get('created_at', ''))
            formatted_date = updated_at
            if updated_at:
                try:
                    date_obj = datetime.strptime(updated_at, '%Y-%m-%d %H:%M:%S')
                    formatted_date = date_obj.strftime('%B %Y')
                except Exception:
                    formatted_date = str(updated_at)[:10]

            recent_conversations.append({
                'id': conv.get('id'),
                'title': conv.get('title', 'New Conversation'),
                'last_message': conv.get('first_message', '')[:100],
                'message_count': conv.get('message_count', 0),
                'updated_at': formatted_date,
                'created_at': conv.get('created_at', '')
            })

        return render_template('chat.html',
                               current_user=current_user_obj,
                               conversations=user_conversations,
                               current_conversation_id=conversation_id,
                               messages=current_messages,
                               gemini_status=gemini_status,
                               status_message=status_message,
                               recent_conversations=recent_conversations,
                               user_stats=user_stats)

    except Exception as e:
        logger.error(f"Chat route error: {e}")
        return f"Error loading chat: {e}", 500

@app.route('/jobs')
def jobs():
    """Real-time job search page with fresh job filtering"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('jobs.html')

@app.route('/api/search_jobs', methods=['POST'])
def api_search_jobs():
    """Enhanced real-time job search API with freshness filtering"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401

    try:
        data = request.get_json() or {}
        job_title = data.get('job_title', '').strip()
        location = data.get('location', '').strip()
        max_age_hours = min(int(data.get('max_age_hours', 168)), 720)
        limit = min(int(data.get('limit', 15)), 50)

        if not job_title:
            return jsonify({'success': False, 'error': 'Job title is required'}), 400

        logger.info(
            f"Real-time job search: title='{job_title}', location='{location}', max_age_hours={max_age_hours}, limit={limit}")

        jobs = job_searcher.search_jobs(job_title, location, max_age_hours, limit)

        job_searcher.save_job_search(session['user_id'], job_title, location, len(jobs))

        jobs_data = []
        fresh_jobs_count = 0
        company_direct_count = 0

        for job in jobs:
            job_dict = job.to_dict()
            jobs_data.append(job_dict)

            if job.is_fresh:
                fresh_jobs_count += 1

            if (job_dict.get('company_url') and
                    not any(site in job_dict['company_url'].lower()
                            for site in ['naukri', 'monster', 'indeed', 'glassdoor', 'linkedin'])):
                company_direct_count += 1

        logger.info(
            f"Returning {len(jobs_data)} jobs ({fresh_jobs_count} fresh, {company_direct_count} direct company links)")

        return jsonify({
            'success': True,
            'jobs': jobs_data,
            'total_found': len(jobs_data),
            'fresh_jobs_count': fresh_jobs_count,
            'company_direct_count': company_direct_count,
            'search_query': job_title,
            'location': location,
            'max_age_hours': max_age_hours,
            'timestamp': datetime.now().isoformat(),
            'search_metadata': {
                'api_version': 'realtime_v2',
                'freshness_filter': True,
                'company_enhancement': True
            }
        })

    except Exception as e:
        logger.error(f"Real-time job search error: {e}")
        return jsonify({
            'success': False,
            'error': 'Real-time job search failed',
            'details': str(e)
        }), 500

@app.route('/api/save_job', methods=['POST'])
def api_save_job():
    """Enhanced save job API with real-time job data"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401

    try:
        data = request.get_json() or {}

        required_fields = ['title', 'company']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'success': False, 'error': f'{field} is required'}), 400

        posted_timestamp = None
        if data.get('posted_timestamp'):
            try:
                posted_timestamp = datetime.fromisoformat(data['posted_timestamp'])
            except:
                pass

        job = RealTimeJobListing(
            title=data.get('title', ''),
            company=data.get('company', ''),
            location=data.get('location', ''),
            description=data.get('description', ''),
            url=data.get('url', ''),
            source=data.get('source', ''),
            salary=data.get('salary', ''),
            job_type=data.get('job_type', ''),
            posted_date=data.get('posted_date', ''),
            company_url=data.get('company_url', ''),
            job_id=data.get('job_id', ''),
            posted_timestamp=posted_timestamp,
            is_fresh=data.get('is_fresh', False)
        )

        success = job_searcher.save_job(session['user_id'], job)

        if success:
            return jsonify({'success': True, 'message': 'Job saved successfully'})
        else:
            return jsonify({'success': False, 'error': 'Failed to save job'}), 500

    except Exception as e:
        logger.error(f"Enhanced save job error: {e}")
        return jsonify({'success': False, 'error': 'Failed to save job', 'details': str(e)}), 500

@app.route('/api/saved_jobs')
def api_saved_jobs():
    """Enhanced saved jobs API with real-time data"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401

    try:
        limit = request.args.get('limit')
        limit = int(limit) if limit else None

        saved_jobs = job_searcher.get_saved_jobs(session['user_id'], limit)

        for job in saved_jobs:
            if job.get('posted_timestamp'):
                try:
                    posted_time = datetime.fromisoformat(job['posted_timestamp'])
                    hours_old = (datetime.now() - posted_time).total_seconds() / 3600
                    job['hours_old'] = hours_old
                    job['time_display'] = format_time_ago(hours_old)
                except:
                    job['hours_old'] = 999
                    job['time_display'] = 'Unknown'

        return jsonify({
            'success': True,
            'saved_jobs': saved_jobs,
            'metadata': {
                'total_count': len(saved_jobs),
                'fresh_count': sum(1 for job in saved_jobs if job.get('is_fresh')),
                'company_direct_count': sum(1 for job in saved_jobs
                                            if job.get('company_url') and
                                            not any(site in job['company_url'].lower()
                                                    for site in ['naukri', 'monster', 'indeed']))
            }
        })

    except Exception as e:
        logger.error(f"Enhanced get saved jobs error: {e}")
        return jsonify({'success': False, 'error': 'Failed to get saved jobs', 'details': str(e)}), 500

@app.route('/api/search_history')
def api_search_history():
    """Enhanced search history API"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401

    try:
        limit = min(int(request.args.get('limit', 10)), 50)
        history = job_searcher.get_search_history(session['user_id'], limit)

        for item in history:
            if item.get('search_metadata'):
                try:
                    metadata = json.loads(item['search_metadata'])
                    item['search_metadata_parsed'] = metadata
                except:
                    item['search_metadata_parsed'] = {}

        return jsonify({
            'success': True,
            'history': history,
            'metadata': {
                'total_searches': len(history),
                'recent_searches': len([h for h in history
                                        if datetime.now() - datetime.fromisoformat(
                        h.get('search_time', '2000-01-01')) < timedelta(days=7)])
            }
        })

    except Exception as e:
        logger.error(f"Get search history error: {e}")
        return jsonify({'success': False, 'error': 'Failed to get search history', 'details': str(e)}), 500

@app.route('/api/job_freshness_stats')
def api_job_freshness_stats():
    """API to get real-time job freshness statistics"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401

    try:
        conn = get_db_connection()
        c = conn.cursor()

        c.execute('''
                  SELECT COUNT(*)                                      as total_saved,
                         SUM(CASE WHEN is_fresh = 1 THEN 1 ELSE 0 END) as fresh_jobs,
                         AVG(freshness_score)                          as avg_freshness_score,
                         COUNT(CASE WHEN company_url != '' AND company_url NOT LIKE '%naukri%'
                                   AND company_url NOT LIKE '%monster%' AND company_url NOT LIKE '%indeed%'
                               THEN 1 END)                             as direct_company_jobs
                  FROM saved_jobs
                  WHERE user_id = ?
                  ''', (session['user_id'],))

        stats = dict(c.fetchone())
        conn.close()

        return jsonify({
            'success': True,
            'stats': {
                'total_saved_jobs': stats['total_saved'] or 0,
                'fresh_jobs': stats['fresh_jobs'] or 0,
                'avg_freshness_score': round(stats['avg_freshness_score'] or 0, 1),
                'direct_company_jobs': stats['direct_company_jobs'] or 0,
                'freshness_percentage': round((stats['fresh_jobs'] or 0) / max(stats['total_saved'] or 1, 1) * 100, 1)
            }
        })

    except Exception as e:
        logger.error(f"Job freshness stats error: {e}")
        return jsonify({'success': False, 'error': 'Failed to get job statistics'}), 500

def format_time_ago(hours: float) -> str:
    """Format hours into human-readable time ago string"""
    if hours < 1:
        minutes = int(hours * 60)
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    elif hours < 24:
        return f"{int(hours)} hour{'s' if int(hours) != 1 else ''} ago"
    elif hours < 168:
        days = int(hours / 24)
        return f"{days} day{'s' if days != 1 else ''} ago"
    else:
        weeks = int(hours / 168)
        return f"{weeks} week{'s' if weeks != 1 else ''} ago"

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    try:
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{session['user_id']}_{timestamp}_{filename}"
        path = os.path.join('static/uploads', filename)
        file.save(path)

        return jsonify({
            'success': True,
            'image_path': '/' + path.replace('\\', '/')
        })
    except Exception as e:
        logger.error(f"Image upload error: {e}")
        return jsonify({'error': 'Failed to upload image'}), 500

@app.route('/api/conversations')
def api_conversations():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401

    conversations = get_user_conversations(session['user_id'])
    return jsonify(conversations)

@app.route('/api/conversation/<int:conversation_id>/messages')
def api_conversation_messages(conversation_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401

    messages = get_conversation_messages(conversation_id)
    return jsonify(messages)

@app.route('/send_message', methods=['POST'])
def send_message():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401

    data = request.get_json() or {}
    message = (data.get('message') or '').strip()
    conversation_id = data.get('conversation_id')
    image_path_for_analysis = data.get('image_path')

    if not message:
        return jsonify({'error': 'Empty message'}), 400

    try:
        if not conversation_id:
            conversation_id = get_or_create_conversation(session['user_id'])

        save_message(conversation_id, 'user', message, image_path=image_path_for_analysis)

        conversation_history = get_conversation_messages(conversation_id)[:-1]

        if len(conversation_history) == 0 and gemini_client and gemini_client.available:
            title = gemini_client.generate_title(message)
            conn = get_db_connection()
            c = conn.cursor()
            try:
                c.execute('UPDATE conversations SET title = ? WHERE id = ?', (title, conversation_id))
                conn.commit()
            except Exception as e:
                logger.warning(f"Error updating title: {e}")
            finally:
                conn.close()

        if image_path_for_analysis:
            path = image_path_for_analysis.lstrip('/')
            ai_response = get_gemini_response(message, conversation_history=conversation_history, image_data=path)
        else:
            ai_response = get_gemini_response(message, conversation_history=conversation_history)

        generated_image_url = None
        if isinstance(ai_response, str) and '[IMAGE_GENERATED:' in ai_response:
            start = ai_response.find('[IMAGE_GENERATED:') + len('[IMAGE_GENERATED:')
            end = ai_response.find(']', start)
            if end != -1:
                generated_image_url = ai_response[start:end]
                ai_response = ai_response.replace(f'[IMAGE_GENERATED:{generated_image_url}]', '').strip()

        save_message(conversation_id, 'assistant', ai_response,
                     image_path=generated_image_url if generated_image_url else None,
                     message_type='image_generation' if generated_image_url else 'text')

        audio_url = None
        try:
            if ai_response and len(ai_response.strip()) < 500:
                tts = gTTS(ai_response.strip()[:300])
                audio_filename = f"static/audio/response_{datetime.now().strftime('%Y%m%d%H%M%S')}.mp3"
                tts.save(audio_filename)
                audio_url = '/' + audio_filename.replace('\\', '/')
        except Exception as e:
            logger.warning(f"[TTS] Error: {e}")

        response_data = {
            'success': True,
            'user_message': message,
            'ai_response': ai_response,
            'conversation_id': conversation_id,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        if audio_url:
            response_data['audio_url'] = audio_url
        if generated_image_url:
            response_data['generated_image_url'] = generated_image_url

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Send message error: {e}")
        return jsonify({'error': f'Server error: {e}'}), 500

@app.route('/process_voice_text', methods=['POST'])
def process_voice_text():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401

    try:
        data = request.get_json()
        transcribed_text = data.get('text', '').strip()

        if not transcribed_text:
            return jsonify({'error': 'No text provided'}), 400

        return jsonify({
            'success': True,
            'text': transcribed_text
        })

    except Exception as e:
        logger.error(f"Voice processing error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/new_conversation', methods=['POST'])
def new_conversation():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401

    conversation_id = get_or_create_conversation(session['user_id'])
    return jsonify({'success': True, 'conversation_id': conversation_id})

@app.route('/api/conversation/<int:conversation_id>/favorite', methods=['POST'])
def toggle_conversation_favorite(conversation_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401

    try:
        conn = get_db_connection()
        c = conn.cursor()

        c.execute('SELECT is_favorite FROM conversations WHERE id = ? AND user_id = ?',
                  (conversation_id, session['user_id']))
        result = c.fetchone()

        if not result:
            return jsonify({'error': 'Conversation not found'}), 404

        new_favorite_status = not bool(result['is_favorite'])
        c.execute('UPDATE conversations SET is_favorite = ? WHERE id = ?', (new_favorite_status, conversation_id))
        conn.commit()
        conn.close()

        return jsonify({'success': True, 'is_favorite': new_favorite_status})

    except Exception as e:
        logger.error(f"Toggle favorite error: {e}")
        return jsonify({'error': 'Failed to toggle favorite', 'details': str(e)}), 500

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000, debug=True)