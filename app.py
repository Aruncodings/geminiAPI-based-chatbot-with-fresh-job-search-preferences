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

try:
    import google.generativeai as genai
except Exception:
    genai = None

try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    BLIP_AVAILABLE = True
except Exception:
    BLIP_AVAILABLE = False

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY','AIzaSyCfC0cXzoXSLMPTjTeynwEJNZdACoApYgo').strip() or None
HUGGINGFACE_API_KEY = os.environ.get('HUGGINGFACE_API_KEY', '').strip() or None
GEMINI_MODEL_NAME = 'gemini-2.0-flash'

for directory in ['static/generated', 'static/uploads', 'static/audio', 'static/temp', 'exports']:
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
            print(f"[Gemini] Error: {e}")

    def generate_text(self, prompt, conversation_history=None):
        if not self.available:
            raise RuntimeError("Gemini client not available")

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
            raise RuntimeError(f'Gemini generation failed: {e}')

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
            print(f"[BLIP] Error: {e}")

    def analyze_image(self, image_path, prompt=None):
        if not self.available:
            return "BLIP image analysis not available."

        try:
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(image, prompt, return_tensors="pt") if prompt else self.processor(image, return_tensors="pt")
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

def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL DEFAULT 'New Conversation',
            is_favorite BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            image_path TEXT,
            audio_path TEXT,
            message_type TEXT DEFAULT 'text',
            metadata TEXT,
            is_favorite BOOLEAN DEFAULT FALSE,
            reaction TEXT,
            FOREIGN KEY (conversation_id) REFERENCES conversations (id)
        )
    ''')

    c.execute('CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id)')

    conn.commit()
    conn.close()

def migrate_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()

    try:
        # Check and add missing columns
        c.execute("PRAGMA table_info(conversations)")
        conv_columns = [column[1] for column in c.fetchall()]

        if 'is_favorite' not in conv_columns:
            c.execute('ALTER TABLE conversations ADD COLUMN is_favorite BOOLEAN DEFAULT FALSE')
        if 'updated_at' not in conv_columns:
            c.execute('ALTER TABLE conversations ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP')

        c.execute("PRAGMA table_info(messages)")
        msg_columns = [column[1] for column in c.fetchall()]

        missing_msg_columns = [
            ('image_path', 'TEXT'),
            ('audio_path', 'TEXT'),
            ('message_type', 'TEXT DEFAULT "text"'),
            ('metadata', 'TEXT'),
            ('is_favorite', 'BOOLEAN DEFAULT FALSE'),
            ('reaction', 'TEXT')
        ]

        for col_name, col_def in missing_msg_columns:
            if col_name not in msg_columns:
                c.execute(f'ALTER TABLE messages ADD COLUMN {col_name} {col_def}')

        conn.commit()
    except Exception as e:
        print(f"Migration error: {e}")
    finally:
        conn.close()

def save_message(conversation_id, role, content, image_path=None, audio_path=None, message_type='text', metadata=None):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()

    c.execute('''
        INSERT INTO messages 
        (conversation_id, role, content, image_path, audio_path, message_type, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (conversation_id, role, content, image_path, audio_path, message_type,
          json.dumps(metadata) if metadata else None))

    c.execute('UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?', (conversation_id,))

    conn.commit()
    message_id = c.lastrowid
    conn.close()
    return message_id

def get_or_create_conversation(user_id, title='New Conversation'):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('INSERT INTO conversations (user_id, title) VALUES (?, ?)', (user_id, title))
    conv_id = c.lastrowid
    conn.commit()
    conn.close()
    return conv_id

def get_user_conversations(user_id, limit=50):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()

    c.execute('''
        SELECT c.id, c.title, c.is_favorite, c.created_at, c.updated_at,
               (SELECT content FROM messages WHERE conversation_id = c.id ORDER BY timestamp ASC LIMIT 1) as first_message,
               (SELECT COUNT(*) FROM messages WHERE conversation_id = c.id) as message_count
        FROM conversations c
        WHERE c.user_id = ?
        ORDER BY c.updated_at DESC
        LIMIT ?
    ''', (user_id, limit))

    conversations = []
    for row in c.fetchall():
        conversations.append({
            'id': row[0],
            'title': row[1],
            'is_favorite': bool(row[2]) if row[2] is not None else False,
            'created_at': row[3],
            'updated_at': row[4] if row[4] else row[3],
            'first_message': row[5] or '',
            'message_count': row[6]
        })

    conn.close()
    return conversations

def get_conversation_messages(conversation_id):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()

    c.execute('''
        SELECT id, role, content, timestamp, image_path, audio_path, message_type, metadata, is_favorite, reaction
        FROM messages
        WHERE conversation_id = ?
        ORDER BY timestamp ASC
    ''', (conversation_id,))

    messages = []
    for row in c.fetchall():
        messages.append({
            'id': row[0],
            'role': row[1],
            'content': row[2],
            'timestamp': row[3],
            'image_path': row[4],
            'audio_path': row[5],
            'message_type': row[6],
            'metadata': json.loads(row[7]) if row[7] else None,
            'is_favorite': bool(row[8]),
            'reaction': row[9]
        })

    conn.close()
    return messages

def search_conversations(user_id, query):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()

    c.execute('''
        SELECT DISTINCT c.id, c.title, m.content, m.timestamp
        FROM conversations c
        JOIN messages m ON c.id = m.conversation_id
        WHERE c.user_id = ? AND (c.title LIKE ? OR m.content LIKE ?)
        ORDER BY m.timestamp DESC
        LIMIT 20
    ''', (user_id, f'%{query}%', f'%{query}%'))

    results = []
    for row in c.fetchall():
        results.append({
            'conversation_id': row[0],
            'title': row[1],
            'content': row[2],
            'timestamp': row[3]
        })

    conn.close()
    return results

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
            print(f"[HuggingFace] Exception: {e}")

    try:
        encoded = requests.utils.quote(prompt)
        pollinations_url = f'https://image.pollinations.ai/prompt/{encoded}?width=512&height=512&nologo=true'
        resp = requests.get(pollinations_url, timeout=30)
        if resp.status_code == 200 and resp.headers.get('Content-Type', '').startswith('image'):
            return save_image_bytes(resp.content)
    except Exception as e:
        print(f"[Pollinations] Exception: {e}")

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

    return "I'm currently not available. Please check if Gemini API is configured."

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
        email = request.form.get('email')
        password = request.form.get('password')

        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('SELECT id, first_name, last_name, email, password FROM users WHERE email = ?', (email,))
        user = c.fetchone()
        conn.close()

        if user and check_password_hash(user[4], password):
            session['user_id'] = user[0]
            session['first_name'] = user[1]
            session['last_name'] = user[2]
            session['email'] = user[3]
            return redirect(url_for('chat'))
        else:
            flash('Invalid email or password')

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if password != confirm_password:
            flash('Passwords do not match')
            return render_template('register.html')

        hashed = generate_password_hash(password)

        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute('INSERT INTO users (first_name, last_name, email, password) VALUES (?, ?, ?, ?)',
                      (first_name, last_name, email, hashed))
            conn.commit()
            conn.close()
            flash('Registration successful! Please log in.')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Email already exists')

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

        # Create current_user object
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

        # Calculate days active
        days_active = 30
        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute('SELECT created_at FROM users WHERE id = ?', (session.get('user_id'),))
            user_data = c.fetchone()
            if user_data and user_data[0]:
                created_date = datetime.strptime(user_data[0], '%Y-%m-%d %H:%M:%S')
                current_user_obj['created_at'] = created_date
                days_active = max(1, (datetime.now() - created_date).days)
            conn.close()
        except Exception:
            current_user_obj['created_at'] = datetime.now() - timedelta(days=30)

        # Calculate stats
        total_conversations = len(user_conversations)
        total_messages = sum(conv.get('message_count', 0) for conv in user_conversations)

        user_stats = {
            'total_conversations': total_conversations,
            'total_messages': total_messages,
            'days_active': days_active
        }

        # Process recent conversations
        recent_conversations = []
        for conv in user_conversations:
            updated_at = conv.get('updated_at', conv.get('created_at', ''))
            formatted_date = updated_at
            if updated_at:
                try:
                    date_obj = datetime.strptime(updated_at, '%Y-%m-%d %H:%M:%S')
                    formatted_date = date_obj.strftime('%B %Y')
                except:
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
        return f"Error loading chat: {e}", 500

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    filename = secure_filename(file.filename)
    path = os.path.join('static/uploads', filename)
    file.save(path)

    return jsonify({
        'success': True,
        'image_path': '/' + path.replace('\\', '/')
    })

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
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute('UPDATE conversations SET title = ? WHERE id = ?', (title, conversation_id))
            conn.commit()
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
            print(f"[TTS] Error: {e}")

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
        conn = sqlite3.connect('users.db')
        c = conn.cursor()

        c.execute('SELECT is_favorite FROM conversations WHERE id = ? AND user_id = ?',
                  (conversation_id, session['user_id']))
        result = c.fetchone()

        if not result:
            return jsonify({'error': 'Conversation not found'}), 404

        new_favorite_status = not bool(result[0])
        c.execute('UPDATE conversations SET is_favorite = ? WHERE id = ?',
                  (new_favorite_status, conversation_id))
        conn.commit()
        conn.close()

        return jsonify({'success': True, 'is_favorite': new_favorite_status})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/conversation/<int:conversation_id>/delete', methods=['DELETE'])
def delete_conversation(conversation_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401

    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()

        c.execute('SELECT id FROM conversations WHERE id = ? AND user_id = ?',
                  (conversation_id, session['user_id']))
        if not c.fetchone():
            return jsonify({'error': 'Conversation not found'}), 404

        c.execute('DELETE FROM messages WHERE conversation_id = ?', (conversation_id,))
        c.execute('DELETE FROM conversations WHERE id = ?', (conversation_id,))

        conn.commit()
        conn.close()

        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/conversation/<int:conversation_id>/rename', methods=['POST'])
def rename_conversation(conversation_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401

    data = request.get_json()
    new_title = data.get('title', '').strip()

    if not new_title:
        return jsonify({'error': 'Title cannot be empty'}), 400

    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()

        c.execute('UPDATE conversations SET title = ? WHERE id = ? AND user_id = ?',
                  (new_title, conversation_id, session['user_id']))

        if c.rowcount == 0:
            return jsonify({'error': 'Conversation not found'}), 404

        conn.commit()
        conn.close()

        return jsonify({'success': True, 'title': new_title})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/message/<int:message_id>/favorite', methods=['POST'])
def toggle_message_favorite(message_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401

    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()

        c.execute('''
            SELECT m.is_favorite FROM messages m
            JOIN conversations c ON m.conversation_id = c.id
            WHERE m.id = ? AND c.user_id = ?
        ''', (message_id, session['user_id']))

        result = c.fetchone()
        if not result:
            return jsonify({'error': 'Message not found'}), 404

        new_favorite_status = not bool(result[0])
        c.execute('UPDATE messages SET is_favorite = ? WHERE id = ?',
                  (new_favorite_status, message_id))
        conn.commit()
        conn.close()

        return jsonify({'success': True, 'is_favorite': new_favorite_status})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/search')
def search():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401

    query = request.args.get('q', '').strip()
    if not query:
        return jsonify({'error': 'Search query required'}), 400

    results = search_conversations(session['user_id'], query)
    return jsonify(results)

@app.route('/api/export/<int:conversation_id>')
def export_conversation(conversation_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401

    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()

        c.execute('SELECT title FROM conversations WHERE id = ? AND user_id = ?',
                  (conversation_id, session['user_id']))
        conv_result = c.fetchone()

        if not conv_result:
            return jsonify({'error': 'Conversation not found'}), 404

        messages = get_conversation_messages(conversation_id)

        export_data = {
            'conversation_title': conv_result[0],
            'export_date': datetime.now().isoformat(),
            'messages': []
        }

        for msg in messages:
            export_data['messages'].append({
                'role': msg['role'],
                'content': msg['content'],
                'timestamp': msg['timestamp'],
                'message_type': msg['message_type'],
                'is_favorite': msg['is_favorite']
            })

        export_filename = f"conversation_{conversation_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        export_path = os.path.join('exports', export_filename)

        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        return send_file(export_path, as_attachment=True, download_name=export_filename)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/status')
def status():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401

    gemini_status, gemini_message = check_gemini_connection()

    return jsonify({
        'gemini_running': gemini_status,
        'gemini_message': gemini_message,
        'gemini_model': GEMINI_MODEL_NAME,
        'blip_available': blip_model.available,
        'voice_recognition_available': voice_recognizer.available,
        'features': {
            'text_chat': gemini_status,
            'image_analysis': blip_model.available,
            'image_generation': bool(HUGGINGFACE_API_KEY) or True,
            'audio_transcription': voice_recognizer.available,
            'text_to_speech': True,
            'conversation_export': True,
            'message_search': True
        }
    })

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out successfully')
    return redirect(url_for('login'))

if __name__ == '__main__':
    init_db()
    migrate_db()
    print('=' * 60)

    status_ok, msg = check_gemini_connection()
    print(f'Gemini status: {msg}')
    print(f'BLIP status: {"Available" if blip_model.available else "Not available"}')
    print(f'Voice Recognition: {"Available" if voice_recognizer.available else "Not available"}')

    if not status_ok:
        print('\nTo enable Gemini features: set GEMINI_API_KEY environment variable')

    app.run(debug=True, host='0.0.0.0', port=5000)

    # AIzaSyCfC0cXzoXSLMPTjTeynwEJNZdACoApYgo  - gemini
    982
    # f2429d9msh72009ff83ea93c7p183569jsn4e2c993e0aa3 - rapid api key