"""
Multi-Modal Study Assistant - Flask Application
"""
import os
import json
from flask import Flask, render_template, request, jsonify, session, Response
from werkzeug.utils import secure_filename
import tempfile

from agents.orchestrator import StudyAssistantOrchestrator
from rag.qa import answer_question

UPLOAD_FOLDER = tempfile.mkdtemp()
_current_quiz_storage = {}
ALLOWED_EXTENSIONS = {
    'txt', 'pdf', 'doc', 'docx', 'ppt', 'pptx',
    'mp3', 'wav', 'm4a', 'ogg',
    'mp4', 'avi', 'mov', 'mkv',
    'jpg', 'jpeg', 'png', 'gif', 'bmp'
}
CHUNK_DIR = "rag/data/chunks"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
app.secret_key = 'study-assistant-secret-key-2024'

orchestrator = None


def get_orchestrator():
    global orchestrator
    if orchestrator is None:
        orchestrator = StudyAssistantOrchestrator()
    return orchestrator


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def list_lecture_ids():
    if not os.path.exists(CHUNK_DIR):
        return []
    return sorted([os.path.splitext(f)[0] for f in os.listdir(CHUNK_DIR) if f.endswith(".json")])


@app.route("/", methods=["GET"])
def index():
    return render_template("multimodal_interface.html", lecture_ids=list_lecture_ids())


@app.route("/api/ask", methods=["POST"])
def api_ask():
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        lecture_id = data.get('lecture_id')
        content = data.get('content')

        if not question:
            return jsonify({'error': 'Question is required'}), 400

        result = get_orchestrator().ask_question(question=question, content=content, lecture_id=lecture_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route("/api/upload", methods=["POST"])
def api_upload():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        result = get_orchestrator().process_input(filepath)
        session['last_upload'] = {
            'filename': filename, 'filepath': filepath,
            'content': result.get('content', ''), 'modality': result.get('modality', 'unknown')
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route("/api/quiz/generate", methods=["POST"])
def api_generate_quiz():
    try:
        data = request.get_json()
        content = data.get('content', '').strip()
        num_questions = data.get('num_questions', 5)
        difficulty = data.get('difficulty', 'medium')

        if not content and 'last_upload' in session:
            content = session['last_upload'].get('content', '')

        if not content:
            return jsonify({'error': 'Content is required'}), 400

        result = get_orchestrator().generate_quiz(content=content, num_questions=num_questions, difficulty=difficulty)

        if result['status'] == 'success':
            quiz_id = session.get('_id', os.urandom(8).hex())
            session['_id'] = quiz_id
            session['quiz_id'] = quiz_id
            _current_quiz_storage[quiz_id] = result['questions']
            print(f"[Quiz] Stored {len(result['questions'])} questions (id: {quiz_id})")

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route("/api/quiz/evaluate", methods=["POST"])
def api_evaluate_quiz():
    try:
        data = request.get_json()
        student_answers = data.get('answers', [])

        quiz_id = session.get('quiz_id')
        quiz_questions = _current_quiz_storage.get(quiz_id) if quiz_id else None

        if not quiz_questions:
            return jsonify({'error': 'No active quiz. Generate a quiz first.'}), 400
        if not student_answers:
            return jsonify({'error': 'Answers are required'}), 400

        result = get_orchestrator().evaluate_answers(quiz_questions, student_answers)

        if result['status'] == 'success':
            result['study_suggestions'] = get_orchestrator().get_study_suggestions(result)

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route("/api/workflow", methods=["POST"])
def api_full_workflow():
    try:
        content = None

        if 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                content = filepath
        else:
            data = request.get_json() or {}
            content = data.get('content', '').strip()

        if not content:
            return jsonify({'error': 'Content or file is required'}), 400

        workflow_type = request.form.get('workflow_type') or request.json.get('workflow_type', 'ask')
        question = request.form.get('question') or request.json.get('question', '')
        num_questions = int(request.form.get('num_questions', 5) or request.json.get('num_questions', 5))

        result = get_orchestrator().full_workflow(
            input_data=content, workflow_type=workflow_type,
            question=question, num_questions=num_questions
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route("/api/youtube", methods=["POST"])
def api_youtube():
    try:
        data = request.get_json()
        url = data.get('url', '').strip()

        if not url:
            return jsonify({'error': 'YouTube URL is required'}), 400

        result = get_orchestrator().process_youtube(url)

        if result.get('status') == 'success':
            session['last_upload'] = {
                'filename': 'youtube_video', 'filepath': url,
                'content': result.get('content', ''), 'modality': 'video'
            }
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route("/api/youtube/stream", methods=["GET"])
def api_youtube_stream():
    import time
    import re

    url = request.args.get('url', '').strip()
    if not url:
        return jsonify({'error': 'YouTube URL is required'}), 400

    def generate():
        try:
            yield f"data: {json.dumps({'step': 1, 'total': 5, 'message': 'Validating URL...', 'status': 'processing'})}\n\n"
            time.sleep(0.3)

            patterns = [
                r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
                r'youtube\.com\/shorts\/([^&\n?#]+)'
            ]
            video_id = None
            for pattern in patterns:
                match = re.search(pattern, url)
                if match:
                    video_id = match.group(1)
                    break

            if not video_id:
                yield f"data: {json.dumps({'status': 'error', 'error': 'Invalid YouTube URL'})}\n\n"
                return

            yield f"data: {json.dumps({'step': 2, 'total': 5, 'message': 'Checking subtitles...', 'status': 'processing'})}\n\n"

            transcript_text = None
            try:
                from youtube_transcript_api import YouTubeTranscriptApi
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
                transcript_text = ' '.join([item['text'] for item in transcript_list])
                yield f"data: {json.dumps({'step': 5, 'total': 5, 'message': 'Complete!', 'status': 'success', 'content': transcript_text, 'source': 'subtitles', 'video_id': video_id})}\n\n"
                return
            except:
                yield f"data: {json.dumps({'step': 3, 'total': 5, 'message': 'Downloading audio...', 'status': 'processing'})}\n\n"

            import subprocess
            import tempfile as tmp
            import glob

            temp_dir = tmp.mkdtemp()
            audio_path = os.path.join(temp_dir, 'youtube_audio.mp3')

            try:
                subprocess.run([
                    'yt-dlp', '-x', '--audio-format', 'mp3',
                    '-o', audio_path.replace('.mp3', '.%(ext)s'), url
                ], capture_output=True, timeout=120)

                audio_files = glob.glob(os.path.join(temp_dir, 'youtube_audio.*'))
                if audio_files:
                    actual_audio = audio_files[0]
                    if not actual_audio.endswith('.mp3'):
                        subprocess.run(['ffmpeg', '-i', actual_audio, '-q:a', '0', audio_path, '-y'], capture_output=True)
                    else:
                        audio_path = actual_audio

                    yield f"data: {json.dumps({'step': 4, 'total': 5, 'message': 'Transcribing with Whisper...', 'status': 'processing'})}\n\n"

                    import whisper
                    import torch
                    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
                    model = whisper.load_model("base", device=device)
                    result = model.transcribe(audio_path)
                    transcript_text = result["text"]

                    yield f"data: {json.dumps({'step': 5, 'total': 5, 'message': 'Complete!', 'status': 'success', 'content': transcript_text, 'source': 'whisper', 'video_id': video_id})}\n\n"
                else:
                    yield f"data: {json.dumps({'status': 'error', 'error': 'Audio download failed'})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'status': 'error', 'error': str(e)})}\n\n"
            finally:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)

        except Exception as e:
            yield f"data: {json.dumps({'status': 'error', 'error': str(e)})}\n\n"

    return Response(generate(), mimetype='text/event-stream')


@app.route("/api/lectures", methods=["GET"])
def api_list_lectures():
    return jsonify({'lectures': list_lecture_ids()})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        'status': 'healthy',
        'orchestrator_loaded': orchestrator is not None,
        'ml_quiz_available': orchestrator.ml_quiz_available if orchestrator else False
    })


if __name__ == "__main__":
    print("\n" + "="*50)
    print("Multi-Modal Study Assistant")
    print("="*50)
    print("\nSupported: PDF, Word, PPT, Audio, Video, YouTube")
    print("Features: Quiz Generation, RAG Q&A, Answer Evaluation")
    print("="*50 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=True)
