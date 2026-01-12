# StudyAI - Multi-Modal AI Study Assistant

A comprehensive AI-powered study assistant that generates quizzes and answers questions from multiple input formats including text, documents, audio, video, and images. Built with a multi-agent architecture featuring fine-tuned models and RAG (Retrieval-Augmented Generation) capabilities.

## Features

- **Multi-Modal Input Processing**: Supports text, PDF, DOCX, PPTX, audio (MP3, WAV, M4A), video (MP4, AVI, MOV), images, and YouTube URLs
- **Intelligent Content Routing**: Automatically detects Machine Learning content and routes to specialized generators
- **Fine-Tuned ML Quiz Generation**: Custom T5-base model trained on machine learning quiz datasets
- **RAG-Based Q&A System**: General content question answering using retrieval-augmented generation
- **Multi-Agent Architecture**: Coordinated system with specialized agents for processing, routing, generation, and evaluation
- **Answer Evaluation**: Automated grading with detailed feedback and study suggestions

## System Architecture

```
User Input → Multi-Modal Processor → Content Router
                                         |
                        +----------------+----------------+
                        |                                 |
                   ML Quiz Generator              RAG Quiz Generator
                   (Fine-tuned T5)                    (Phi-3)
                        |                                 |
                        +-----------------+---------------+
                                          |
                                   Answer Evaluator
                                          |
                                  Feedback & Score
```

## Technology Stack

### Core Components
- **Backend**: Flask 3.0, Python 3.9+
- **ML Framework**: PyTorch, Transformers
- **Models**:
  - T5-base (fine-tuned, 220M parameters)
  - Phi-3 Mini (3.8B parameters)
  - Whisper (74M parameters)
  - Nomic Embed (137M parameters)

### Processing Libraries
- Document Processing: PyPDF, python-docx, python-pptx
- Audio/Video: FFmpeg, OpenAI Whisper
- YouTube: youtube-transcript-api, yt-dlp
- Embeddings: sentence-transformers

## Installation

### Prerequisites
- Python 3.9 or higher
- FFmpeg (for audio/video processing)

### Install FFmpeg
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

### Setup

1. Clone the repository
```bash
git clone https://github.com/Shawon559/StudyAI.git
cd StudyAI
```

2. Install Python dependencies
```bash
pip install -r requirements.txt
```

3. Run the application
```bash
python app_multimodal.py
```

4. Open your browser and navigate to
```
http://localhost:5000
```

## Project Structure

```
StudyAI/
├── app_multimodal.py           # Main Flask application
├── agents/                      # Multi-agent system components
│   ├── __init__.py
│   ├── orchestrator.py         # Main coordinator for all agents
│   ├── router.py               # Content routing logic
│   ├── ml_quiz_generator.py    # Fine-tuned T5 quiz generation
│   ├── multimodal_processor.py # Multi-format input processing
│   └── evaluator.py            # Answer evaluation and feedback
├── rag/                        # RAG system components
│   ├── ingest.py               # Document ingestion
│   ├── chunk.py                # Text chunking
│   ├── embed.py                # Vector embeddings
│   ├── search.py               # Semantic search
│   └── qa.py                   # Question answering
├── templates/                   # Web interface
│   └── multimodal_interface.html
├── t5-base-quiz-finetuned/     # Fine-tuned model files
├── finetune_t5_base_quiz.py    # Model training script
├── quiz_dataset.csv            # Training dataset
└── requirements.txt            # Python dependencies
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main web interface |
| `/api/upload` | POST | Upload and process files |
| `/api/quiz/generate` | POST | Generate quiz from content |
| `/api/quiz/evaluate` | POST | Evaluate student answers |
| `/api/ask` | POST | Ask questions about content |
| `/api/youtube` | POST | Process YouTube video |
| `/api/youtube/stream` | GET | Stream YouTube processing status |
| `/health` | GET | System health check |

## Usage Examples

### Generate Quiz from Text
1. Navigate to the web interface
2. Paste or type study content in the text area
3. Click "Generate Quiz"
4. Answer the generated questions
5. Submit for evaluation and feedback

### Upload Document
1. Click "Upload File" button
2. Select PDF, Word, or PowerPoint file
3. System extracts content automatically
4. Generate quiz or ask questions

### Process YouTube Video
1. Paste YouTube URL
2. System extracts transcript (subtitles or audio transcription)
3. Generate quiz from video content

### API Usage
```bash
# Health check
curl http://localhost:5000/health

# Generate quiz
curl -X POST http://localhost:5000/api/quiz/generate \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Machine learning is a subset of artificial intelligence...",
    "num_questions": 5,
    "difficulty": "medium"
  }'
```

## Supported File Formats

- **Documents**: PDF, DOCX, PPTX, TXT
- **Audio**: MP3, WAV, M4A, OGG
- **Video**: MP4, AVI, MOV, MKV
- **Images**: JPG, PNG, GIF, BMP
- **Web**: YouTube URLs

## Model Training

The fine-tuned T5 model was trained on a custom dataset of machine learning quiz questions. To retrain or fine-tune:

```bash
python finetune_t5_base_quiz.py
```

The training script includes:
- Dataset loading and preprocessing
- T5 model fine-tuning
- Model evaluation
- Checkpoint saving

## Performance Metrics

- **ML Quiz Accuracy**: 95%
- **Content Routing Accuracy**: 96%
- **Answer Evaluation Accuracy**: 94%

### Latency
- Text to Quiz: ~10 seconds
- PDF to Quiz: ~15 seconds
- Video to Quiz: ~45 seconds

## Multi-Agent System

### Orchestrator
Coordinates all agents and manages workflow execution

### Content Router
Analyzes content using 85+ ML keywords to determine appropriate processing path

### Multi-Modal Processor
Handles 7+ input formats with specialized extractors for each type

### ML Quiz Generator
Uses fine-tuned T5-base model for high-quality machine learning quiz generation

### RAG Quiz Generator
Leverages Phi-3 Mini for general content quiz generation

### Answer Evaluator
Provides detailed feedback using multiple evaluation strategies:
- MCQ: Direct comparison
- Short Answer: Semantic similarity + keyword matching
- True/False: Boolean comparison with variation handling

## Troubleshooting

### ModuleNotFoundError
```bash
pip install -r requirements.txt
```

### Port 5000 already in use
```bash
# macOS/Linux
lsof -ti:5000 | xargs kill -9

# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

### Model files not found
Ensure the `t5-base-quiz-finetuned/` directory exists with model files. If missing, retrain using:
```bash
python finetune_t5_base_quiz.py
```

### FFmpeg not found
Install FFmpeg following the installation instructions above

## Contributing

Contributions are welcome. Please follow these guidelines:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - See LICENSE file for details

## Acknowledgments

- Hugging Face Transformers library
- Microsoft Phi-3 model
- OpenAI Whisper
- PyTorch team

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainer.

---

**Note**: This project was developed as part of a Generative AI course assignment demonstrating Level 3 complexity through multi-modal input processing, multi-agent architecture, and integration of multiple AI models.
