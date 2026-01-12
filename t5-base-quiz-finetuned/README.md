# T5-Base Fine-Tuned Model

This directory contains the fine-tuned T5-base model for machine learning quiz generation.

## Model Files

The `model.safetensors` file (850 MB) is too large to be stored on GitHub directly.

## Options to Get the Model

### Option 1: Train the Model Yourself
Run the training script from the project root:
```bash
python finetune_t5_base_quiz.py
```

This will generate all the model files including `model.safetensors`.

### Option 2: Download from Hugging Face (if uploaded)
If the model has been uploaded to Hugging Face Hub, you can download it:
```python
from transformers import T5ForConditionalGeneration

model = T5ForConditionalGeneration.from_pretrained("your-username/t5-base-quiz-finetuned")
model.save_pretrained("t5-base-quiz-finetuned/")
```

### Option 3: Use Without the Fine-Tuned Model
The application will automatically fall back to using Phi-3 Mini for quiz generation if the fine-tuned model is not available.

## Model Details

- Base Model: T5-base (220M parameters)
- Fine-tuned on: Machine learning quiz dataset
- Training Dataset: quiz_dataset.csv (included in repository)
- Task: Text-to-text generation for quiz questions and answers
