"""
ML Quiz Generator - Fine-tuned T5 + Phi-3 fallback
"""
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Dict


class MLQuizGenerator:
    """Quiz generation using fine-tuned T5 model"""

    def __init__(self, model_path: str = "t5-base-quiz-finetuned"):
        self.model_path = model_path
        self.task_prefix = "generate quiz questions: "

        print(f"Loading fine-tuned model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"Model loaded on {self.device}")

    def create_summary(self, content: str, max_length: int = 512) -> str:
        """Truncate content for quiz generation"""
        words = content.split()
        if len(words) <= max_length:
            return content
        return ' '.join(words[:max_length]) + "..."

    def generate_quiz(self, content: str, num_questions: int = 5,
                      difficulty: str = "medium", question_types: List[str] = None) -> Dict:
        """Generate quiz from ML content"""
        try:
            summarized = self.create_summary(content, max_length=400)
            hints = f"\nGenerate {num_questions} {difficulty} level quiz questions."
            if question_types:
                hints += f" Types: {', '.join(question_types)}."

            input_text = self.task_prefix + summarized + hints

            encoded = self.tokenizer(
                input_text, return_tensors="pt",
                truncation=True, max_length=512
            ).to(self.device)

            output_ids = self.model.generate(
                **encoded, max_length=512, num_beams=5,
                early_stopping=True, no_repeat_ngram_size=3
            )

            generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            questions = self._parse_quiz_text(generated_text)

            # Fallback to Phi-3 if T5 fails
            if len(questions) == 0:
                print("[MLQuizGenerator] T5 failed, using Phi-3 fallback...")
                return self._generate_with_phi3(content, num_questions, difficulty)

            return {
                'status': 'success',
                'questions': questions,
                'raw_output': generated_text,
                'num_questions': len(questions),
                'difficulty': difficulty
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e), 'questions': []}

    def _generate_with_phi3(self, content: str, num_questions: int, difficulty: str) -> Dict:
        """Phi-3 fallback for quiz generation"""
        try:
            from .orchestrator import get_phi3_model
            model, tokenizer, device = get_phi3_model()

            words = content.split()
            if len(words) > 800:
                content = ' '.join(words[:800])

            prompt = f"""Create {num_questions} {difficulty} multiple choice questions from this text.

Format each question exactly like this:
Q1: Question here?
A) Option A
B) Option B
C) Option C
D) Option D
Answer: A

Text:
{content}

Generate {num_questions} questions:"""

            messages = [{"role": "user", "content": prompt}]
            inputs = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    inputs, max_new_tokens=1200,
                    do_sample=False, pad_token_id=tokenizer.eos_token_id
                )

            response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            questions = self._parse_quiz_text(response)

            return {
                'status': 'success',
                'questions': questions,
                'raw_output': response,
                'num_questions': len(questions),
                'difficulty': difficulty,
                'generator_note': 'Phi-3 fallback'
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e), 'questions': []}

    def _parse_quiz_text(self, text: str) -> List[Dict]:
        """Parse quiz text into structured format"""
        questions = []
        lines = text.strip().split('\n')

        current_question = None
        current_options = []
        current_answer = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith('Q') and ':' in line:
                if current_question:
                    questions.append({
                        'question': current_question,
                        'options': current_options if current_options else None,
                        'answer': current_answer,
                        'type': 'mcq' if current_options else 'short_answer'
                    })
                current_question = line.split(':', 1)[1].strip()
                current_options = []
                current_answer = None

            elif line and line[0] in ['A', 'B', 'C', 'D', 'E'] and (line[1:2] in [')', '.', ':']):
                current_options.append({'label': line[0], 'text': line[2:].strip()})

            elif line.lower().startswith('answer:'):
                current_answer = line.split(':', 1)[1].strip()

        if current_question:
            questions.append({
                'question': current_question,
                'options': current_options if current_options else None,
                'answer': current_answer,
                'type': 'mcq' if current_options else 'short_answer'
            })

        return questions

    def generate_from_summary(self, ml_summary: str, num_questions: int = 5) -> Dict:
        """Generate quiz from pre-made summary"""
        return self.generate_quiz(content=ml_summary, num_questions=num_questions)


class RAGQuizGenerator:
    """Quiz generation using Phi-3 for non-ML content"""

    def __init__(self):
        pass

    def generate_quiz(self, content: str, num_questions: int = 5) -> Dict:
        """Generate quiz using Phi-3"""
        try:
            from .orchestrator import get_phi3_model
            model, tokenizer, device = get_phi3_model()

            words = content.split()
            if len(words) > 800:
                content = ' '.join(words[:800])

            prompt = f"""Create {num_questions} multiple choice questions from this text.

Format each question exactly like this:
Q1: Question here?
A) Option A
B) Option B
C) Option C
D) Option D
Answer: A

Text:
{content}

Generate {num_questions} questions:"""

            messages = [{"role": "user", "content": prompt}]
            inputs = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                output_ids = model.generate(
                    inputs, max_new_tokens=1200,
                    do_sample=False, pad_token_id=tokenizer.eos_token_id
                )

            gen_ids = output_ids[0][inputs.shape[-1]:]
            generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

            ml_gen = MLQuizGenerator.__new__(MLQuizGenerator)
            questions = ml_gen._parse_quiz_text(generated_text)

            return {
                'status': 'success',
                'questions': questions,
                'raw_output': generated_text,
                'num_questions': len(questions),
                'source': 'phi3-local'
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e), 'questions': []}
