"""
Main Orchestrator - Coordinates all agents
"""
from typing import Dict, List, Union
from .router import ContentRouter
from .multimodal_processor import MultiModalProcessor
from .ml_quiz_generator import MLQuizGenerator, RAGQuizGenerator
from .evaluator import AnswerEvaluator
from rag.qa import answer_question as rag_answer

# Global Phi-3 model cache
_PHI3_MODEL = None
_PHI3_TOKENIZER = None
_PHI3_DEVICE = None


def get_phi3_model():
    """Get cached Phi-3 model"""
    global _PHI3_MODEL, _PHI3_TOKENIZER, _PHI3_DEVICE

    if _PHI3_MODEL is None:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        if torch.backends.mps.is_available():
            _PHI3_DEVICE = "mps"
        elif torch.cuda.is_available():
            _PHI3_DEVICE = "cuda"
        else:
            _PHI3_DEVICE = "cpu"

        model_name = "microsoft/Phi-3-mini-4k-instruct"
        print(f"[Orchestrator] Loading {model_name} on {_PHI3_DEVICE}...")

        _PHI3_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        _PHI3_MODEL = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if _PHI3_DEVICE in ("mps", "cuda") else torch.float32,
            device_map=_PHI3_DEVICE,
        )
        _PHI3_MODEL.eval()
        print(f"[Orchestrator] Phi-3 ready!")

    return _PHI3_MODEL, _PHI3_TOKENIZER, _PHI3_DEVICE


class StudyAssistantOrchestrator:
    """Main orchestrator for multi-agent system"""

    def __init__(self, ml_model_path: str = "t5-base-quiz-finetuned"):
        print("Initializing Study Assistant Orchestrator...")

        self.multimodal_processor = MultiModalProcessor()
        self.router = ContentRouter()
        self.evaluator = AnswerEvaluator()

        try:
            self.ml_quiz_gen = MLQuizGenerator(ml_model_path)
            self.ml_quiz_available = True
        except Exception as e:
            print(f"Warning: ML Quiz Generator not available: {e}")
            self.ml_quiz_gen = None
            self.ml_quiz_available = False

        self.rag_quiz_gen = RAGQuizGenerator()
        print("Orchestrator initialized successfully!")

    def process_input(self, input_data: Union[str, Dict], input_type: str = None) -> Dict:
        """Process any input through multi-modal processor"""
        result = self.multimodal_processor.process(input_data, input_type)
        return result

    def process_youtube(self, youtube_url: str) -> Dict:
        """Process YouTube video"""
        result = self.multimodal_processor.process_youtube(youtube_url)
        return result

    def ask_question(self, question: str, content: str = None, lecture_id: str = None) -> Dict:
        """Answer a question using RAG or Phi-3"""
        print(f"Answering question: {question[:50]}...")

        # Use RAG if lecture_id provided
        if lecture_id:
            try:
                answer, sources = rag_answer(question, lecture_id)
                return {
                    'status': 'success',
                    'answer': answer,
                    'sources': sources,
                    'method': 'rag',
                    'lecture_id': lecture_id
                }
            except Exception as e:
                return {'status': 'error', 'error': str(e)}

        # Use Phi-3 if content provided
        if content:
            routing = self.router.route(content, action="ask")

            try:
                import torch
                model, tokenizer, device = get_phi3_model()

                truncated_content = content[:3000] if len(content) > 3000 else content

                messages = [
                    {"role": "system", "content": "You are a helpful study assistant. Answer based on the provided content."},
                    {"role": "user", "content": f"Content:\n{truncated_content}\n\nQuestion: {question}"}
                ]

                inputs = tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
                ).to(device)

                with torch.no_grad():
                    output_ids = model.generate(inputs, max_new_tokens=512, do_sample=False)

                gen_ids = output_ids[0][inputs.shape[-1]:]
                answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

                return {'status': 'success', 'answer': answer, 'routing': routing, 'method': 'phi3'}

            except Exception as e:
                return {'status': 'error', 'error': str(e)}

        return {'status': 'error', 'message': 'Either lecture_id or content required'}

    def generate_quiz(self, content: str, num_questions: int = 5, difficulty: str = "medium") -> Dict:
        """Generate quiz using appropriate generator"""
        print(f"Generating quiz ({num_questions} questions, {difficulty})...")

        routing = self.router.route(content, action="quiz")
        print(f"Routing: {routing['target_agent']} ({routing['confidence']:.0%})")

        if routing['target_agent'] == 'ml_quiz_generator' and self.ml_quiz_available:
            result = self.ml_quiz_gen.generate_quiz(content=content, num_questions=num_questions, difficulty=difficulty)
            result['generator'] = 'fine_tuned_ml_model'
        else:
            result = self.rag_quiz_gen.generate_quiz(content=content, num_questions=num_questions)
            result['generator'] = 'rag_openai'

        result['routing'] = routing
        return result

    def evaluate_answers(self, quiz_questions: List[Dict], student_answers: List[str]) -> Dict:
        """Evaluate student answers"""
        print(f"Evaluating {len(student_answers)} answers...")

        evaluations = []
        total_score = 0

        for i, (question, answer) in enumerate(zip(quiz_questions, student_answers)):
            q_type = question.get('type', 'mcq')
            correct = question.get('answer', '')

            if q_type == 'mcq':
                eval_result = self.evaluator.evaluate_mcq(answer, correct)
            elif q_type == 'true_false':
                eval_result = self.evaluator.evaluate_true_false(answer, correct)
            else:
                eval_result = self.evaluator.evaluate_short_answer(answer, correct, question.get('question', ''))

            eval_result['question_num'] = i + 1
            eval_result['question_text'] = question.get('question', '')
            evaluations.append(eval_result)
            total_score += eval_result['score']

        return {
            'status': 'success',
            'evaluations': evaluations,
            'total_score': total_score,
            'max_score': len(quiz_questions),
            'percentage': (total_score / len(quiz_questions)) * 100 if quiz_questions else 0
        }

    def get_study_suggestions(self, evaluation_result: Dict) -> List[str]:
        """Generate study suggestions based on performance"""
        suggestions = []
        percentage = evaluation_result.get('percentage', 0)

        if percentage < 50:
            suggestions.append("Review the core concepts thoroughly")
            suggestions.append("Focus on understanding fundamentals before details")
        elif percentage < 75:
            suggestions.append("Good progress! Review missed questions")
            suggestions.append("Practice with more challenging material")
        else:
            suggestions.append("Excellent work! Ready for advanced topics")
            suggestions.append("Consider teaching concepts to others")

        wrong_topics = []
        for eval_item in evaluation_result.get('evaluations', []):
            if not eval_item.get('is_correct', False):
                wrong_topics.append(eval_item.get('question_text', '')[:50])

        if wrong_topics:
            suggestions.append(f"Review these areas: {', '.join(wrong_topics[:3])}")

        return suggestions

    def full_workflow(self, input_data, workflow_type="quiz", input_type=None, question=None, num_questions=5) -> Dict:
        """Execute full workflow"""
        # Process input
        processed = self.process_input(input_data, input_type)
        if processed['status'] != 'success':
            return processed

        content = processed['content']

        if workflow_type == "ask" and question:
            return self.ask_question(question, content)
        elif workflow_type == "quiz":
            return self.generate_quiz(content, num_questions)

        return {'status': 'error', 'message': 'Invalid workflow type'}
