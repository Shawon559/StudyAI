"""
Answer Evaluator - Quiz answer evaluation with feedback
"""
import re
from typing import Dict, List
from difflib import SequenceMatcher


class AnswerEvaluator:
    """Evaluate quiz answers and provide feedback"""

    def __init__(self):
        pass

    def evaluate_mcq(self, student_answer: str, correct_answer: str) -> Dict:
        """Evaluate MCQ answer"""
        student = (student_answer or '').strip().upper()
        correct = (correct_answer or '').strip().upper()
        is_correct = student == correct

        return {
            'is_correct': is_correct,
            'score': 1.0 if is_correct else 0.0,
            'student_answer': student,
            'correct_answer': correct,
            'feedback': self._generate_mcq_feedback(is_correct, student, correct)
        }

    def evaluate_short_answer(self, student_answer: str, correct_answer: str,
                               question: str = None) -> Dict:
        """Evaluate short answer using similarity and keywords"""
        student = (student_answer or '').strip().lower()
        correct = (correct_answer or '').strip().lower()

        if student == correct:
            return {
                'is_correct': True, 'score': 1.0,
                'student_answer': student_answer, 'correct_answer': correct_answer,
                'feedback': "Perfect! Your answer is exactly correct."
            }

        similarity = self._calculate_similarity(student, correct)
        correct_keywords = set(re.findall(r'\b\w+\b', correct))
        student_keywords = set(re.findall(r'\b\w+\b', student))
        keyword_overlap = len(correct_keywords & student_keywords) / max(len(correct_keywords), 1)

        final_score = (similarity * 0.6) + (keyword_overlap * 0.4)
        is_correct = final_score >= 0.7
        partial_credit = 0.3 <= final_score < 0.7

        feedback = self._generate_short_answer_feedback(
            is_correct, partial_credit, final_score,
            student_answer, correct_answer, student_keywords, correct_keywords
        )

        return {
            'is_correct': is_correct, 'partial_credit': partial_credit,
            'score': final_score, 'similarity': similarity,
            'keyword_match': keyword_overlap,
            'student_answer': student_answer, 'correct_answer': correct_answer,
            'feedback': feedback
        }

    def evaluate_true_false(self, student_answer: str, correct_answer: str) -> Dict:
        """Evaluate true/false answer"""
        student = (student_answer or '').strip().lower()
        correct = (correct_answer or '').strip().lower()

        true_variations = {'true', 't', 'yes', '1'}
        false_variations = {'false', 'f', 'no', '0'}

        student_bool = student in true_variations
        correct_bool = correct in true_variations
        is_correct = student_bool == correct_bool

        return {
            'is_correct': is_correct,
            'score': 1.0 if is_correct else 0.0,
            'student_answer': student_answer,
            'correct_answer': correct_answer,
            'feedback': self._generate_true_false_feedback(is_correct, correct_bool)
        }

    def evaluate_quiz(self, quiz_questions: List[Dict], student_answers: List[str]) -> Dict:
        """Evaluate entire quiz"""
        if len(quiz_questions) != len(student_answers):
            return {'status': 'error', 'message': 'Question/answer count mismatch'}

        results = []
        total_score = 0.0

        for idx, (question, answer) in enumerate(zip(quiz_questions, student_answers)):
            q_type = question.get('type', 'short_answer')
            correct_answer = question.get('answer', '')

            if q_type == 'mcq':
                result = self.evaluate_mcq(answer, correct_answer)
            elif q_type == 'true_false':
                result = self.evaluate_true_false(answer, correct_answer)
            else:
                result = self.evaluate_short_answer(answer, correct_answer, question.get('question'))

            result['question_number'] = idx + 1
            result['question_text'] = question.get('question')
            results.append(result)
            total_score += result['score']

        percentage = (total_score / len(quiz_questions)) * 100

        return {
            'status': 'success',
            'total_questions': len(quiz_questions),
            'total_score': total_score,
            'max_score': len(quiz_questions),
            'percentage': percentage,
            'grade': self._calculate_grade(percentage),
            'results': results,
            'overall_feedback': self._generate_overall_feedback(percentage, len(quiz_questions))
        }

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        return SequenceMatcher(None, text1, text2).ratio()

    def _generate_mcq_feedback(self, is_correct: bool, student: str, correct: str) -> str:
        if is_correct:
            return f"Correct! The answer is {correct}."
        return f"Incorrect. You selected {student}, correct answer is {correct}."

    def _generate_short_answer_feedback(self, is_correct: bool, partial_credit: bool,
                                         score: float, student: str, correct: str,
                                         student_kw: set, correct_kw: set) -> str:
        if is_correct:
            return "Correct! Good understanding."

        if partial_credit:
            missing = correct_kw - student_kw
            feedback = f"Partially correct ({score:.0%}). "
            if missing:
                feedback += f"Consider including: {', '.join(list(missing)[:3])}. "
            feedback += f"\nExpected: {correct}"
            return feedback

        return f"Incorrect.\nYour answer: {student}\nExpected: {correct}"

    def _generate_true_false_feedback(self, is_correct: bool, correct_bool: bool) -> str:
        correct_text = "True" if correct_bool else "False"
        if is_correct:
            return f"Correct! The statement is {correct_text}."
        return f"Incorrect. The statement is {correct_text}."

    def _generate_overall_feedback(self, percentage: float, num_questions: int) -> str:
        if percentage >= 90:
            return f"Excellent! {percentage:.1f}% - Strong understanding."
        elif percentage >= 80:
            return f"Great job! {percentage:.1f}% - Good grasp of concepts."
        elif percentage >= 70:
            return f"Good effort! {percentage:.1f}% - Review missed questions."
        elif percentage >= 60:
            return f"Score: {percentage:.1f}% - Consider reviewing material."
        return f"Score: {percentage:.1f}% - More study needed."

    def _calculate_grade(self, percentage: float) -> str:
        if percentage >= 93: return "A"
        elif percentage >= 90: return "A-"
        elif percentage >= 87: return "B+"
        elif percentage >= 83: return "B"
        elif percentage >= 80: return "B-"
        elif percentage >= 77: return "C+"
        elif percentage >= 73: return "C"
        elif percentage >= 70: return "C-"
        elif percentage >= 67: return "D+"
        elif percentage >= 60: return "D"
        return "F"

    def generate_study_suggestions(self, quiz_results: Dict, content_source: str = None) -> List[str]:
        """Generate study suggestions based on performance"""
        suggestions = []
        results = quiz_results.get('results', [])
        incorrect = [r for r in results if not r['is_correct']]

        if not incorrect:
            suggestions.append("You've mastered this material!")
            return suggestions

        if len(incorrect) > len(results) / 2:
            suggestions.append(f"Review fundamentals from {content_source or 'this material'}.")
            suggestions.append("Re-read the source material carefully.")

        for result in incorrect[:3]:
            q_text = result.get('question_text', '')
            suggestions.append(f"Study more about: {q_text[:80]}...")

        if quiz_results.get('percentage', 0) < 70:
            suggestions.append("Practice with additional quizzes.")
            suggestions.append("Create summary notes of key concepts.")

        return suggestions
