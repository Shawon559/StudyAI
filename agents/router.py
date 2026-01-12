"""
Content Router - Routes content to appropriate agent (ML vs RAG)
"""
import re
from typing import Dict, Tuple


class ContentRouter:
    """Route content based on ML keyword analysis"""

    def __init__(self):
        self.ml_keywords = {
            'supervised learning', 'unsupervised learning', 'reinforcement learning',
            'neural network', 'deep learning', 'machine learning', 'cnn', 'rnn', 'lstm',
            'classification', 'regression', 'clustering', 'decision tree', 'random forest',
            'gradient descent', 'backpropagation', 'activation function', 'overfitting',
            'underfitting', 'cross validation', 'feature engineering', 'dimensionality reduction',
            'svm', 'support vector', 'k-means', 'pca', 'principal component', 'logistic regression',
            'linear regression', 'naive bayes', 'knn', 'k nearest', 'ensemble', 'boosting',
            'bagging', 'hyperparameter', 'loss function', 'optimizer', 'adam', 'sgd',
            'convolution', 'pooling', 'dropout', 'batch normalization', 'transfer learning',
            'fine tuning', 'data augmentation', 'confusion matrix', 'precision', 'recall',
            'f1 score', 'accuracy', 'roc curve', 'auc', 'train test split', 'validation set',
            'bias variance', 'regularization', 'l1', 'l2', 'ridge', 'lasso', 'elastic net',
            'perceptron', 'mlp', 'autoencoder', 'gan', 'generative adversarial', 'transformer',
            'attention mechanism', 'bert', 'gpt', 'word embedding', 'word2vec', 'glove',
            'tokenization', 'nlp', 'natural language processing', 'computer vision',
            'tensor', 'pytorch', 'tensorflow', 'keras', 'scikit-learn', 'sklearn',
            'model training', 'model evaluation', 'learning rate', 'epoch', 'batch size'
        }

    def analyze_content(self, text: str) -> Tuple[bool, float, Dict]:
        """Analyze if content is ML-related"""
        text_lower = text.lower()

        matches = [kw for kw in self.ml_keywords if kw in text_lower]
        word_count = len(text.split())
        match_count = len(matches)

        if word_count > 0:
            confidence = min(match_count / max(word_count / 10, 1), 1.0)
        else:
            confidence = 0.0

        strong_keywords = {'machine learning', 'deep learning', 'neural network', 'cnn', 'rnn'}
        has_strong = any(kw in text_lower for kw in strong_keywords)

        is_ml = confidence > 0.15 or has_strong

        return is_ml, confidence, {
            'matched_keywords': matches[:10],
            'match_count': match_count,
            'word_count': word_count,
            'has_strong_keyword': has_strong
        }

    def route(self, content: str, action: str = "ask") -> Dict:
        """Route content to appropriate agent"""
        is_ml, confidence, metadata = self.analyze_content(content)

        if action == "quiz" and is_ml:
            target = "ml_quiz_generator"
            reason = "ML content detected"
        elif action == "quiz":
            target = "rag_quiz_generator"
            reason = "Non-ML content"
        elif is_ml and confidence > 0.3:
            target = "ml_enhanced_rag"
            reason = "High ML content"
        else:
            target = "rag"
            reason = "General content"

        return {
            'target_agent': target,
            'is_ml_content': is_ml,
            'confidence': confidence,
            'reasoning': reason,
            'metadata': metadata
        }

    def extract_ml_concepts(self, text: str) -> list:
        """Extract ML concepts from text"""
        text_lower = text.lower()
        concepts = []

        for keyword in self.ml_keywords:
            if keyword in text_lower:
                pattern = r'(.{0,50}' + re.escape(keyword) + r'.{0,50})'
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    concepts.append({'concept': keyword, 'context': matches[0].strip()})

        return concepts[:15]
