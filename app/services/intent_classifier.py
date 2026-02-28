"""IntentClassifier — классификация намерений через sentence-transformers."""
import logging
from typing import Optional, Tuple
from sentence_transformers import SentenceTransformer

import numpy as np

from app.core.enums import IntentType

logger = logging.getLogger(__name__)

# Имя модели для загрузки
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# Порог косинусного сходства
SIMILARITY_THRESHOLD = 0.45


# Эталонные фразы для каждого интента
INTENT_EXAMPLES = {
    IntentType.SUMMARIZE: [
        # Русский 
        "суммаризируй",
        "сделай краткое содержание",
        "перескажи кратко",
        "сделай выжимку",
        "резюмируй",
        "о чём это",
        "о чем это видео",
        "расскажи вкратце",
        "краткий пересказ",
        "дай краткое описание",
        "что там было",
        "в двух словах",
        # English
        "summarize",
        "summary",
        "give me a summary",
        "what is it about",
        "brief overview",
        "tldr",
    ],
    IntentType.RAG_QUERY: [
        # Русский
        "найди в моих заметках",
        "что я записывал про",
        "где у меня написано",
        "какие файлы у меня есть",
        "поищи в документах",
        "что говорится в",
        "найди информацию о",
        "есть ли у меня",
        "в каком файле",
        # English
        "search in my notes",
        "find in my documents",
        "what do I have about",
        "look up",
    ],
    IntentType.SAVE_FILE: [
        # Русский
        "сохрани",
        "запомни это",
        "добавь в базу",
        "загрузи файл",
        "сохрани ссылку",
        "добавь в заметки",
        # English
        "save this",
        "remember this",
        "add to my knowledge",
        "upload",
    ],
}


class IntentClassifier:
    """
    Классификатор намерений на базе sentence-transformers.
    
    Использует косинусное сходство между эмбеддингом запроса
    и эталонными фразами для определения интента.
    """
    
    _instance: Optional["IntentClassifier"] = None
    _initialized: bool = False
    
    def __new__(cls) -> "IntentClassifier":
        """Singleton pattern — одна модель на всё приложение."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Инициализация модели (выполняется один раз)."""
        if IntentClassifier._initialized:
            return
        
        logger.info("[IntentClassifier] Loading model: %s", MODEL_NAME)
        logger.info("[IntentClassifier] Downloading/loading weights (first run may take 1–2 min)...")
        
        try:
            self.model = SentenceTransformer(MODEL_NAME)
            logger.info("[IntentClassifier] Model weights loaded, encoding intent examples...")
            # Предвычисляем эмбеддинги эталонных фраз
            self.intent_embeddings: dict[str, np.ndarray] = {}
            for intent, examples in INTENT_EXAMPLES.items():
                embeddings = self.model.encode(examples, convert_to_numpy=True)
                self.intent_embeddings[intent] = embeddings
                logger.info("[IntentClassifier] Encoded %d examples for intent '%s'", len(examples), intent)
            
            IntentClassifier._initialized = True
            logger.info("[IntentClassifier] Model loaded successfully")
            
        except ImportError:
            logger.warning("[IntentClassifier] sentence-transformers not installed, falling back to keyword matching")
            self.model = None
            IntentClassifier._initialized = True
        except Exception as e:
            logger.error("[IntentClassifier] Failed to load model: %s", e)
            self.model = None
            IntentClassifier._initialized = True
    
    def classify(self, text: str) -> Tuple[Optional[str], float]:
        """
        Классифицирует текст по намерению.
        
        Args:
            text: Текст для классификации
            
        Returns:
            Tuple[intent, score]: Название интента и уверенность.
            Если score < SIMILARITY_THRESHOLD, intent = None
        """
        if not self.model or not text.strip():
            return None, 0.0
        
        # Получаем эмбеддинг запроса
        query_embedding = self.model.encode(text, convert_to_numpy=True)
        
        best_intent = None
        best_score = 0.0
        
        for intent, embeddings in self.intent_embeddings.items():
            # Косинусное сходство с каждой эталонной фразой
            similarities = self._cosine_similarity(query_embedding, embeddings)
            max_similarity = float(np.max(similarities))
            
            if max_similarity > best_score:
                best_score = max_similarity
                best_intent = intent
        
        logger.info("[IntentClassifier] Text: '%s' -> Intent: %s (score: %.3f)", 
                    text[:50], best_intent, best_score)
        
        if best_score < SIMILARITY_THRESHOLD:
            return None, best_score
        
        return best_intent, best_score
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Вычисляет косинусное сходство между вектором и матрицей."""
        # Нормализуем векторы
        a_norm = a / np.linalg.norm(a)
        b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
        return np.dot(b_norm, a_norm)
