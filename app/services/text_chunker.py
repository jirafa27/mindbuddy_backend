import tiktoken
from typing import List
from app.core.config import settings


class TextChunkerService:
    """Сервис для разбиения текста на чанки с сохранением контекста"""

    def __init__(self):
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP

    def chunk_text(self, text: str, filename: str | None = None) -> List[str]:
        """
        Разбивает текст на чанки по заданному размеру с перекрытием.
        Старается не разрывать предложения.

        Args:
            text: Исходный текст для разбиения
            filename: Название файла (добавляется в начало текста для индексации)

        Returns:
            Список текстовых чанков
        """
        # Добавляем название файла в начало текста для лучшего поиска
        if filename:
            # Убираем расширение файла
            name_without_ext = filename.rsplit(".", 1)[0] if "." in filename else filename
            text = f"# {name_without_ext}\n\n{text}"
        # Разбиваем на предложения (простая эвристика)
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = len(self.encoding.encode(sentence))
            
            # Если предложение само по себе больше чанка, разбиваем его
            if sentence_tokens > self.chunk_size:
                # Сохраняем текущий чанк, если он не пустой
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
                
                # Разбиваем большое предложение на части
                large_chunks = self._chunk_large_text(sentence)
                chunks.extend(large_chunks)
                continue

            # Проверяем, поместится ли предложение в текущий чанк
            if current_tokens + sentence_tokens <= self.chunk_size:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
            else:
                # Сохраняем текущий чанк
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                
                # Начинаем новый чанк с перекрытием
                if chunks and self.chunk_overlap > 0:
                    # Берем последние предложения из предыдущего чанка для перекрытия
                    overlap_text = " ".join(current_chunk[-2:]) if len(current_chunk) >= 2 else current_chunk[-1] if current_chunk else ""
                    overlap_tokens = len(self.encoding.encode(overlap_text))
                    
                    if overlap_tokens <= self.chunk_overlap:
                        current_chunk = [overlap_text, sentence]
                        current_tokens = overlap_tokens + sentence_tokens
                    else:
                        current_chunk = [sentence]
                        current_tokens = sentence_tokens
                else:
                    current_chunk = [sentence]
                    current_tokens = sentence_tokens

        # Добавляем последний чанк, если он не пустой
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks if chunks else [text]

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Простое разбиение текста на предложения.
        Улучшенная версия может использовать библиотеку nltk или spaCy.
        """
        # Удаляем лишние пробелы
        text = " ".join(text.split())
        
        # Разбиваем по знакам препинания
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Фильтруем пустые предложения
        return [s.strip() for s in sentences if s.strip()]

    def _chunk_large_text(self, text: str) -> List[str]:
        """
        Разбивает большой текст на чанки фиксированного размера.
        Используется для предложений, которые больше размера чанка.
        """
        tokens = self.encoding.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), self.chunk_size - self.chunk_overlap):
            chunk_tokens = tokens[i:i + self.chunk_size]
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
        
        return chunks

