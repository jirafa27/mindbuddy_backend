from typing import List, Optional
from sqlalchemy.orm import Session

from app.db.models import File as FileModel, VectorEmbedding


class FileRepository:
    """Репозиторий для работы с файлами"""

    def __init__(self, db: Session):
        self.db = db

    def get_by_id(self, file_id: int) -> Optional[FileModel]:
        """Получает файл по ID"""
        return self.db.query(FileModel).filter(FileModel.id == file_id).first()

    def delete(self, file_id: int) -> bool:
        """
        Удаляет файл по ID.
        Commit должен быть выполнен вызывающим кодом.
        
        Returns:
            True если файл был удален, False если не найден
        """
        db_file = self.get_by_id(file_id)
        if db_file:
            self.db.delete(db_file)
            return True
        return False


class VectorEmbeddingRepository:
    """Репозиторий для работы с векторными эмбеддингами"""

    def __init__(self, db: Session):
        self.db = db

    def create_batch(
        self,
        file_id: int,
        namespace_id: int,
        chunks: List[str],
        embeddings: List[List[float]],
    ) -> List[VectorEmbedding]:
        """
        Создает эмбеддинги для всех чанков.
        Commit должен быть выполнен вызывающим кодом.
        
        Args:
            file_id: ID файла
            namespace_id: ID пространства знаний
            chunks: Список текстовых чанков
            embeddings: Список эмбеддингов
            
        Returns:
            Список созданных VectorEmbedding объектов
        """
        vector_embeddings = []
        for idx, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
            vector_embedding = VectorEmbedding(
                file_id=file_id,
                namespace_id=namespace_id,
                chunk_index=idx,
                chunk_text=chunk_text,
                embedding=embedding,
            )
            vector_embeddings.append(vector_embedding)

        self.db.add_all(vector_embeddings)
        return vector_embeddings
