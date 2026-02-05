# File Readers

Модульная система для чтения различных типов файлов.

## Структура

```
file_readers/
├── __init__.py          # Экспорты
├── base.py              # Базовый абстрактный класс
├── txt_reader.py        # Reader для .txt файлов
├── md_reader.py         # Reader для .md файлов
├── pdf_reader.py        # Reader для .pdf файлов
├── docx_reader.py       # Reader для .docx файлов
└── factory.py           # Фабрика для создания нужного reader'а
```

## Использование

### Через фабрику (рекомендуется)

```python
from app.utils.file_readers import FileReaderFactory

factory = FileReaderFactory()

# Получить reader для конкретного типа файла
reader = factory.get_reader("pdf")
text = reader.read(file_content)

# Получить список всех поддерживаемых расширений
extensions = factory.get_supported_extensions()
```

### Напрямую

```python
from app.utils.file_readers import PdfReader

reader = PdfReader()
text = reader.read(file_content)
```

## Добавление нового типа файла

1. Создайте новый reader, наследуясь от `BaseFileReader`:

```python
from .base import BaseFileReader

class NewReader(BaseFileReader):
    def read(self, file_content: bytes) -> str:
        # Ваша логика извлечения текста
        return extracted_text
    
    @property
    def supported_extensions(self) -> list[str]:
        return ["new_ext"]
```

2. Зарегистрируйте его в `factory.py`:

```python
def _register_readers(self):
    readers = [
        TxtReader(),
        MarkdownReader(),
        PdfReader(),
        DocxReader(),
        NewReader(),  # Добавьте сюда
    ]
    ...
```

3. Экспортируйте в `__init__.py`:

```python
from .new_reader import NewReader

__all__ = [
    ...,
    "NewReader",
]
```

## Поддерживаемые форматы

- `.txt` - текстовые файлы (UTF-8)
- `.md` - Markdown файлы
- `.pdf` - PDF документы
- `.docx` - Word документы
