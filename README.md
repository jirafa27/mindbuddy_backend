# MindBuddy Backend

Серверная часть MindBuddy: FastAPI-приложение для хранения пользовательских материалов, RAG-поиска, суммаризации, диалогового управления файлами и синхронизации с desktop-клиентом.

Клиентская часть на React Native / TypeScript доступна по запросу.

---

## Функциональная роль

Backend объединяет REST API, сервисный слой, PostgreSQL/pgvector, объектное хранилище MinIO, Celery-задачи и LangGraph-граф обработки пользовательских запросов.

Основные задачи:

- обработка пользовательских сообщений через `/api/v1/ask`;
- загрузка, скачивание, переименование, удаление и замена файлов;
- организация материалов по пространствам знаний;
- извлечение текста из `txt`, `md`, `pdf`, `docx` и URL-источников;
- разбиение текста на чанки, генерация эмбеддингов и сохранение в `pgvector`;
- RAG-поиск по базе знаний и генерация ответа через LLM;
- суммаризация файлов, ссылок и материалов из истории диалога;
- синхронизация файлов с desktop watcher через `/api/v1/sync`;
- запуск фоновой обработки через Celery.

---

## Технологический стек

- **Core:** Python 3.10+, FastAPI, SQLAlchemy Async
- **Graph:** LangGraph, LangChain Core
- **Database:** PostgreSQL 16 + pgvector
- **Storage:** MinIO, S3-совместимое хранилище файлов и временных blob-объектов
- **Task Queue:** Celery + RabbitMQ
- **Cache / Result Backend:** Redis
- **LLM / Embeddings:** Ollama, Yandex Cloud, OpenRouter
- **Testing:** pytest, pytest-asyncio, httpx
- **Environment:** Docker, Docker Compose

---

## Архитектура обработки запросов

Обработка сообщений построена вокруг `ChatService` и LangGraph-графа из `app/graph/graph.py`.

Основные узлы графа:

1. **RouterNode** выполняет быструю маршрутизацию для простых сценариев: URL-only, file-only, override intent.
2. **IntentNode** определяет намерение пользователя через LLM, если сценарий не был определён детерминированно.
3. **ActionResolverNode** извлекает параметры действия: имя файла, пространство, поисковый запрос, URL или содержимое заметки.
4. **FileAgent** извлекает текст из файлов, разбивает его на чанки и генерирует эмбеддинги.
5. **SaveFileNode** сохраняет файл, пользовательскую связь и векторные представления.
6. **QueryEmbeddingNode** формирует эмбеддинг поискового запроса.
7. **ExecuteSearchNode** выполняет векторный или гибридный поиск по `pgvector`.
8. **MindBuddyAgent** формирует итоговый ответ на основе найденных фрагментов.
9. **SummaryNode** управляет сценариями суммаризации.
10. **CrudNode** выполняет операции над файлами и пространствами знаний.
11. **MultiActionNode** последовательно выполняет несколько действий из одного пользовательского сообщения.

---

## Структура проекта

```plaintext
mindbuddy-backend/
├── app/
│   ├── api/                    # FastAPI endpoints v1
│   ├── core/                   # Конфигурация, зависимости, исключения, middleware
│   ├── domain/                 # Доменные сущности и протоколы
│   ├── graph/                  # LangGraph state, graph builder и узлы агентов
│   ├── infrastructure/         # БД, репозитории, LLM, storage, workers, parsers
│   ├── schemas/                # Pydantic-схемы API
│   ├── services/               # Бизнес-логика приложения
│   └── utils/                  # File readers, URL/file helpers
├── tests/
│   └── integration/            # Интеграционные backend-тесты
├── docker-compose.yml          # Основной dev-контур
├── docker-compose.test.yml     # Тестовые PostgreSQL, MinIO, Redis, RabbitMQ
├── Dockerfile
└── requirements.txt
```

---

## Быстрый старт

### 1. Подготовка окружения

Создайте `.env` на основе требуемых переменных из `app/core/config.py`.

Минимально важные группы настроек:

- `DATABASE_URL`
- `REDIS_URL`
- `RABBITMQ_URL`
- `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `MINIO_BUCKET_NAME`
- `SECRET_KEY`
- настройки LLM-провайдера: Ollama, Yandex Cloud или OpenRouter

### 2. Запуск через Docker Compose

```bash
docker compose up -d
```

Сервисы будут доступны:

- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- MinIO Console: http://localhost:9001
- RabbitMQ Management: http://localhost:15672
- PostgreSQL: localhost:5432
- Ollama: http://localhost:11434

### 3. Проверка здоровья системы

```bash
curl http://localhost:8000/health
```

Ожидаемый ответ:

```json
{"status": "healthy"}
```

---

## API-модули

Роутеры подключаются с префиксом `/api/v1`.

- `auth` — регистрация и аутентификация;
- `users` — операции с пользователями;
- `files` — загрузка, скачивание, удаление и замена файлов;
- `user-files` — пользовательские представления файлов;
- `namespaces` — пространства знаний;
- `chat` — диалоговый endpoint `/ask` и история чатов;
- `summary` — суммаризация материалов;
- `content` — извлечение контента из внешних источников;
- `tasks` — статус фоновых задач;
- `sync` — синхронизация desktop watcher.

---

## Основные сценарии

### Запрос в чат

```bash
curl -X POST http://localhost:8000/api/v1/ask \
  -H "Authorization: Bearer <your_jwt_token>" \
  -H "Content-Type: application/json" \
  -d '{"question": "Что есть в моей базе знаний?"}'
```

### Загрузка файла

```bash
curl -X POST http://localhost:8000/api/v1/files/upload \
  -H "Authorization: Bearer <your_jwt_token>" \
  -F "file=@document.pdf" \
  -F "namespace_id=1"
```

### Скачивание файла

```bash
curl -X GET "http://localhost:8000/api/v1/files/download/1" \
  -H "Authorization: Bearer <your_jwt_token>" \
  --output downloaded_file.pdf
```

### Удаление файла

```bash
curl -X DELETE "http://localhost:8000/api/v1/files/1" \
  -H "Authorization: Bearer <your_jwt_token>"
```

### Синхронизация desktop watcher

```bash
curl -X POST http://localhost:8000/api/v1/sync/upload \
  -F "token=<watcher_token>" \
  -F "device_id=MY-PC" \
  -F "vault_name=Vault" \
  -F "relative_path=subfolder/document.md" \
  -F "content_hash=<sha256>" \
  -F "desktop_updated_at=2026-01-01T12:00:00Z" \
  -F "file=@document.md"
```

---

## Хранение и обработка материалов

MindBuddy использует комбинированное хранение:

- исходные файлы и временные blob-объекты сохраняются в MinIO;
- метаданные, пользователи, пространства, чаты и связи файлов хранятся в PostgreSQL;
- текстовые чанки и эмбеддинги сохраняются в `vector_embeddings`;
- поиск выполняется через pgvector и, при наличии текстового запроса, гибридный SQL с полнотекстовым поиском.

Поддерживаемые форматы файлов:

- `.txt`
- `.md`
- `.pdf`
- `.docx`

---

## Фоновые задачи

Celery используется для длительных операций, связанных с обработкой материалов:

- генерация эмбеддингов для сохранённых файлов и URL-источников;
- обработка задач после синхронизации;
- отдельные сценарии суммаризации.

RabbitMQ используется как брокер задач Celery, Redis — как backend результатов.

---

## Тестирование

Интеграционные тесты находятся в `tests/integration`.

Перед запуском поднимите тестовые сервисы:

```bash
docker compose -f docker-compose.test.yml up -d
```

Запуск тестов:

```bash
python -m pytest tests/
```

Запуск с покрытием:

```bash
python -m pytest --cov=app --cov-report=term-missing
```

Тестовый контур использует PostgreSQL на `localhost:5433`, MinIO на `localhost:9002`, Redis на `localhost:6380` и RabbitMQ на `localhost:5673`.

---

## Локальная разработка

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```