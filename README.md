# MindBuddy Backend 🧠 (Core Engine)

Центральный узел мультиагентной системы управления знаниями. Этот репозиторий содержит серверную логику, оркестрацию нейросетевых агентов и систему индексации данных.

---

## 🚀 Функциональная роль

Бэкенд реализует **Multi-Agent RAG** (Retrieval-Augmented Generation) архитектуру, превращая разрозненные данные пользователя в структурированную базу знаний с доступом через API.

### Основные задачи:
* **Orchestration:** Управление жизненным циклом агентов через `LangGraph`.
* **Vector Search:** Индексация и семантический поиск по `pgvector`.
* **Data Processing:** Асинхронная обработка файлов (PDF, YouTube, Web) через `Celery`.
* **API Service:** Обеспечение взаимодействия между Telegram-ботом и Desktop-клиентом.

---

## 🛠 Технологический стек

* **Core:** Python 3.10+, FastAPI
* **AI Framework:** LangChain / LangGraph
* **Database:** 
    * `PostgreSQL 16` + `pgvector` (основные данные и векторный индекс)
    * `SQLite` (изолированные таблицы созданные пользователями)
* **Storage:** MinIO (S3-совместимое хранилище файлов)
* **Task Queue:** RabbitMQ + Celery
* **Cache:** Redis
* **Environment:** Docker / Docker Compose

---

## 🤖 Архитектура Агентов (Workflow)

Бэкенд использует графовую модель для обработки запросов. Логика распределена между специализированными агентами:



1.  **Router:** Определяет намерение (поиск, правка файла, диалог).
2.  **Context Manager:** Подгружает нужный `namespace` (Работа/Личное).
3.  **RAG Agent:** Извлекает контекст из векторной БД.
4.  **Validation Agent:** Проверяет ответ на отсутствие галлюцинаций.
5.  **Editor Agent:** Формирует команды на изменение локальных файлов через Desktop Agent.

---

## 📂 Структура проекта

```plaintext
mindbuddy_backend/
├── app/
│   ├── api/            # Эндпоинты FastAPI (v1)
│   ├── core/           # Конфигурация и настройки (env)
│   ├── db/             # Модели SQLAlchemy и сессии
│   ├── repositories/   # Репозитории для работы с БД
│   ├── schemas/        # Pydantic схемы
│   ├── services/       # Бизнес-логика (эмбеддинги, обработка файлов, MinIO)
│   ├── tasks/          # Задачи Celery для тяжелой обработки
│   └── utils/          # Утилиты (file readers)
├── tests/              # Юнит-тесты и тесты агентов
├── docker-compose.yml
└── requirements.txt
```

---

## 🚀 Быстрый старт

### 1. Клонирование и настройка

```bash
git clone https://github.com/yourusername/mindbuddy-backend.git
cd mindbuddy-backend
cp env.example .env
# Отредактируйте .env с вашими ключами API
```

### 2. Запуск через Docker Compose

#### Режим разработки (с hot reload):

```bash
docker-compose up -d
```

При изменении кода в `app/` сервер автоматически перезапустится! ⚡

#### Режим production (без hot reload):

```bash
docker-compose -f docker-compose.prod.yml up -d
```

Сервисы будут доступны:
- **Backend API:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs
- **MinIO Console:** http://localhost:9001 (admin/admin)
- **RabbitMQ Management:** http://localhost:15672 (guest/guest)
- **PostgreSQL:** localhost:5432

### 3. Проверка здоровья системы

```bash
curl http://localhost:8000/health
```

### 4. Интеграционные тесты (полный прогон: БД + MinIO)

Поднимите тестовые сервисы (PostgreSQL на 5433, MinIO на 9002), затем запустите pytest:

```bash
docker compose -f docker-compose.test.yml up -d
# Подождите пару секунд, пока MinIO поднимется
pytest tests/
```

Тесты используют `localhost:5433` (БД) и `localhost:9002` (MinIO) — моки не используются.

---

## 📦 Основные возможности

### Загрузка файлов

```bash
curl -X POST http://localhost:8000/api/v1/files/upload \
  -H "Authorization: Bearer <your_jwt_token>" \
  -F "file=@document.pdf" \
  -F "namespace_id=1"
```

**Desktop Watcher** использует отдельный эндпоинт с watcher-токеном:
```bash
curl -X POST http://localhost:8000/api/v1/sync/upload \
  -F "token=<watcher_token>" \
  -F "device_id=MY-PC" \
  -F "vault_name=MyVault" \
  -F "relative_path=subfolder/document.md" \
  -F "content_hash=<sha256>" \
  -F "desktop_updated_at=2026-01-01T12:00:00Z" \
  -F "file=@document.md"
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

---

## 🗄️ Хранилище файлов (MinIO)

Проект использует MinIO для хранения загруженных файлов:

- **Bucket:** `mindbuddy-files`
- **Структура:** `users/{user_id}/namespaces/{namespace_id}/{timestamp}_{filename}`
- **Преимущества:**
  - Масштабируемое хранилище
  - S3-совместимый API
  - Возможность перепроцессинга файлов
  - Версионирование (опционально)

---

## 🔧 Разработка

### Установка зависимостей локально

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 📝 Поддерживаемые форматы файлов

- `.txt` - текстовые файлы
- `.md` - Markdown документы  
- `.pdf` - PDF документы
- `.docx` - Word документы