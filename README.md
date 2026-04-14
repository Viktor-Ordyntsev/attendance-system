# Attendance System

система учёта посещаемости с двумя частями:

- `backend` на `FastAPI` принимает события посещаемости и хранит данные в памяти процесса.
- `recognition-client` получает кадры с камеры, детектит лицо, строит face embedding и отправляет событие в backend после подтверждения распознавания.

Проект сейчас рассчитан на локальный запуск на одной машине под Windows и Python 3.13.

## Структура проекта

```text
attendance-system/
|-- backend/
|   |-- app/main.py
|   `-- requirements.txt
|-- recognition-client/
|   |-- app/main.py
|   |-- app/core/
|   |-- app/models/face_recognizer.onnx
|   |-- samples/
|   `-- requirements.txt
`-- docs/
```

## Как это работает

### Backend

`backend/app/main.py` поднимает HTTP API и хранит:

- участников (`participants_store`)
- события (`events_store`)
- отметки посещаемости (`attendance_events_store`)

Хранилище in-memory. После перезапуска backend все данные теряются.

### Recognition Client

`recognition-client/app/main.py` запускает пайплайн:

1. открывает камеру
2. детектит лица через `insightface`
3. строит embedding лица
4. сравнивает embedding с локальной reference-базой
5. после нескольких подтверждённых кадров отправляет `POST /attendance-events` в backend

Сейчас reference-база собирается при старте из файлов:

- `recognition-client/samples/person1.jpg`
- `recognition-client/samples/person2.jpg`

## Требования

- Windows
- Python 3.13
- веб-камера или IP/USB camera stream
- доступ к локальному файлу модели `recognition-client/app/models/face_recognizer.onnx`

## Установка

### 1. Backend

```powershell
cd backend
python -m venv .venv_backend
.\.venv_backend\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Recognition Client

```powershell
cd recognition-client
python -m venv .venv_client
.\.venv_client\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Запуск

### 1. Запустить backend

Из каталога `backend`:

```powershell
.\.venv_backend\Scripts\Activate.ps1
uvicorn app.main:app --reload
```

По умолчанию backend будет доступен на:

- `http://127.0.0.1:8000`

### 2. Подготовить данные для recognition-client

Проверь:

- модель лежит по пути `recognition-client/app/models/face_recognizer.onnx`
- эталонные фотографии лежат в `recognition-client/samples/`

Если хочешь использовать свои фото, замени:

- `recognition-client/samples/person1.jpg`
- `recognition-client/samples/person2.jpg`

И при необходимости обнови список `samples` в `recognition-client/app/main.py`.

### 3. Запустить recognition-client

Из каталога `recognition-client`:

```powershell
.\.venv_client\Scripts\Activate.ps1
python -m app.main
```

Клавиша выхода:

- `q`

## Настройка камеры

Источник камеры задаётся через переменную окружения `CAMERA_SOURCE`.

Поддерживаются:

- индекс устройства, например `0`
- путь к видеофайлу
- URL потока камеры

Пример для PowerShell:

```powershell
$env:CAMERA_SOURCE="0"
python -m app.main
```

Пример для RTSP/HTTP-стрима:

```powershell
$env:CAMERA_SOURCE="rtsp://user:password@host:554/stream"
python -m app.main
```

Если переменная не задана, используется камера `0`.

## Модель распознавания

Клиент использует кастомную ONNX-модель распознавания лиц:

- путь по умолчанию: `recognition-client/app/models/face_recognizer.onnx`

В текущем коде:

- детектор лиц остаётся на `insightface`
- embedding считается через `onnxruntime`
- preprocess в `CustomFaceRecognizer` использует `112x112`, `RGB` и нормализацию в диапазон `[-1, 1]`

Если меняешь модель:

- пересобери reference embeddings, то есть перезапусти клиент с актуальными `samples`
- проверь `input_size`
- проверь preprocess под требования новой модели
- перекалибруй `threshold` и `min_margin` в `recognition-client/app/main.py`

## Логика распознавания

Матчинг работает через cosine similarity в `recognition-client/app/core/matcher.py`.

Ключевые параметры:

- `threshold`: минимальный score для признания совпадения
- `min_margin`: минимальный отрыв лучшего кандидата от второго

Если в логах часто появляется:

- `below_threshold`

значит score слишком низкий.

Если часто появляется:

- `low_margin_to_second_candidate`

значит лучший и второй кандидат слишком близки, и система считает распознавание неоднозначным.

## Backend API

### `GET /`

Проверка, что API запущен.

### `GET /health`

Возвращает статус сервиса.

### `GET /participants`

Список участников.

### `POST /participants`

Создание участника.

Пример тела:

```json
{
  "full_name": "Ivan Ivanov",
  "group_name": "Group A"
}
```

### `GET /events`

Список мероприятий.

### `POST /events`

Создание мероприятия.

Пример тела:

```json
{
  "title": "Math Lecture",
  "location": "Room 101",
  "start_at": "2026-04-15T09:00:00Z"
}
```

### `GET /attendance-events`

Список принятых событий посещаемости.

### `POST /attendance-events`

Приём события посещаемости от recognition-client.

Пример тела:

```json
{
  "event_id": 1,
  "participant_id": 1,
  "label": "Person 1",
  "score": 0.97,
  "timestamp": "2026-04-15T10:15:00Z"
}
```

Условия успешной записи:

- событие `event_id` существует
- участник `participant_id` существует
- для пары `event_id + participant_id` ещё нет записи

## Типовой сценарий запуска

1. Запусти backend.
2. Создай участников через `POST /participants`.
3. Создай мероприятие через `POST /events`.
4. Убедись, что `event_id` в `recognition-client/app/core/pipeline.py` соответствует созданному событию.
5. Положи эталонные фото в `recognition-client/samples/`.
6. Запусти recognition-client.
7. Наведи камеру на лицо и дождись подтверждённого распознавания.

## Известные ограничения

- backend не использует БД и теряет данные после перезапуска
- `.env.example` файлы пока пустые
- reference-база жёстко задана в `recognition-client/app/main.py`
- `backend_url` и `event_id` сейчас захардкожены в `recognition-client/app/core/pipeline.py`
- клиент выполняет inference и HTTP-запросы синхронно в одном процессе
- CUDA provider для `onnxruntime` не настроен, если установлен только `onnxruntime`
- детектор и embedding-модель могут требовать дополнительной калибровки порогов на реальных данных

## Отладка

### Ошибка `NoSuchFile` для ONNX-модели

Проверь, что файл существует по пути:

```text
recognition-client/app/models/face_recognizer.onnx
```

### На видео всегда `unknown`

Сначала проверь логи клиента:

- если причина `below_threshold`, понижай `threshold` или улучшай эталонные фото
- если причина `low_margin_to_second_candidate`, уменьшай `min_margin` и улучшай reference-базу

### Backend отвечает `404 Event not found`

Значит в backend нет события с тем `event_id`, который указан в `recognition-client/app/core/pipeline.py`.

### Backend отвечает `404 Participant not found`

Значит `participant_id` из reference-базы не создан в backend.

### Backend отвечает `409 Participant already marked for this event`

Это штатное поведение: один и тот же участник не может быть отмечен дважды для одного события.
