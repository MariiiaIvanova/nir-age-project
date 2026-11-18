# Единый пайплайн оценки возраста по голосу

Из фиксированного train (TIMIT, NNCES, AgeVoxCeleb) один раз выделяется validation: около 10% дикторов каждого корпуса с распределением по медианному возрасту. Все экстракторы используют одинаковые сохранённые списки. После каждой из 30 эпох рассчитываются validation MAE, RMSE, bias и CCC. Лучший checkpoint выбирается по среднему MAE трёх validation-корпусов и тестируется отдельно на TIMIT, NNCES, AgeVoxCeleb, SeniorTalk и VoiceBiometricAge.

## Структура и этапы пайплайна

```text
app/
  prepare.py             # Проверка корпусов и подготовка рабочих списков
  split_validation.py    # Фиксированное разделение train/validation по дикторам
  benchmark.py           # Формат списков и перенос путей
  audio_data.py          # Загрузка, обработка аудио и формирование батчей
  extractors/
    __init__.py          # Выбор экстрактора по имени
    whisper.py           # Адаптер Whisper tiny
    wavlm.py             # Адаптер WavLM Base
  model.py               # Общая сеть оценки возраста после экстрактора
  train.py               # Loss, оптимизатор, обучение, checkpoint и resume
  evaluation.py          # Validation и отдельная оценка пяти тестов
  runtime.py             # Запись JSON и измерение времени
tests/
  check_pipeline.py      # Проверка разбиения, обучения, тестов, resume и таймера
  check_extractor.py     # Формы, градиенты и память выбранного экстрактора
tools/
  import_train.py        # Однократный импорт train из старого проекта
  download_weights.py   # Официальные веса выбранного экстрактора
  run_experiment.sh     # Общий запуск экстрактора с логом
  download_seniortalk.py # Получение SeniorTalk при отсутствии локальной копии
  pack.py                # Архив для передачи проекта
  run_whisper_validation.sh # Запуск Whisper на GPU 0 с сохранением лога
benchmark/               # Фиксированные исходные списки; не аудиофайлы
weights/                 # Предобученные веса
output/                  # Подготовленные списки и результаты экспериментов
logs/                    # Вывод запусков
Makefile                 # Команды подготовки, обучения, проверок и упаковки
README.md                # Структура, протокол и инструкции запуска
.env                     # Локальные пути к корпусам и номер GPU
env.example              # Образец конфигурации
Dockerfile, compose.yaml # Необязательный запуск через Docker
requirements*.txt        # Зависимости обучения и загрузчика данных
```

Последовательность работы:

1. `prepare.py` читает шесть фиксированных списков из `benchmark/`, проверяет аудио и сохраняет рабочие пути с аудитом. Повторять для каждого экстрактора не нужно.
2. `split_validation.py` один раз создаёт train и validation из трёх обучающих корпусов. Пять test остаются неизменными. Все модели используют один сохранённый каталог и одинаковые хеши списков.
3. `audio_data.py` подаёт моно 16 кГц, 4 секунды; специфичный препроцессинг выполняет адаптер в `extractors/`.
4. `model.py` соединяет экстрактор с общей сетью регрессии. Интерфейс адаптера: waveform `[B,T]` → признаки `[B,D,T′]`, атрибут `output_dim=D`.
5. `train.py` обучает 30 эпох: AdamW, lr=1e-4, weight_decay=1e-4, batch=32, seed=42, cosine scheduler, loss `1−CCC`. После каждой эпохи `evaluation.py` считает validation; лучший checkpoint выбирается по среднему MAE трёх корпусов.
6. Лучшая модель тестируется отдельно на пяти наборах. В новой папке эксперимента сохраняются конфигурация, история, checkpoint, метрики, предсказания и замеры времени.

Обучение, подготовка и проверка — отдельные действия. `tests/check_pipeline.py` запускает короткую техническую проверку на реальном аудио и GPU 0; это не научный эксперимент. Старые результаты при изменении структуры кода не пересчитываются.

Добавление экстрактора: новый модуль в `app/extractors/` и регистрация в `__init__.py`. Общую сеть, списки, loss, правила validation/test и параметры протокола сохраняем. Особенности входа и предобученного checkpoint документируем для каждой модели. Реализованы `whisper_tiny` и `wavlm_base`. Выбор экстрактора и локальных весов задаётся через `--extractor` и `--weights`.

## Запуск на nid-sin-host8

Нужны Linux, Docker Engine с Compose v2 и NVIDIA Container Toolkit, make и системный python3 для однократного импорта. Нужны права на Docker и чтение корпусов. Все команды выполняются в папке с этим README. Распакуй проект рядом с workPipeline.

```bash
cp env.example .env
nvidia-smi
```

В примере уже указаны пути сервера Ивановой. Если карта 0 занята, укажи номер свободной карты в GPU_ID. Docker подключает только выбранную карту, внутри контейнера это cuda:0.

```bash
make build
make prepare
```

При первом запуске make prepare берёт оригинальный all_train.txt из LEGACY_SPLITS, сверяет Git blob SHA-1 и помещает переносимый список в benchmark/all_train.tsv.gz. Затем проверяет доступность и декодирование аудио во всех шести списках. Интернет для подготовки данных не требуется. Пять тестовых списков уже включены. Если all_train.txt изменён, возьми оригинальный файл из репозитория MariiiaIvanova/workPipeline, lab6/benchmark/splits/all_train.txt с Git blob 9fabf1c721d10b6734dd2f39d70af6cf895805af. После успешного импорта LEGACY_SPLITS больше не используется.

```bash
cat output/current/audit.json
head output/current/age_differences.tsv
```

Для модели положи имеющийся многоязычный tiny.pt в weights/tiny.pt. Если весов нет и сервер имеет доступ в интернет, выполни один раз `make weights`. Для сборки образа также нужен доступ к реестру Docker, пакетам Debian и Python. Затем:

```bash
python3 app/split_validation.py --source output/current --dest output/validation_seed42 --seed 42
make smoke
make train
```

Smoke использует два реальных примера разного возраста, проверяет конечность loss и градиентов, делает шаг AdamW и вызов модели в eval. Длительный запуск по умолчанию сохраняет результаты в output/runs/whisper_tiny_ccc/. Выбери новое имя папки для нового эксперимента. Для продолжения после последней законченной эпохи выполни `make resume` с теми же параметрами и подготовленными списками.

## Один конфиг с путями

| Переменная в .env | Что должно лежать в папке | Путь в контейнере |
|---|---|---|
| TIMIT_ROOT | scp/{train,test}/{wav.scp,utt2age_reg} и wav16k/{train,test} | /data/timit |
| NNCES_AUDIO_ROOT | Содержимое внутреннего Read_Speech_Data, включая 10Y и 12Y | /data/nnces |
| NNCES_SCP_ROOT | wav.scp и utt2age_reg | /data/nnces_scp |
| AGEVOX_AUDIO_ROOT | Аудио VoxCeleb2, начиная с id…/видео/запись | /data/agevoxceleb |
| VOICEBIO_ROOT | Внутренняя папка voice_biometrics_age с id… | /data/voicebio |
| SENIORTALK_ROOT | Папка полного корпуса с wav/ | /data/seniortalk |

Данные подключаются read-only. Для TIMIT ключ записи сопоставляется с wav.scp только внутри исходного split, для NNCES используется часть пути после последнего Read_Speech_Data/. Метки возраста из scp связываются по ID. При расхождении возраст из фиксированного бенчмарка сохраняется, оба значения записываются в age_differences.tsv для проверки. Для остальных корпусов метки уже сохранены в переносимых списках. Каталог AgeVoxCeleb с разметкой и таблица VoiceBio нужны при построении нового бенчмарка с нуля.

Каждая подготовка создаёт отдельный каталог output/splits-… и после успеха переключает ссылку output/current. Аудит содержит числа исходных, выбранных и оставшихся строк и хеши списков. Повторные пути, пересечение train/test по ID диктора внутри корпуса, отсутствие файла или неоднозначный TIMIT ID останавливают подготовку. При ошибке предыдущая успешная версия сохраняется.

| Список | Исходных записей |
|---|---:|
| all_train | 159287 |
| timit_test | 1680 |
| nnces_test | 1500 |
| agevoxceleb_test | 16050 |
| seniortalk_test | 50648 |
| voicebio_test | 791 |

Быстрый предварительный проход по путям: `make prepare ARGS=--paths-only`. Для полного запуска затем выполни `make prepare`. Чтобы разрешить исключение только повреждённых при декодировании записей, используй `make prepare ARGS=--exclude-decode-errors`, затем проверь excluded.tsv и audit.json. Ошибки отсутствующих файлов остаются обязательными к исправлению.

Короткий тест SeniorTalk: `make prepare ARGS="--senior-per-speaker 20 --seed 42"`. Этот выбор относится ко всему эксперименту. Используй один и тот же подготовленный каталог для всех сравниваемых экстракторов. Полный SeniorTalk используется по умолчанию.

## Обучение и результаты

Сигнал приводится к моно, 16 кГц и четырём секундам. Длинные записи обрезаются случайно на train и по центру на test, короткие дополняются нулями. Применяется нормировка по пику. Log-mel (80 полос) вычисляется для каждой записи отдельно. Все четыре блока энкодера tiny дообучаются. После TDNN (1024) считаются среднее и стандартное отклонение по времени, затем maxout (512), BatchNorm и согласованная RegAgeHead2 с clamp от 0 до 100. Смещение последнего слоя инициализируется средним возрастом train.

По умолчанию AdamW, lr 0.0001, weight decay 0.0001, cosine scheduler, batch 32, seed 42, loss 1 − CCC по батчу. Неполный последний батч сохраняется. Если остаётся одна запись, она присоединяется к предыдущему батчу. Для MSE задай отдельную папку:

```bash
make train ARGS="--loss mse --run /output/runs/whisper_tiny_mse"
```

Папка эксперимента содержит best.pt с лучшими весами, selection.json с выбранной эпохой, config.json с настройками и хешами, history.json, last.pt с состоянием модели, оптимизатора, планировщика и генераторов, metrics.json и пять CSV с предсказаниями. В metrics.json записываются MAE и RMSE в годах, средняя знаковая ошибка (прогноз минус возраст) и MAE с равным весом дикторов. После прерывания во время теста make resume возобновляет оценку из последнего checkpoint.

Для нового экстрактора реализуй модуль с входом waveform [B,T], выходом [B,D,T′] и атрибутом output_dim, зарегистрируй его в app/extractors/__init__.py и передай имя через --extractor. Препроцессинг относится к конкретному экстрактору. Реализованы whisper_tiny и wavlm_base.

## Передача проекта

После первой подготовки выполни:

```bash
make pack
```

Архив dist/nir-age.zip содержит код, Dockerfile, compose.yaml, env.example и все шесть переносимых списков, включая train. Аудио, рабочий .env, веса и результаты в архив не входят. Получатель распаковывает архив, заполняет .env путями своих копий тех же корпусов и выполняет команды сборки, подготовки и запуска. Для офлайн-сдачи можно также передать файл образа и веса отдельно:

```bash
docker image save -o dist/nir-age-image.tar nir-age:whisper-tiny
# На принимающей машине вместо make build:
docker image load -i dist/nir-age-image.tar
```

## Получение корпусов

На описанном сервере используй существующие копии. На новой машине размести доступные тебе копии TIMIT, NNCES, VoxCeleb2 и VoiceBiometricAge в структуре из таблицы. SeniorTalk можно подготовить отдельно в пустом каталоге. Команды ниже используют отдельное окружение загрузчика:

```bash
python3.10 -m venv ../seniortalk_venv
../seniortalk_venv/bin/python -m pip install -r requirements-download.txt
. ./.env
../seniortalk_venv/bin/python tools/download_seniortalk.py "$SENIORTALK_ROOT"
```

## Если Docker пока недоступен

На сервере с системным FFmpeg и Python 3.10 можно использовать тот же код и .env в nir_venv. Если окружение уже создано, начни с его активации.

```bash
python3.10 -m venv ../nir_venv
source ../nir_venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
make prepare-native
python app/split_validation.py --source output/native/current --dest output/native/validation_seed42 --seed 42
mkdir -p weights
# Помести tiny.pt в weights/ или скачай веса один раз:
python -c "import whisper; whisper.load_model('tiny', device='cpu', download_root='weights')"
. ./.env
export CUDA_VISIBLE_DEVICES="$GPU_ID"
python app/train.py --splits output/native/validation_seed42 --weights weights/tiny.pt --smoke
python app/train.py --splits output/native/validation_seed42 --weights weights/tiny.pt --run output/native/runs/whisper_tiny_ccc
```

Нативная подготовка записывает пути хоста в output/native. Docker использует пути контейнера в output/current. Фиксированные списки benchmark/ общие.

## Фиксированная validation (протокол v1)

Повторное декодирование аудио не требуется. `app/split_validation.py` проверяет хеши исходных списков и отсутствие пересечений дикторов train/test. В каждом корпусе дикторы сортируются по медианному возрасту и делятся на группы; из каждой группы выбирается один диктор фиксированным генератором. Созданный каталог нельзя перезаписывать: используйте его для всех моделей. `partition.json` хранит seed, метод, размеры и хеши. Исходные `all_train` и пять тестов копируются без изменений; обучение читает `train.txt`.

AdamW, lr=1e-4, weight_decay=1e-4, batch=32, 30 эпох, cosine scheduler, loss=1-CCC. Веса экстрактора дообучаются. Validation использует центральные 4 секунды, без случайной обрезки. Критерий выбора — среднее трёх MAE по записям (равный вес корпусов). При равенстве сохраняется более ранняя эпоха. Последний checkpoint включает оптимизатор, scheduler, RNG и лучшие веса для надёжного resume. Старые эксперименты без validation несовместимы с этим протоколом и остаются отдельными результатами.

Запуск нового эксперимента требует новой папки `--run`. Для продолжения используйте ту же команду с `--resume`, теми же списками и числом эпох. В консоль выводятся validation после каждой эпохи и метрики каждого из пяти тестов.

### Измерение времени

Новые запуски записывают `timing.json`: отдельная сессия на каждый запуск/resume, начало и конец в UTC, статус, общее прошедшее время и длительность обучения/validation каждой эпохи и итогового тестирования. Используется монотонный `time.perf_counter`; GPU синхронизируется на границах этапов. Время обучения включает чтение аудио; общее время также включает инициализацию и сохранение checkpoint. При resume время простоя между запусками не включается. При жёстком завершении процесса последняя запись может оставаться со статусом `running`; потерянное время не восстанавливается. Старые результаты не получают ретроспективных замеров.

### Команды без Docker

После активации Python-окружения доступны `make split-native`, `make smoke-native`, `make train-native`, `make resume-native`, `make check`. Разбиение создаётся только один раз. Для нового обучения задайте новую папку: `make train-native NATIVE_RUN=output/native/runs/new_experiment`. Параметры передаются через `ARGS`, например `ARGS="--extractor whisper_tiny"`. Выбор GPU остаётся через `CUDA_VISIBLE_DEVICES`. Эти команды используют те же defaults `app/train.py`, что и прямой запуск Python.

## WavLM Base и сохранение запуска Whisper

Используется официальный `torchaudio.pipelines.WAVLM_BASE` (LibriSpeech 960 h), не Base+ и не Large: https://docs.pytorch.org/audio/2.4.0/generated/torchaudio.pipelines.WAVLM_BASE.html. Зависимости не меняются. Адаптер строит публичную модель `torchaudio.models.wavlm_base` с настройками bundle версии 2.4.1: projection/attention/output dropout=0.1, FF intermediate dropout=0, LayerDrop=0.05. Эти настройки относятся к предобученному экстрактору.

Вход — общий waveform 16 кГц, 4 секунды после существующей обработки. Дополнительная нормировка waveform для WAVLM_BASE не требуется. Выход последнего слоя — `[B,768,T]` (199 кадров для 64000 отсчётов). Все веса дообучаются; ASR-голова не добавляется. Общая сеть, AdamW, loss, batch, списки и выбор лучшего checkpoint сохранены. Для Whisper по-прежнему используются его log-mel и 384 признака.

```bash
source ../nir_venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
python tools/download_weights.py --extractor wavlm_base
python tests/check_extractor.py --extractor wavlm_base --weights weights/wavlm_base.pth --batch-size 32
bash tools/run_experiment.sh wavlm_base weights/wavlm_base.pth output/native/runs/wavlm_base_adamw_val_seed42
```

Скачивание выполняется отдельно. Обучение загружает только локальный файл; SHA-256 весов записывается в конфиг. Не запускайте повторную команду, пока эксперимент уже идёт.

Whisper остаётся доступен без изменения своей модели:

```bash
python app/train.py --extractor whisper_tiny --weights weights/tiny.pt --splits output/native/validation_seed42 --smoke
# Для нового полного эксперимента задайте новую папку:
bash tools/run_experiment.sh whisper_tiny weights/tiny.pt output/native/runs/whisper_new
```

Makefile также поддерживает явный выбор весов: `make train-native NATIVE_WEIGHTS=weights/wavlm_base.pth NATIVE_RUN=output/native/runs/wavlm_new ARGS="--extractor wavlm_base"`. Defaults Whisper сохранены.
