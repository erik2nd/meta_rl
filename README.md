# Meta-RL Thesis: Experimental Framework

Реализация и сравнение методов мета-обучения с подкреплением:
- **MAML** (Model-Agnostic Meta-Learning) — текущий статус
- **RL?** (Recurrent Meta-RL) — в планах
- **PEARL** (Probabilistic Embeddings for Actor-critic RL) — в планах
- **Hybrid (PEARL + RNN)** — собственная модификация

## Установка

```bash
# Создать venv (рекомендуется)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate    # Windows

# Установить зависимости
pip install -r requirements.txt
```

Если Meta-World не устанавливается через pip, поставьте вручную:
```bash
pip install git+https://github.com/Farama-Foundation/Metaworld.git@master
```

## Структура проекта

```
meta-rl-thesis/
??? configs/                # YAML конфиги гиперпараметров
??? algorithms/             # реализации алгоритмов
?   ??? maml/
?       ??? policy.py       # Gaussian MLP policy
?       ??? maml.py         # сам алгоритм MAML
?       ??? trainer.py      # обучающий цикл
??? envs/                   # обертки для Meta-World
??? utils/                  # логгер, метрики, сбор траекторий
??? scripts/                # точки входа
??? results/                # чекпоинты моделей
??? logs/                   # tensorboard/wandb логи
```

## Запуск

### Обучение MAML на ML1 (одна задача reach-v2)

```bash
python scripts/train_maml.py --config configs/maml_config.yaml
```

### Переопределение параметров из командной строки

```bash
python scripts/train_maml.py --config configs/maml_config.yaml \
    --override experiment.seed=7 training.total_meta_iterations=100
```

### Запуск с разными seed (для статистики)

```bash
for seed in 1 2 3 4 5; do
    python scripts/train_maml.py --config configs/maml_config.yaml \
        --override experiment.seed=$seed experiment.name=maml_seed_$seed
done
```

### Просмотр логов в TensorBoard

```bash
tensorboard --logdir=./logs
# открыть http://localhost:6006
```

## Конфигурация

Основные параметры в `configs/maml_config.yaml`:

| Параметр | Значение по умолчанию | Описание |
|---|---|---|
| `maml.inner_lr` | 0.05 | learning rate внутреннего цикла (?) |
| `maml.outer_lr` | 3e-4 | learning rate внешнего цикла (?) |
| `maml.num_inner_steps` | 1 | сколько шагов адаптации |
| `maml.first_order` | false | FOMAML (без вторых производных) |
| `training.tasks_per_meta_batch` | 20 | задач на одну мета-итерацию |
| `training.trajectories_per_task` | 10 | траекторий на support set |

## Замечания по реализации

- **Inner loop loss:** vanilla policy gradient (REINFORCE) с baseline = средний return по траекториям. Не используется learned critic, как в оригинальной статье Finn et al. 2017.
- **GAE:** доступна в `algorithms/maml/maml.py:compute_gae`, но в текущем pipeline используется простая discounted-return baseline. Можно включить через модификацию `compute_advantages`.
- **First-order approximation:** включается флагом `maml.first_order=true`. Сильно ускоряет обучение ценой небольшой потери качества.

## Дальше

После того как MAML заработает на ML1, переходим к:
1. ML10 (10 разных задач Meta-World).
2. Имплементации RL?.
3. Имплементации PEARL.
4. Сравнительный анализ.
