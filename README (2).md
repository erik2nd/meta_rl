# Meta-RL Thesis

**Исследование методов мета-обучения с подкреплением в условиях адаптации к новым задачам**

Самигуллин Э.Э., КФУ, магистратура 09-535  
Научный руководитель: канд. физ.-мат. наук, доцент КСАИТ Андрианова А.А.

---

## О проекте

Цель — сравнительное исследование трёх ключевых парадигм Meta-RL и разработка гибридного метода с повышенной устойчивостью адаптации.

**Проблема:** классический RL неэффективен по данным — миллионы шагов на одну задачу, нулевой перенос знаний. Meta-RL решает это, обучая агента не решать задачу, а *учиться решать* задачи из заданного распределения.

**Три реализованных метода:**

| Метод | Суть | Файл запуска |
|---|---|---|
| **MAML** | Ищет универсальную инициализацию политики, от которой 1-2 шага градиента выводят на новую задачу | `scripts/train_maml.py` |
| **RL²** | Рекуррентная политика (GRU), которая адаптируется неявно через скрытое состояние. Адаптация = накопление памяти | `scripts/train_rl2.py` |
| **PEARL** | Явное кодирование задачи: encoder строит вероятностный вектор контекста $z$ из собранных переходов, политика обусловлена на $z$ | `scripts/train_pearl.py` |

**Гипотеза:** контекстные методы (PEARL) наиболее эффективны, но хрупки на этапе начальной адаптации. Рекуррентные методы (RL²) устойчивее накапливают информацию. Их синтез — перспективное направление.

---

## Структура проекта

```
meta-rl-thesis/
│
├── configs/                        # гиперпараметры в YAML
│   ├── maml_config.yaml
│   ├── rl2_config.yaml
│   └── pearl_config.yaml
│
├── algorithms/
│   ├── maml/
│   │   ├── policy.py               # Gaussian MLP policy
│   │   ├── maml.py                 # inner/outer loop через learn2learn
│   │   └── trainer.py
│   ├── rl2/
│   │   ├── policy.py               # GRU policy: вход = (s, a_prev, r_prev, done_prev)
│   │   ├── ppo.py                  # PPO loss + GAE
│   │   ├── rl2.py                  # сбор мета-трилов (K эпизодов без сброса hidden state)
│   │   └── trainer.py
│   └── pearl/
│       ├── encoder.py              # q_phi(z|c): product-of-experts Gaussian posterior
│       ├── networks.py             # TanhGaussianPolicy + QNetwork, обусловленные на z
│       ├── buffer.py               # MultiTaskBuffer: отдельный буфер на каждую задачу
│       ├── pearl.py                # SAC + KL-регуляризация encoder
│       ├── rollout.py              # rollouts с z из prior / posterior
│       └── trainer.py             # warm-up → collect (prior+posterior) → update
│
├── envs/
│   └── env_utils.py                # MetaWorldTaskSampler: обёртка над ML1/ML10/ML45
│
├── utils/
│   ├── logger.py                   # console + TensorBoard + wandb
│   ├── metrics.py                  # adaptation curves, aggregate stats
│   └── replay.py                   # сбор траекторий (для MAML)
│
├── scripts/
│   ├── train_maml.py               # запуск MAML
│   ├── train_rl2.py                # запуск RL²
│   ├── train_pearl.py              # запуск PEARL
│   ├── sanity_check_rl2_numpy.py   # проверка RL² без torch (~30 сек)
│   ├── plot_results.py             # графики одного метода из history.json
│   └── plot_sanity_rl2.py          # графики sanity-проверки
│
├── logs/                           # TensorBoard / wandb (создаётся при запуске)
└── results/
    └── sanity_rl2/                 # результаты sanity-проверки RL²
        ├── REPORT.md               # подробный анализ результатов
        ├── history.json
        ├── fig1_learning_curves.png
        ├── fig2_adaptation_curves.png
        └── fig3_log_std.png
```

---

## Установка

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Если Meta-World не ставится через pip:
```bash
pip install git+https://github.com/Farama-Foundation/Metaworld.git@master
```

---

## Быстрая проверка (без Meta-World и PyTorch)

Sanity-проверка RL² на синтетической задаче 2D continuous bandit.
Запускается за ~30 секунд, доказывает корректность алгоритма.

```bash
python scripts/sanity_check_rl2_numpy.py   # обучение, выводит прогресс
python scripts/plot_sanity_rl2.py          # строит три графика в results/sanity_rl2/
```

**Что должно получиться:** adaptation_gain в late stage ≈ +1.5  
(агент в 3-м эпизоде трила систематически лучше, чем в 1-м — GRU научился использовать прошлые награды).

---

## Запуск обучения

### MAML
```bash
python scripts/train_maml.py --config configs/maml_config.yaml
```

### RL²
```bash
python scripts/train_rl2.py --config configs/rl2_config.yaml
```

### PEARL
```bash
python scripts/train_pearl.py --config configs/pearl_config.yaml
```

### Переопределение параметров из командной строки
```bash
python scripts/train_rl2.py --config configs/rl2_config.yaml \
    --override experiment.seed=7 training.total_meta_iterations=100
```

### Несколько seed для статистики
```bash
for seed in 1 2 3 4 5; do
    python scripts/train_maml.py --config configs/maml_config.yaml \
        --override experiment.seed=$seed experiment.name=maml_seed_$seed
done
```

---

## Просмотр результатов

### TensorBoard (в реальном времени)
```bash
tensorboard --logdir=./logs
# открыть http://localhost:6006
```

### Графики из history.json (после окончания)
```bash
python scripts/plot_results.py --history results/maml_ml1_reach/history.json --out plots/maml
python scripts/plot_results.py --history results/rl2_ml1_reach/history.json  --out plots/rl2
python scripts/plot_results.py --history results/pearl_ml1_reach/history.json --out plots/pearl
```

Каждый вызов создаёт:
- `training_curves.png` — return по эпизодам и adaptation gain в ходе мета-обучения
- `loss_curves.png` — policy loss, value loss, entropy
- `eval_curves.png` — оценка на тестовых задачах (pre/post адаптации)

---

## Ключевые метрики

| Метрика | Что измеряет | Где смотреть |
|---|---|---|
| `adaptation_gain` | post_return − pre_return на тест-задачах | eval_curves.png |
| `post_adapt_return_std` | разброс результатов (устойчивость) | eval_curves.png |
| `first_ep` / `last_ep` | адаптация внутри трила (только RL²) | training_curves.png |
| `kl` | насколько encoder отклоняется от prior (только PEARL) | loss_curves.png |

---

## Бенчмарк Meta-World

Эксперименты проводятся на [Meta-World](https://meta-world.github.io/):

| Сценарий | Задачи | Использование |
|---|---|---|
| **ML1** | 1 задача (напр. `reach-v2`) с вариациями | отладка, baseline |
| **ML10** | 10 разных задач | основные эксперименты |
| **ML45** | 45 задач | тест масштабируемости |

Менять бенчмарк:
```yaml
# в конфиге:
env:
  benchmark: "ML10"
  task_name: "reach-v2"   # игнорируется для ML10/ML45
```

---

## Технический стек

- **Python 3.10+**
- **PyTorch** — все три алгоритма
- **learn2learn** — дифференцируемый inner loop для MAML
- **Meta-World** — среда для экспериментов
- **TensorBoard / wandb** — логирование
- **NumPy, Matplotlib, Seaborn** — анализ и визуализация

---

## Статус и план

- [x] MAML — реализован
- [x] RL² — реализован, sanity-проверка пройдена
- [x] PEARL — реализован
- [ ] Запуск baseline-экспериментов на Meta-World ML10
- [ ] Сравнительный анализ (скрипт `compare_methods.py`)
- [ ] Гибридный метод: PEARL + RNN-encoder
- [ ] Финальные эксперименты и статистический анализ
