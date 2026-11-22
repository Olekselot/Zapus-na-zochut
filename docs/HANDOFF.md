# Передача роботи

Стан гілки `improvements` на 22.11.2026.

## Що готово

- Усі виправлення з аналізу плану, список «було / стало» — у [CHANGES.md](CHANGES.md).
- 47 тестів проходять: `python -m unittest discover -s tests -t .`. Застосунок стартує: `python app.py`.
- Інструменти даних і метрик лежать у `tools/` і `eval/`, ноутбук для Colab — у `notebooks/`.

**Числових результатів ще немає.** Прогін LFW і навчання антиспуфу не завершились: на ноутбуці без GPU, де було вільно ~2,5 ГБ RAM, процеси зупинились. Цифр для звіту поки немає.

## Дані

Даних у git немає: `data/` у `.gitignore`. Вони займають ~2,4 ГБ, а ліцензія CelebA-Spoof дозволяє лише некомерційні дослідження. Та сама команда з тим самим `--seed` завантажує той самий набір.

| Набір | Що було зібрано локально | Як отримати |
|---|---|---|
| CelebA-Spoof train | 7718 live + 8000 spoof, 5 шардів | `python tools/prepare_celeba_spoof.py --split train --shards 5 --max-live 8000 --max-spoof 8000` |
| CelebA-Spoof test | 419 + 1403, лише 1 шард із 3 | `python tools/prepare_celeba_spoof.py --split test --shards 3 --max-live 1500 --max-spoof 1500` (продовжує з місця зупинки) |
| LFW | — | sklearn завантажує його сам при першому запуску `eval/recognition_metrics.py` |
| Клієнти демо | 6 людей з LFW | `python tools/prepare_lfw_enroll.py`, потім `python tools/enroll_folder.py` |
| Власний набір `data/custom` | не зібраний | `tools/collect.py` за протоколом [DATA_COLLECTION.md](DATA_COLLECTION.md) |

Для даних потрібно `pip install -r requirements-data.txt`. Найпростіше працювати в Colab: `notebooks/train_antispoof_colab.ipynb` сам завантажує дані й вчить модель на GPU.

У дзеркалі CelebA-Spoof немає ні типу атаки, ні ID людини. Тому:

- підробки вчаться як «друк або екран» (мітка `spoof`);
- APCER окремо для друку й екрана можна отримати лише на власному наборі;
- чесний test береться з окремого test-спліту, тому навчання запускається з `--test-ratio 0`.

## Що лишилось, за порядком

1. **Поріг розпізнавання.** `python eval/recognition_metrics.py` рахує LFW на 10 фолдах і на всіх парах та пише поріг демо в `models/recognition_threshold.json`. На CPU це ~20 хв, запускати без інших важких програм.
2. **Навчання антиспуфу.** Через Colab або локально: `python training/train_antispoof.py --data data/celeba_spoof --balance --test-ratio 0 --epochs 8 --patience 3`.
3. **Метрики антиспуфу.** `python eval/antispoof_metrics.py --data data/celeba_spoof_test --split all` дає APCER, BPCER, DET-криву і матрицю помилок.
4. **Порівняння «з нуля проти донавчання».** Те саме навчання з `--arch topologies_mnv2 --run-name scratch`, у таблицю поруч з `mnv3_small`. Викладач оцінює саме навчання, тож це важливо.
5. **Власний набір.** Щонайменше 5 людей за [DATA_COLLECTION.md](DATA_COLLECTION.md), класи live / print / phone / laptop. Потім:
   - перевірити, як модель переноситься на нашу камеру: `eval/antispoof_metrics.py --data data/custom --split all`;
   - донавчити на `--data data/celeba_spoof data/custom` або перевірити крос-валідацією `--folds 5`.
6. **Інші метрики.** `eval/identification_metrics.py` (1:N), `eval/embedding_map.py` (t-SNE), пробні оплати `tools/enroll_folder.py --check --db data/check.sqlite`.
7. **Справжня камера.** `tools/collect.py` і активна перевірка ще не запускались, бо камери не було. Варто перевірити, чи не переплутані поворот ліворуч і праворуч: код розрахований на камеру без дзеркалення.
8. **Порівняння з deepface.** `pip install -r requirements-eval.txt` (тягне TensorFlow), потім `python eval/compare_deepface.py --per-fold 60`.
9. **Не зроблено зовсім:**
   - вимір затримки оплати;
   - «адаптер поверх векторів» з плану (необов'язковий).
