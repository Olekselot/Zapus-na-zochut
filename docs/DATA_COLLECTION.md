# Збір даних

Три набори, кожен для свого:

| Набір | Де лежить | Навіщо | Звідки |
|---|---|---|---|
| CelebA-Spoof (підмножина) | `data/celeba_spoof`, `data/celeba_spoof_test` | навчити захист на тисячах різних людей | `tools/prepare_celeba_spoof.py` |
| Власний набір атак | `data/custom/<клас>/<pNN>/` | перевірити й донавчити на нашій камері й світлі | `tools/collect.py` |
| Клієнти демо | `data/enroll/<людина>/{enroll,pay}` | реєстрація й оплата на захисті | `tools/collect.py --kind enroll/pay` або `tools/prepare_lfw_enroll.py` |

Усе в `data/` ігнорується git. Фото не кладемо ні в репозиторій, ні в чати.

## 1. CelebA-Spoof

```text
python tools/prepare_celeba_spoof.py --split train --shards 5 --max-live 8000 --max-spoof 8000
python tools/prepare_celeba_spoof.py --split test --shards 3 --max-live 1500 --max-spoof 1500
```

Дзеркало на Hugging Face зберігає лише мітку live/spoof, без типу атаки і без коду людини. Тому підробки лягають у `spoof/`, тобто тип невідомий, і навчання вчить для них «друк або екран». Чесний test береться з test-спліту дзеркала: це офіційний intra-test з іншими людьми. Ліцензія дозволяє лише некомерційне дослідження і забороняє поширення.

## 2. Власний набір атак

Ціль: **8–12 людей × 4 типи × 20–25 кадрів**, тобто 640–1200 фото. Різні люди важливіші за кількість кадрів однієї людини.

Перед зйомкою:

- та сама вебкамера, що буде на захисті; відстань 40–70 см;
- для кожної людини код `p01`, `p02`…, однаковий у всіх папках; прізвищ у назвах немає;
- для атак потрібне фото обличчя цієї ж людини: роздрукувати на A4 і мати на телефоні та ноутбуці.

Команди для однієї людини (у вікні: ПРОБІЛ — серія, S — один кадр, Q — вихід):

```text
python tools/collect.py --person p01 --kind live   --condition рівне_світло
python tools/collect.py --person p01 --kind live   --condition бокове_світло
python tools/collect.py --person p01 --kind live   --condition легкий_поворот
python tools/collect.py --person p01 --kind print  --condition a4_плоско
python tools/collect.py --person p01 --kind print  --condition a4_зігнуто
python tools/collect.py --person p01 --kind phone  --condition фото
python tools/collect.py --person p01 --kind phone  --condition відео
python tools/collect.py --person p01 --kind laptop --condition фото
```

Кадр зберігається, лише якщо в кадрі рівно одне достатньо велике і різке обличчя. Кожен кадр записується в `data/custom/manifest.csv`: людина, тип, умова, камера, час.

Чекліст на людину:

- [ ] `live`: рівне світло, бокове світло, легкий поворот, окуляри (якщо носить)
- [ ] `print`: A4 плоско, A4 зігнуто, ближче/далі
- [ ] `phone`: фото, відео, максимальна яскравість, під кутом
- [ ] `laptop`: фото, відео

Перевірка, чого бракує:

```text
python tools/dataset_report.py data/custom
```

Звіт показує таблицю «людина × клас», людей без якогось типу, класи з менш ніж 15 фото, дисбаланс і майже однакові кадри.

## 3. Клієнти демо

Справжні клієнти, лише за згодою: 8–12 фото на реєстрацію і 3–5 інших фото «на оплату».

```text
python tools/collect.py --person p01 --kind enroll --count 10
python tools/collect.py --person p01 --kind pay --count 4
```

Відомі люди з LFW, щоб база не була порожньою:

```text
python tools/prepare_lfw_enroll.py --people 6
python tools/enroll_folder.py --check --db data/check.sqlite   # пробна база: реєстрація + оплати
python tools/enroll_folder.py                                   # справжня база демо
```

## Згода

`tools/collect.py` перед першим знімком людини питає згоду і записує в `data/custom/consent.csv` код людини, дату і «так». Текст, який зачитуємо або даємо прочитати:

> Я погоджуюсь, що мої фото й відео обличчя, а також фото з екрана і роздруківки з моїм обличчям, використають лише для навчального проєкту «Оплата за обличчям» на ФПМІ ЛНУ. Дані зберігаються локально, не публікуються і не передаються третім особам. У базі демо зберігаються лише числові вектори, без фото. Після заліку всі фото й база видаляються. Я можу відкликати згоду будь-коли, і мої фото видалять.

Хто не хоче давати своє обличчя, той не дає. У тексті звіту людей позначаємо лише кодами.

## Після заліку

```text
Remove-Item -Recurse -Force data\custom, data\enroll, data\cache, data\bank.sqlite
```
