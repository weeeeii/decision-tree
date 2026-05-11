# Decision Tree with Bagging — реализация на C

Дерево решений с бэггингом и случайной подвыборкой признаков.  
Задача: классификация команд управления роботом по сенсорным данным.

## Структура проекта

```
decision_tree/
├── src/
│   ├── decision_tree.h   # публичный API
│   ├── decision_tree.c   # алгоритмы
│   └── main.c            # консольная программа
├── tests/
│   ├── CMakeLists.txt
│   └── test_all.c        # юнит-тесты
├── data/
│   └── robot.csv         # пример датасета
├── CMakeLists.txt
└── README.md
```

## Алгоритмы

**Дерево решений (CART)**  
Рекурсивное построение по критерию минимума индекса Джини.  
Предсказание — DFS от корня до листа.

**Случайная подвыборка признаков**  
На каждом узле перебирается случайное подмножество из √n_features признаков.

**Бэггинг**  
Каждое дерево обучается на bootstrap-выборке (с возвращением).  
Итоговое предсказание — мажоритарное голосование всех деревьев.

## Сборка

```bash
cmake -S . -B build
cmake --build build
```

## Запуск

```bash
# встроенная демонстрация
./build/robot_dt demo

# обучение на CSV
./build/robot_dt train data/robot.csv 10 5

# предсказание по признакам
./build/robot_dt predict data/robot.csv 4.5 0.8 3.0 0.85
```

## Тесты

```bash
ctest --test-dir build --output-on-failure -V
```

## Формат CSV

Последний столбец — метка класса, первая строка — заголовок.

```
obstacle_dist,speed,turn_angle,battery,command
5.0,1.0,0.0,0.9,forward
0.3,0.0,0.0,0.2,stop
```
