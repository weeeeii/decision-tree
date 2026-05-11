// main.c — обучение и предсказание команд робота
//
// Использование:
//   ./robot_dt demo
//   ./robot_dt train  <dataset.csv> [n_trees] [max_depth]
//   ./robot_dt predict <dataset.csv> <f1> <f2> ... <fN>

#include "decision_tree.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//  Встроенная демонстрация 
// Признаки: [расстояние до препятствия, скорость, угол, заряд]
// Классы:   forward / turn_left / turn_right / stop
static void run_demo(void)
{
    puts("=== Robot Navigation Decision Tree Demo ===\n");

    const int N = 20, F = 4, C = 4;
    Dataset *ds = dataset_alloc((size_t)N, (size_t)F, C);
    if (!ds) { fputs("Out of memory\n", stderr); return; }

    strncpy(ds->class_names[0], "forward",    DT_LABEL_LEN - 1);
    strncpy(ds->class_names[1], "turn_left",  DT_LABEL_LEN - 1);
    strncpy(ds->class_names[2], "turn_right", DT_LABEL_LEN - 1);
    strncpy(ds->class_names[3], "stop",       DT_LABEL_LEN - 1);

    double raw[20][4] = {
        {5.0, 1.0,  0.0, 0.9},
        {4.5, 0.8,  5.0, 0.8},
        {4.0, 0.7,  2.0, 0.7},
        {3.0, 0.5, -30.0, 0.6},
        {2.5, 0.4, -25.0, 0.5},
        {2.0, 0.3, -20.0, 0.4},
        {3.0, 0.5,  30.0, 0.6},
        {2.5, 0.4,  25.0, 0.5},
        {2.0, 0.3,  20.0, 0.4},
        {0.5, 0.1,   0.0, 0.3},
        {0.3, 0.0,   0.0, 0.2},
        {0.4, 0.05,  5.0, 0.1},
        {4.8, 0.9,   1.0, 0.95},
        {4.6, 0.85,  3.0, 0.85},
        {3.5, 0.6, -35.0, 0.55},
        {3.2, 0.55, 35.0, 0.65},
        {0.6, 0.05,  0.0, 0.25},
        {1.5, 0.2, -15.0, 0.4},
        {1.5, 0.2,  15.0, 0.4},
        {5.5, 1.0,   0.0, 1.0},
    };
    int labels[20] = {0,0,0, 1,1,1, 2,2,2, 3,3,3, 0,0,1,2,3,1,2,0};

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < F; j++) ds->X[i][j] = raw[i][j];
        ds->y[i] = labels[i];
    }

    int all_idx[20];
    for (int i = 0; i < N; i++) all_idx[i] = i;

    // одиночное дерево
    puts("--- Single Decision Tree ---");
    DecisionTree *dt = dt_create(5, 2, 2, 42u);
    dt_fit(dt, ds, all_idx, (size_t)N);

    double test1[4] = {4.5, 0.8,  2.0, 0.8};
    double test2[4] = {0.4, 0.0,  0.0, 0.2};
    double test3[4] = {2.8, 0.5, -28.0, 0.6};

    int p1 = dt_predict(dt, test1);
    int p2 = dt_predict(dt, test2);
    int p3 = dt_predict(dt, test3);

    printf("  [dist=4.5, spd=0.8, ang= 2.0, bat=0.8] -> %s\n", p1 >= 0 ? ds->class_names[p1] : "?");
    printf("  [dist=0.4, spd=0.0, ang= 0.0, bat=0.2] -> %s\n", p2 >= 0 ? ds->class_names[p2] : "?");
    printf("  [dist=2.8, spd=0.5, ang=-28.0, bat=0.6] -> %s\n", p3 >= 0 ? ds->class_names[p3] : "?");
    dt_free(dt);

    // бэггинг
    puts("\n--- Bagging Classifier (10 trees) ---");
    BaggingClassifier *bc = bag_create(10, 5, 2, 7u);
    bag_fit(bc, ds);

    printf("  Train accuracy: %.1f%%\n", bag_score(bc, ds) * 100.0);

    int b1 = bag_predict(bc, test1);
    int b2 = bag_predict(bc, test2);
    int b3 = bag_predict(bc, test3);

    printf("  [dist=4.5, spd=0.8, ang= 2.0, bat=0.8] -> %s\n", b1 >= 0 ? ds->class_names[b1] : "?");
    printf("  [dist=0.4, spd=0.0, ang= 0.0, bat=0.2] -> %s\n", b2 >= 0 ? ds->class_names[b2] : "?");
    printf("  [dist=2.8, spd=0.5, ang=-28.0, bat=0.6] -> %s\n", b3 >= 0 ? ds->class_names[b3] : "?");

    bag_free(bc);
    dataset_free(ds);
    puts("\nDemo complete.");
}

//  Команда train 
static void cmd_train(int argc, char *argv[])
{
    if (argc < 3) {
        fprintf(stderr, "Usage: %s train <dataset.csv> [n_trees] [max_depth]\n", argv[0]);
        return;
    }
    const char *path    = argv[2];
    int         n_trees = argc > 3 ? atoi(argv[3]) : 10;
    int         depth   = argc > 4 ? atoi(argv[4]) : 5;

    if (n_trees < 1 || depth < 1) {
        fputs("n_trees и max_depth должны быть >= 1\n", stderr);
        return;
    }

    printf("Загрузка датасета: %s\n", path);
    Dataset *ds = dataset_load_csv(path, 1);
    if (!ds) { fputs("Ошибка загрузки\n", stderr); return; }

    printf("Образцов: %zu  Признаков: %zu  Классов: %d\n",
           ds->n_samples, ds->n_features, ds->n_classes);
    printf("Обучение (%d деревьев, глубина %d)...\n", n_trees, depth);

    BaggingClassifier *bc = bag_create((size_t)n_trees, (size_t)depth, 2, 42u);
    if (!bc) { fputs("Out of memory\n", stderr); dataset_free(ds); return; }

    bag_fit(bc, ds);
    printf("Точность на обучающей выборке: %.2f%%\n", bag_score(bc, ds) * 100.0);

    bag_free(bc);
    dataset_free(ds);
}

//  Команда predict 
static void cmd_predict(int argc, char *argv[])
{
    if (argc < 4) {
        fprintf(stderr, "Usage: %s predict <dataset.csv> <f1> [f2 ...]\n", argv[0]);
        return;
    }

    Dataset *ds = dataset_load_csv(argv[2], 1);
    if (!ds) { fputs("Ошибка загрузки\n", stderr); return; }

    size_t n_feat = ds->n_features;
    if ((size_t)(argc - 3) < n_feat) {
        fprintf(stderr, "Нужно %zu значений признаков, получено %d\n", n_feat, argc - 3);
        dataset_free(ds); return;
    }

    double *x = (double *)malloc(n_feat * sizeof(double));
    if (!x) { fputs("Out of memory\n", stderr); dataset_free(ds); return; }
    for (size_t i = 0; i < n_feat; i++)
        x[i] = atof(argv[3 + (int)i]);

    BaggingClassifier *bc = bag_create(10, 5, 2, 42u);
    if (!bc) { free(x); dataset_free(ds); return; }
    bag_fit(bc, ds);

    int pred = bag_predict(bc, x);
    printf("Предсказание: %s\n",
           (pred >= 0 && pred < ds->n_classes) ? ds->class_names[pred] : "?");

    bag_free(bc);
    free(x);
    dataset_free(ds);
}

//  Точка входа 
static void print_usage(const char *prog)
{
    printf("Использование:\n"
           "  %s demo\n"
           "  %s train  <dataset.csv> [n_trees] [max_depth]\n"
           "  %s predict <dataset.csv> <f1> [f2 ...]\n",
           prog, prog, prog);
}

int main(int argc, char *argv[])
{
    if (argc < 2) { print_usage(argv[0]); return 1; }

    if      (strcmp(argv[1], "demo")    == 0) run_demo();
    else if (strcmp(argv[1], "train")   == 0) cmd_train(argc, argv);
    else if (strcmp(argv[1], "predict") == 0) cmd_predict(argc, argv);
    else {
        fprintf(stderr, "Неизвестная команда: %s\n", argv[1]);
        print_usage(argv[0]);
        return 1;
    }
    return 0;
}
