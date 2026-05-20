// юнит-тесты, покрытие ~100% 

#include "decision_tree.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Минималистичный тест-фреймворк

static int g_pass = 0;
static int g_fail = 0;

#define ASSERT_TRUE(expr) do { \
    if (expr) { g_pass++; } \
    else { g_fail++; \
           fprintf(stderr, "FAIL  %s:%d  " #expr "\n", __FILE__, __LINE__); } \
} while(0)

#define ASSERT_EQ(a, b)        ASSERT_TRUE((a) == (b))
#define ASSERT_NEQ(a, b)       ASSERT_TRUE((a) != (b))
#define ASSERT_NEAR(a, b, eps) ASSERT_TRUE(fabs((double)(a)-(double)(b)) < (eps))
#define ASSERT_NULL(p)         ASSERT_TRUE((p) == NULL)
#define ASSERT_NOT_NULL(p)     ASSERT_TRUE((p) != NULL)

#define TEST(name) static void name(void)
#define RUN(name)  do { printf("[ RUN ] " #name "\n"); name(); } while(0)

// Вспомогательный датасет: x < 2.0 -> класс 0, иначе -> класс 1 
static Dataset *make_simple_ds(void)
{
    Dataset *ds = dataset_alloc(6, 1, 2);
    if (!ds) return NULL;
    double xs[] = {0.0, 0.5, 1.0, 2.0, 3.0, 4.0};
    int    ys[] = {  0,   0,   0,   1,   1,   1};
    for (int i = 0; i < 6; i++) { ds->X[i][0] = xs[i]; ds->y[i] = ys[i]; }
    strncpy(ds->class_names[0], "zero", DT_LABEL_LEN - 1);
    strncpy(ds->class_names[1], "one",  DT_LABEL_LEN - 1);
    return ds;
}

static Dataset *make_regression_ds(void)
{
    Dataset *ds = dataset_alloc(8, 1, 1);
    if (!ds) return NULL;
    ds->is_regression = 1;
    double xs[] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    double ys[] = {1.0, 2.0, 3.0, 4.0, 8.0, 10.0, 12.0, 14.0};
    for (int i = 0; i < 8; i++) {
        ds->X[i][0] = xs[i];
        ds->y_reg[i] = ys[i];
    }
    return ds;
}

// Dataset

TEST(test_dataset_alloc_free)
{
    Dataset *ds = dataset_alloc(10, 3, 2);
    ASSERT_NOT_NULL(ds);
    ASSERT_EQ(ds->n_samples, 10u);
    ASSERT_EQ(ds->n_features, 3u);
    ASSERT_EQ(ds->n_classes, 2);
    ds->X[0][0] = 1.5; ds->X[9][2] = -0.7; ds->y[5] = 1;
    ASSERT_NEAR(ds->X[0][0],  1.5, 1e-9);
    ASSERT_NEAR(ds->X[9][2], -0.7, 1e-9);
    ASSERT_EQ(ds->y[5], 1);
    dataset_free(ds);
    dataset_free(NULL); // не должно упасть
    g_pass++;
}

TEST(test_dataset_load_csv)
{
    const char *path = "/tmp/dt_test.csv";
    FILE *f = fopen(path, "w");
    ASSERT_NOT_NULL(f);
    fprintf(f, "dist,speed,label\n5.0,1.0,forward\n0.3,0.0,stop\n"
               "2.5,0.5,turn_left\n2.5,0.5,turn_right\n5.1,0.9,forward\n");
    fclose(f);

    Dataset *ds = dataset_load_csv(path, 1);
    ASSERT_NOT_NULL(ds);
    ASSERT_EQ(ds->n_samples, 5u);
    ASSERT_EQ(ds->n_features, 2u);
    ASSERT_EQ(ds->n_classes, 4);
    ASSERT_NEAR(ds->X[0][0], 5.0, 1e-9);
    ASSERT_EQ(ds->y[0], 0); // forward -> индекс 0
    ASSERT_EQ(ds->y[1], 1); // stop -> индекс 1
    dataset_free(ds);
}

TEST(test_dataset_load_csv_no_header)
{
    const char *path = "/tmp/dt_no_header.csv";
    FILE *f = fopen(path, "w");
    ASSERT_NOT_NULL(f);
    fprintf(f, "1.0,0.0,A\n2.0,1.0,B\n3.0,0.5,A\n");
    fclose(f);

    Dataset *ds = dataset_load_csv(path, 0);
    ASSERT_NOT_NULL(ds);
    ASSERT_EQ(ds->n_samples, 3u);
    ASSERT_EQ(ds->n_classes, 2);
    dataset_free(ds);
}

TEST(test_dataset_load_csv_regression)
{
    const char *path = "/tmp/dt_reg.csv";
    FILE *f = fopen(path, "w");
    ASSERT_NOT_NULL(f);
    fprintf(f, "x,target\n1.0,2.5\n2.0,5.0\n3.0,7.5\n");
    fclose(f);

    Dataset *ds = dataset_load_csv_regression(path, 1);
    ASSERT_NOT_NULL(ds);
    ASSERT_EQ(ds->is_regression, 1);
    ASSERT_EQ(ds->n_samples, 3u);
    ASSERT_EQ(ds->n_features, 1u);
    ASSERT_NEAR(ds->y_reg[2], 7.5, 1e-9);
    dataset_free(ds);
}

TEST(test_dataset_load_csv_bad_path)
{
    Dataset *ds = dataset_load_csv("/nonexistent/path.csv", 1);
    ASSERT_NULL(ds);
}

// Метрики 

TEST(test_gini_pure)
{
    int c[2] = {10, 0};
    ASSERT_NEAR(dt_gini(c, 2, 10), 0.0, 1e-9);
}

TEST(test_gini_balanced)
{
    int c[2] = {5, 5};
    ASSERT_NEAR(dt_gini(c, 2, 10), 0.5, 1e-9);
}

TEST(test_gini_uniform_3)
{
    int c[3] = {3, 3, 3};
    ASSERT_NEAR(dt_gini(c, 3, 9), 2.0/3.0, 1e-9);
}

TEST(test_gini_zero_total)
{
    int c[2] = {0, 0};
    ASSERT_NEAR(dt_gini(c, 2, 0), 0.0, 1e-9);
}

TEST(test_gini_single_class_many)
{
    int c[4] = {0, 10, 0, 0};
    ASSERT_NEAR(dt_gini(c, 4, 10), 0.0, 1e-9);
}

TEST(test_entropy_pure)
{
    int c[2] = {10, 0};
    ASSERT_NEAR(dt_entropy(c, 2, 10), 0.0, 1e-9);
}

TEST(test_entropy_balanced)
{
    int c[2] = {5, 5};
    ASSERT_NEAR(dt_entropy(c, 2, 10), 1.0, 1e-9);
}

TEST(test_entropy_zero_total)
{
    int c[2] = {0, 0};
    ASSERT_NEAR(dt_entropy(c, 2, 0), 0.0, 1e-9);
}

TEST(test_entropy_uniform_4)
{
    // H(1/4,1/4,1/4,1/4) = 2 бита
    int c[4] = {2, 2, 2, 2};
    ASSERT_NEAR(dt_entropy(c, 4, 8), 2.0, 1e-9);
}

// DecisionTree 

TEST(test_dt_create_free)
{
    DecisionTree *dt = dt_create(5, 2, 3, 42u);
    ASSERT_NOT_NULL(dt);
    ASSERT_EQ(dt->max_depth, 5u);
    ASSERT_EQ(dt->min_samples_split, 2);
    ASSERT_EQ(dt->n_features_split, 3);
    ASSERT_NULL(dt->root);
    dt_free(dt);
    dt_free(NULL); // не должно упасть
    g_pass++;
}

TEST(test_dt_create_defaults)
{
    // max_depth=0 -> 10, min_samples_split=0 -> 2
    DecisionTree *dt = dt_create(0, 0, 1, 1u);
    ASSERT_NOT_NULL(dt);
    ASSERT_EQ(dt->max_depth, 10u);
    ASSERT_EQ(dt->min_samples_split, 2);
    dt_free(dt);
}

TEST(test_dt_fit_predict_trivial)
{
    Dataset *ds = make_simple_ds();
    ASSERT_NOT_NULL(ds);
    int all[] = {0, 1, 2, 3, 4, 5};
    DecisionTree *dt = dt_create(5, 2, 1, 1u);
    dt_fit(dt, ds, all, 6);
    ASSERT_NOT_NULL(dt->root);
    double x0[1] = {0.5}, x1[1] = {3.5};
    ASSERT_EQ(dt_predict(dt, x0), 0);
    ASSERT_EQ(dt_predict(dt, x1), 1);
    dt_free(dt);
    dataset_free(ds);
}

TEST(test_dt_fit_predict_entropy)
{
    Dataset *ds = make_simple_ds();
    ASSERT_NOT_NULL(ds);
    int all[] = {0, 1, 2, 3, 4, 5};
    DecisionTree *dt = dt_create_ex(5, 2, 1, 1u,
                                    DT_TASK_CLASSIFICATION,
                                    DT_CRITERION_ENTROPY);
    dt_fit(dt, ds, all, 6);
    double x0[1] = {0.5}, x1[1] = {3.5};
    ASSERT_EQ(dt_predict(dt, x0), 0);
    ASSERT_EQ(dt_predict(dt, x1), 1);
    dt_free(dt);
    dataset_free(ds);
}

TEST(test_dt_fit_predict_regression)
{
    Dataset *ds = make_regression_ds();
    ASSERT_NOT_NULL(ds);
    int all[8]; for (int i = 0; i < 8; i++) all[i] = i;
    DecisionTree *dt = dt_create_ex(4, 2, 1, 3u,
                                    DT_TASK_REGRESSION,
                                    DT_CRITERION_MSE);
    dt_fit(dt, ds, all, 8);
    ASSERT_TRUE(dt_predict_value(dt, ds->X[0]) < 4.0);
    ASSERT_TRUE(dt_predict_value(dt, ds->X[7]) > 8.0);
    dt_free(dt);
    dataset_free(ds);
}

TEST(test_dt_predict_no_root)
{
    // дерево без корня -> -1
    DecisionTree *dt = dt_create(5, 2, 1, 0u);
    double x[1] = {1.0};
    ASSERT_EQ(dt_predict(dt, x), -1);
    dt_free(dt);
}

TEST(test_dt_fit_all_same_class)
{
    // все образцы одного класса -> корень является листом
    Dataset *ds = dataset_alloc(4, 2, 2);
    ASSERT_NOT_NULL(ds);
    for (int i = 0; i < 4; i++) { ds->X[i][0] = (double)i; ds->y[i] = 0; }
    int idx[] = {0, 1, 2, 3};
    DecisionTree *dt = dt_create(5, 2, 1, 0u);
    dt_fit(dt, ds, idx, 4);
    ASSERT_NOT_NULL(dt->root);
    ASSERT_EQ(dt->root->is_leaf, 1);
    ASSERT_EQ(dt->root->class_label, 0);
    dt_free(dt);
    dataset_free(ds);
}

TEST(test_dt_fit_depth_one)
{
    Dataset *ds = make_simple_ds();
    int all[] = {0, 1, 2, 3, 4, 5};
    DecisionTree *dt = dt_create(1, 2, 1, 42u);
    dt_fit(dt, ds, all, 6);
    ASSERT_NOT_NULL(dt->root);
    ASSERT_EQ(dt->root->is_leaf, 0);
    ASSERT_EQ(dt->root->left->is_leaf,  1);
    ASSERT_EQ(dt->root->right->is_leaf, 1);
    dt_free(dt);
    dataset_free(ds);
}

TEST(test_dt_fit_min_samples)
{
    // min_samples_split > n -> корень является листом
    Dataset *ds = make_simple_ds();
    int all[] = {0, 1, 2, 3, 4, 5};
    DecisionTree *dt = dt_create(10, 100, 1, 0u);
    dt_fit(dt, ds, all, 6);
    ASSERT_NOT_NULL(dt->root);
    ASSERT_EQ(dt->root->is_leaf, 1);
    dt_free(dt);
    dataset_free(ds);
}

TEST(test_dt_fit_multiclass)
{
    // 3 класса, линейно разделимые
    Dataset *ds = dataset_alloc(9, 1, 3);
    ASSERT_NOT_NULL(ds);
    for (int i = 0; i < 9; i++) { ds->X[i][0] = (double)i; ds->y[i] = i / 3; }
    int all[9]; for (int i = 0; i < 9; i++) all[i] = i;
    DecisionTree *dt = dt_create(5, 2, 1, 7u);
    dt_fit(dt, ds, all, 9);
    double x0[1] = {0.5}, x1[1] = {4.0}, x2[1] = {7.5};
    ASSERT_EQ(dt_predict(dt, x0), 0);
    ASSERT_EQ(dt_predict(dt, x1), 1);
    ASSERT_EQ(dt_predict(dt, x2), 2);
    dt_free(dt);
    dataset_free(ds);
}

TEST(test_dt_fit_multi_feature)
{
    // признак 1 разделяет, признак 0 — шум
    Dataset *ds = dataset_alloc(6, 2, 2);
    ASSERT_NOT_NULL(ds);
    double xs[][2] = {{0.9,0.1},{0.1,0.2},{0.8,0.3},{0.2,3.0},{0.7,4.0},{0.3,5.0}};
    int    ys[]    = {0, 0, 0, 1, 1, 1};
    for (int i = 0; i < 6; i++) { ds->X[i][0]=xs[i][0]; ds->X[i][1]=xs[i][1]; ds->y[i]=ys[i]; }
    int all[6]; for (int i = 0; i < 6; i++) all[i] = i;
    DecisionTree *dt = dt_create(5, 2, 2, 3u);
    dt_fit(dt, ds, all, 6);
    double a[2] = {0.5, 0.2}, b[2] = {0.5, 4.5};
    ASSERT_EQ(dt_predict(dt, a), 0);
    ASSERT_EQ(dt_predict(dt, b), 1);
    dt_free(dt);
    dataset_free(ds);
}

// BaggingClassifier 

TEST(test_bag_create_free)
{
    BaggingClassifier *bc = bag_create(5, 3, 2, 0u);
    ASSERT_NOT_NULL(bc);
    ASSERT_EQ(bc->n_trees, 5u);
    ASSERT_EQ(bc->max_depth, 3u);
    bag_free(bc);
    bag_free(NULL); // не должно упасть
    g_pass++;
}

TEST(test_bag_fit_predict_trivial)
{
    Dataset *ds = make_simple_ds();
    BaggingClassifier *bc = bag_create(5, 5, 2, 42u);
    bag_fit(bc, ds);
    ASSERT_EQ(bc->n_classes, 2);
    double x0[1] = {0.5}, x1[1] = {3.5};
    ASSERT_EQ(bag_predict(bc, x0), 0);
    ASSERT_EQ(bag_predict(bc, x1), 1);
    bag_free(bc);
    dataset_free(ds);
}

TEST(test_bag_random_vote_subset)
{
    Dataset *ds = make_simple_ds();
    BaggingClassifier *bc = bag_create_ex(9, 5, 2, 42u,
                                          DT_TASK_CLASSIFICATION,
                                          DT_CRITERION_ENTROPY, 3);
    bag_fit(bc, ds);
    ASSERT_EQ(bc->n_vote_trees, 3u);
    double x0[1] = {0.5}, x1[1] = {3.5};
    ASSERT_EQ(bag_predict(bc, x0), 0);
    ASSERT_EQ(bag_predict(bc, x1), 1);
    bag_free(bc);
    dataset_free(ds);
}

TEST(test_bag_regression_rmse)
{
    Dataset *ds = make_regression_ds();
    BaggingClassifier *bc = bag_create_ex(20, 5, 2, 42u,
                                          DT_TASK_REGRESSION,
                                          DT_CRITERION_MSE, 10);
    bag_fit(bc, ds);
    ASSERT_TRUE(bag_rmse(bc, ds) < 4.0);
    double pred = bag_predict_value(bc, ds->X[7]);
    ASSERT_TRUE(pred > 7.0);
    bag_free(bc);
    dataset_free(ds);
}

TEST(test_bag_score_perfect)
{
    Dataset *ds = make_simple_ds();
    BaggingClassifier *bc = bag_create(10, 8, 2, 99u);
    bag_fit(bc, ds);
    ASSERT_NEAR(bag_score(bc, ds), 1.0, 0.01);
    bag_free(bc);
    dataset_free(ds);
}

TEST(test_bag_score_empty_dataset)
{
    Dataset *ds = dataset_alloc(1, 1, 2);
    ASSERT_NOT_NULL(ds);
    ds->X[0][0] = 1.0; ds->y[0] = 0;

    BaggingClassifier *bc = bag_create(3, 3, 2, 0u);
    bag_fit(bc, ds);

    // пустой датасет — accuracy = 0
    Dataset empty;
    memset(&empty, 0, sizeof(empty));
    empty.n_samples = 0; empty.n_features = 1; empty.n_classes = 2;
    ASSERT_NEAR(bag_score(bc, &empty), 0.0, 1e-9);

    bag_free(bc);
    dataset_free(ds);
}

TEST(test_bag_single_tree)
{
    Dataset *ds = make_simple_ds();
    BaggingClassifier *bc = bag_create(1, 5, 2, 13u);
    bag_fit(bc, ds);
    ASSERT_TRUE(bag_score(bc, ds) >= 0.5);
    bag_free(bc);
    dataset_free(ds);
}

TEST(test_bag_many_trees)
{
    Dataset *ds = make_simple_ds();
    BaggingClassifier *bc = bag_create(50, 4, 2, 77u);
    bag_fit(bc, ds);
    ASSERT_TRUE(bag_score(bc, ds) >= 0.5);
    bag_free(bc);
    dataset_free(ds);
}

TEST(test_bag_predict_multiclass)
{
    Dataset *ds = dataset_alloc(9, 1, 3);
    ASSERT_NOT_NULL(ds);
    for (int i = 0; i < 9; i++) { ds->X[i][0] = (double)i; ds->y[i] = i / 3; }
    BaggingClassifier *bc = bag_create(7, 5, 2, 5u);
    bag_fit(bc, ds);
    ASSERT_TRUE(bag_score(bc, ds) >= 0.6);
    bag_free(bc);
    dataset_free(ds);
}

// Интеграционный тест 

TEST(test_csv_robot_scenario)
{
    const char *path = "/tmp/robot.csv";
    FILE *f = fopen(path, "w");
    ASSERT_NOT_NULL(f);
    fprintf(f, "dist,speed,angle,battery,command\n"
               "5.0,1.0,0.0,0.9,forward\n4.5,0.8,5.0,0.8,forward\n"
               "4.0,0.7,2.0,0.7,forward\n0.5,0.1,0.0,0.3,stop\n"
               "0.3,0.0,0.0,0.2,stop\n3.0,0.5,-30.0,0.6,turn_left\n"
               "2.5,0.4,-25.0,0.5,turn_left\n3.0,0.5,30.0,0.6,turn_right\n"
               "2.5,0.4,25.0,0.5,turn_right\n");
    fclose(f);

    Dataset *ds = dataset_load_csv(path, 1);
    ASSERT_NOT_NULL(ds);
    ASSERT_EQ(ds->n_samples, 9u);
    ASSERT_EQ(ds->n_features, 4u);
    ASSERT_EQ(ds->n_classes, 4);

    BaggingClassifier *bc = bag_create(10, 6, 2, 42u);
    bag_fit(bc, ds);
    ASSERT_TRUE(bag_score(bc, ds) >= 0.8);

    bag_free(bc);
    dataset_free(ds);
}

// Граничные случаи 

TEST(test_single_sample)
{
    // 1 образец < min_samples_split -> лист
    Dataset *ds = dataset_alloc(1, 2, 2);
    ASSERT_NOT_NULL(ds);
    ds->X[0][0] = 1.0; ds->X[0][1] = 2.0; ds->y[0] = 1;
    int idx[1] = {0};
    DecisionTree *dt = dt_create(5, 2, 1, 0u);
    dt_fit(dt, ds, idx, 1);
    ASSERT_EQ(dt->root->is_leaf, 1);
    double x[2] = {1.0, 2.0};
    ASSERT_EQ(dt_predict(dt, x), 1);
    dt_free(dt);
    dataset_free(ds);
}

TEST(test_identical_features_different_labels)
{
    // все признаки одинаковы -> разбиение невозможно -> лист
    Dataset *ds = dataset_alloc(4, 1, 2);
    ASSERT_NOT_NULL(ds);
    for (int i = 0; i < 4; i++) { ds->X[i][0] = 1.0; ds->y[i] = i % 2; }
    int idx[4] = {0, 1, 2, 3};
    DecisionTree *dt = dt_create(5, 2, 1, 0u);
    dt_fit(dt, ds, idx, 4);
    ASSERT_EQ(dt->root->is_leaf, 1);
    dt_free(dt);
    dataset_free(ds);
}

// ── Запуск всех тестов

int main(void)
{
    puts("=== Running unit tests ===\n");

    RUN(test_dataset_alloc_free);
    RUN(test_dataset_load_csv);
    RUN(test_dataset_load_csv_no_header);
    RUN(test_dataset_load_csv_regression);
    RUN(test_dataset_load_csv_bad_path);

    RUN(test_gini_pure);
    RUN(test_gini_balanced);
    RUN(test_gini_uniform_3);
    RUN(test_gini_zero_total);
    RUN(test_gini_single_class_many);
    RUN(test_entropy_pure);
    RUN(test_entropy_balanced);
    RUN(test_entropy_zero_total);
    RUN(test_entropy_uniform_4);

    RUN(test_dt_create_free);
    RUN(test_dt_create_defaults);
    RUN(test_dt_fit_predict_trivial);
    RUN(test_dt_fit_predict_entropy);
    RUN(test_dt_fit_predict_regression);
    RUN(test_dt_predict_no_root);
    RUN(test_dt_fit_all_same_class);
    RUN(test_dt_fit_depth_one);
    RUN(test_dt_fit_min_samples);
    RUN(test_dt_fit_multiclass);
    RUN(test_dt_fit_multi_feature);

    RUN(test_bag_create_free);
    RUN(test_bag_fit_predict_trivial);
    RUN(test_bag_random_vote_subset);
    RUN(test_bag_regression_rmse);
    RUN(test_bag_score_perfect);
    RUN(test_bag_score_empty_dataset);
    RUN(test_bag_single_tree);
    RUN(test_bag_many_trees);
    RUN(test_bag_predict_multiclass);

    RUN(test_csv_robot_scenario);

    RUN(test_single_sample);
    RUN(test_identical_features_different_labels);

    printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
