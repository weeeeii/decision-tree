// decision_tree.c — дерево решений с бэггингом и случайной подвыборкой признаков

#include "decision_tree.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <assert.h>
#include <ctype.h>

//  ГПСЧ (линейный конгруэнтный генератор) 

static unsigned int lcg_next(unsigned int *s)
{
    *s = *s * 1664525u + 1013904223u;
    return *s;
}

// случайное целое в [lo, hi)
static int lcg_range(unsigned int *s, int lo, int hi)
{
    assert(hi > lo);
    return lo + (int)((lcg_next(s) >> 1) % (unsigned)(hi - lo));
}

//  Dataset 

Dataset *dataset_alloc(size_t n_samples, size_t n_features, int n_classes)
{
    assert(n_samples > 0 && n_features > 0 && n_classes > 0);
    assert(n_features <= DT_MAX_FEATURES && n_classes <= DT_MAX_CLASSES);

    Dataset *ds = (Dataset *)calloc(1, sizeof(Dataset));
    if (!ds) return NULL;

    ds->X = (double **)malloc(n_samples * sizeof(double *));
    if (!ds->X) { free(ds); return NULL; }

    for (size_t i = 0; i < n_samples; i++) {
        ds->X[i] = (double *)calloc(n_features, sizeof(double));
        if (!ds->X[i]) {
            for (size_t j = 0; j < i; j++) free(ds->X[j]);
            free(ds->X); free(ds);
            return NULL;
        }
    }

    ds->y = (int *)calloc(n_samples, sizeof(int));
    if (!ds->y) {
        for (size_t i = 0; i < n_samples; i++) free(ds->X[i]);
        free(ds->X); free(ds);
        return NULL;
    }

    ds->n_samples  = n_samples;
    ds->n_features = n_features;
    ds->n_classes  = n_classes;
    return ds;
}

void dataset_free(Dataset *ds)
{
    if (!ds) return;
    if (ds->X) {
        for (size_t i = 0; i < ds->n_samples; i++) free(ds->X[i]);
        free(ds->X);
    }
    free(ds->y);
    free(ds);
}

// загрузка CSV: последний столбец — метка класса
Dataset *dataset_load_csv(const char *path, int has_header)
{
    FILE *f = fopen(path, "r");
    if (!f) { perror(path); return NULL; }

    // первый проход: считаем строки и столбцы
    char line[4096];
    size_t n_lines = 0;
    int    n_cols  = -1;

    while (fgets(line, sizeof(line), f)) {
        if (n_lines == 0 && has_header) { n_lines++; continue; }
        if (n_cols < 0) {
            n_cols = 1;
            for (char *p = line; *p; p++)
                if (*p == ',') n_cols++;
        }
        n_lines++;
    }
    rewind(f);

    if (n_cols < 2 || n_lines == 0) { fclose(f); return NULL; }
    size_t n_features = (size_t)(n_cols - 1);
    size_t n_data     = n_lines - (has_header ? 1u : 0u);
    if (n_data == 0) { fclose(f); return NULL; }

    char   class_buf[DT_MAX_CLASSES][DT_LABEL_LEN];
    int    n_classes = 0;
    int   *tmp_y   = (int *)malloc(n_data * sizeof(int));
    double **tmp_X = (double **)malloc(n_data * sizeof(double *));
    if (!tmp_y || !tmp_X) { free(tmp_y); free(tmp_X); fclose(f); return NULL; }

    for (size_t i = 0; i < n_data; i++) {
        tmp_X[i] = (double *)malloc(n_features * sizeof(double));
        if (!tmp_X[i]) {
            for (size_t j = 0; j < i; j++) free(tmp_X[j]);
            free(tmp_X); free(tmp_y); fclose(f); return NULL;
        }
    }

    // второй проход: читаем данные
    size_t row = 0;
    int    skip_header = has_header;
    while (fgets(line, sizeof(line), f) && row < n_data) {
        if (skip_header) { skip_header = 0; continue; }

        char *tok = strtok(line, ",\n\r");
        size_t col = 0;
        while (tok && col < (size_t)n_cols) {
            while (isspace((unsigned char)*tok)) tok++;
            char *end = tok + strlen(tok) - 1;
            while (end > tok && isspace((unsigned char)*end)) *end-- = '\0';

            if (col < n_features) {
                tmp_X[row][col] = atof(tok);
            } else {
                // последний столбец — метка, ищем или добавляем класс
                int found = -1;
                for (int c = 0; c < n_classes; c++)
                    if (strcmp(class_buf[c], tok) == 0) { found = c; break; }
                if (found < 0) {
                    if (n_classes >= DT_MAX_CLASSES) {
                        fprintf(stderr, "Too many classes\n");
                        goto cleanup_err;
                    }
                    strncpy(class_buf[n_classes], tok, DT_LABEL_LEN - 1);
                    class_buf[n_classes][DT_LABEL_LEN - 1] = '\0';
                    found = n_classes++;
                }
                tmp_y[row] = found;
            }
            col++;
            tok = strtok(NULL, ",\n\r");
        }
        row++;
    }
    fclose(f);

    Dataset *ds = dataset_alloc(n_data, n_features, n_classes);
    if (!ds) goto cleanup_nomem;

    for (size_t i = 0; i < n_data; i++) {
        memcpy(ds->X[i], tmp_X[i], n_features * sizeof(double));
        ds->y[i] = tmp_y[i];
        free(tmp_X[i]);
    }
    for (int c = 0; c < n_classes; c++) {
        size_t slen = strlen(class_buf[c]);
        if (slen >= DT_LABEL_LEN) slen = DT_LABEL_LEN - 1;
        memcpy(ds->class_names[c], class_buf[c], slen);
        ds->class_names[c][slen] = '\0';
    }

    free(tmp_X); free(tmp_y);
    return ds;

cleanup_err:
    fclose(f);
cleanup_nomem:
    for (size_t i = 0; i < n_data; i++) free(tmp_X[i]);
    free(tmp_X); free(tmp_y);
    return NULL;
}

//  Метрики неоднородности 

// индекс Джини: G = 1 - sum(p^2)
double dt_gini(const int *counts, int n_classes, int total)
{
    if (total <= 0) return 0.0;
    double g = 1.0;
    for (int c = 0; c < n_classes; c++) {
        double p = (double)counts[c] / (double)total;
        g -= p * p;
    }
    return g;
}

// энтропия Шеннона: H = -sum(p * log2(p))
double dt_entropy(const int *counts, int n_classes, int total)
{
    if (total <= 0) return 0.0;
    double h = 0.0;
    for (int c = 0; c < n_classes; c++) {
        if (counts[c] == 0) continue;
        double p = (double)counts[c] / (double)total;
        h -= p * log2(p);
    }
    return h;
}

//  Внутренние функции дерева 

// подсчёт образцов каждого класса
static void count_classes(const int *y, const int *idx, size_t n,
                           int *out, int n_classes)
{
    memset(out, 0, (size_t)n_classes * sizeof(int));
    for (size_t i = 0; i < n; i++) {
        int c = y[idx[i]];
        if (c >= 0 && c < n_classes) out[c]++;
    }
}

// класс с наибольшим числом голосов
static int majority(const int *counts, int n_classes)
{
    int best = 0;
    for (int c = 1; c < n_classes; c++)
        if (counts[c] > counts[best]) best = c;
    return best;
}

// создание листового узла
static DTNode *make_leaf(const int *y, const int *idx, size_t n, int n_classes)
{
    DTNode *node = (DTNode *)calloc(1, sizeof(DTNode));
    if (!node) return NULL;
    node->is_leaf   = 1;
    node->n_classes = n_classes;
    count_classes(y, idx, n, node->class_counts, n_classes);
    node->class_label = majority(node->class_counts, n_classes);
    return node;
}

// поиск лучшего разбиения по случайному подмножеству признаков
static int find_best_split(const Dataset *ds, const int *idx, size_t n,
                            int n_feat_try, unsigned int *rng,
                            int *out_feat, double *out_thr)
{
    int n_classes  = ds->n_classes;
    int n_features = (int)ds->n_features;

    // перемешиваем индексы признаков (Fisher-Yates)
    int feat_order[DT_MAX_FEATURES];
    for (int i = 0; i < n_features; i++) feat_order[i] = i;
    for (int i = n_features - 1; i > 0; i--) {
        int j = lcg_range(rng, 0, i + 1);
        int tmp = feat_order[i]; feat_order[i] = feat_order[j]; feat_order[j] = tmp;
    }

    double best_gini = DBL_MAX;
    *out_feat = -1;

    int counts_all[DT_MAX_CLASSES];
    count_classes(ds->y, idx, n, counts_all, n_classes);

    double *vals = (double *)malloc(n * sizeof(double));
    int    *sidx = (int    *)malloc(n * sizeof(int));
    if (!vals || !sidx) { free(vals); free(sidx); return 0; }

    int tried = 0;
    for (int fi = 0; fi < n_features && tried < n_feat_try; fi++, tried++) {
        int f = feat_order[fi];

        for (size_t i = 0; i < n; i++) {
            vals[i] = ds->X[idx[i]][f];
            sidx[i] = (int)i;
        }

        // insertion sort по значению признака
        for (size_t i = 1; i < n; i++) {
            double kv = vals[i]; int ki = sidx[i];
            size_t j = i;
            while (j > 0 && vals[j-1] > kv) {
                vals[j] = vals[j-1]; sidx[j] = sidx[j-1]; j--;
            }
            vals[j] = kv; sidx[j] = ki;
        }

        // перебор порогов — midpoint между соседними уникальными значениями
        int left_cnt [DT_MAX_CLASSES]; memset(left_cnt,  0, sizeof(left_cnt));
        int right_cnt[DT_MAX_CLASSES];
        memcpy(right_cnt, counts_all, (size_t)n_classes * sizeof(int));

        for (size_t k = 0; k < n - 1; k++) {
            int c = ds->y[idx[sidx[k]]];
            left_cnt[c]++;
            right_cnt[c]--;

            if (vals[k] >= vals[k+1] - 1e-10) continue;

            double thr = (vals[k] + vals[k+1]) * 0.5;
            int nl = (int)(k + 1), nr = (int)(n - k - 1);

            // взвешенный Джини двух дочерних узлов
            double g = ((double)nl * dt_gini(left_cnt,  n_classes, nl) +
                        (double)nr * dt_gini(right_cnt, n_classes, nr)) / (double)n;

            if (g < best_gini) {
                best_gini = g;
                *out_feat = f;
                *out_thr  = thr;
            }
        }
    }

    free(vals); free(sidx);
    return (*out_feat >= 0) ? 1 : 0;
}

// рекурсивное построение узла дерева (CART)
static DTNode *build_node(const Dataset *ds, const int *idx, size_t n,
                           size_t depth, const DecisionTree *dt,
                           unsigned int *rng)
{
    int n_classes = ds->n_classes;

    // критерии остановки: глубина, мало образцов, узел чистый
    if (n < (size_t)dt->min_samples_split || depth >= dt->max_depth)
        return make_leaf(ds->y, idx, n, n_classes);

    int counts[DT_MAX_CLASSES];
    count_classes(ds->y, idx, n, counts, n_classes);
    int nonzero = 0;
    for (int c = 0; c < n_classes; c++) if (counts[c]) nonzero++;
    if (nonzero <= 1) return make_leaf(ds->y, idx, n, n_classes);

    int    best_feat = -1;
    double best_thr  = 0.0;
    int n_feat_try   = dt->n_features_split > 0
                       ? dt->n_features_split : (int)ds->n_features;

    if (!find_best_split(ds, idx, n, n_feat_try, rng, &best_feat, &best_thr))
        return make_leaf(ds->y, idx, n, n_classes);

    // делим индексы на левую и правую ветки
    int *left_idx  = (int *)malloc(n * sizeof(int));
    int *right_idx = (int *)malloc(n * sizeof(int));
    if (!left_idx || !right_idx) {
        free(left_idx); free(right_idx);
        return make_leaf(ds->y, idx, n, n_classes);
    }

    size_t nl = 0, nr = 0;
    for (size_t i = 0; i < n; i++) {
        if (ds->X[idx[i]][best_feat] <= best_thr)
            left_idx[nl++]  = idx[i];
        else
            right_idx[nr++] = idx[i];
    }

    if (nl == 0 || nr == 0) {
        free(left_idx); free(right_idx);
        return make_leaf(ds->y, idx, n, n_classes);
    }

    DTNode *node = (DTNode *)calloc(1, sizeof(DTNode));
    if (!node) { free(left_idx); free(right_idx); return NULL; }

    node->is_leaf     = 0;
    node->feature_idx = best_feat;
    node->threshold   = best_thr;
    node->n_classes   = n_classes;

    // рекурсия в дочерние узлы
    node->left  = build_node(ds, left_idx,  nl, depth + 1, dt, rng);
    node->right = build_node(ds, right_idx, nr, depth + 1, dt, rng);

    free(left_idx);
    free(right_idx);
    return node;
}

static void free_node(DTNode *node)
{
    if (!node) return;
    free_node(node->left);
    free_node(node->right);
    free(node);
}

//  Decision Tree — публичный API 

DecisionTree *dt_create(size_t max_depth, int min_samples_split,
                         int n_features_split, unsigned int seed)
{
    DecisionTree *dt = (DecisionTree *)calloc(1, sizeof(DecisionTree));
    if (!dt) return NULL;
    dt->max_depth         = max_depth > 0 ? max_depth : 10;
    dt->min_samples_split = min_samples_split > 1 ? min_samples_split : 2;
    dt->n_features_split  = n_features_split;
    dt->seed              = seed;
    return dt;
}

void dt_free(DecisionTree *dt)
{
    if (!dt) return;
    free_node(dt->root);
    free(dt);
}

void dt_fit(DecisionTree *dt, const Dataset *ds,
            const int *indices, size_t n_idx)
{
    assert(dt && ds && indices && n_idx > 0);
    free_node(dt->root);

    unsigned int rng = dt->seed;

    // по умолчанию пробуем sqrt(n_features) признаков на узел
    if (dt->n_features_split <= 0) {
        dt->n_features_split = (int)sqrt((double)ds->n_features);
        if (dt->n_features_split < 1) dt->n_features_split = 1;
    }

    dt->root = build_node(ds, indices, n_idx, 0, dt, &rng);
}

// DFS: идём от корня до листа, возвращаем класс
int dt_predict(const DecisionTree *dt, const double *x)
{
    assert(dt && x);
    const DTNode *node = dt->root;
    if (!node) return -1;

    while (!node->is_leaf) {
        if (x[node->feature_idx] <= node->threshold)
            node = node->left;
        else
            node = node->right;
        if (!node) return -1;
    }
    return node->class_label;
}

//  Bagging — публичный API 

BaggingClassifier *bag_create(size_t n_trees, size_t max_depth,
                               int min_samples_split, unsigned int seed)
{
    assert(n_trees > 0);
    BaggingClassifier *bc = (BaggingClassifier *)calloc(1, sizeof(BaggingClassifier));
    if (!bc) return NULL;

    bc->trees = (DecisionTree **)calloc(n_trees, sizeof(DecisionTree *));
    if (!bc->trees) { free(bc); return NULL; }

    bc->n_trees           = n_trees;
    bc->seed              = seed;
    bc->max_depth         = max_depth;
    bc->min_samples_split = min_samples_split;
    return bc;
}

void bag_free(BaggingClassifier *bc)
{
    if (!bc) return;
    for (size_t i = 0; i < bc->n_trees; i++) dt_free(bc->trees[i]);
    free(bc->trees);
    free(bc);
}

// bootstrap aggregating: каждое дерево учится на случайной выборке с возвращением
void bag_fit(BaggingClassifier *bc, const Dataset *ds)
{
    assert(bc && ds && ds->n_samples > 0);
    bc->n_classes = ds->n_classes;

    unsigned int rng = bc->seed;
    int n_feat_split = (int)sqrt((double)ds->n_features);
    if (n_feat_split < 1) n_feat_split = 1;

    int *bootstrap = (int *)malloc(ds->n_samples * sizeof(int));
    if (!bootstrap) return;

    for (size_t t = 0; t < bc->n_trees; t++) {
        unsigned int tree_seed = lcg_next(&rng);

        // bootstrap-выборка с возвращением
        for (size_t i = 0; i < ds->n_samples; i++)
            bootstrap[i] = lcg_range(&rng, 0, (int)ds->n_samples);

        bc->trees[t] = dt_create(bc->max_depth, bc->min_samples_split,
                                  n_feat_split, tree_seed);
        if (!bc->trees[t]) continue;
        dt_fit(bc->trees[t], ds, bootstrap, ds->n_samples);
    }
    free(bootstrap);
}

// мажоритарное голосование всех деревьев
int bag_predict(const BaggingClassifier *bc, const double *x)
{
    assert(bc && x && bc->n_classes > 0);
    int votes[DT_MAX_CLASSES] = {0};
    for (size_t t = 0; t < bc->n_trees; t++) {
        if (!bc->trees[t]) continue;
        int pred = dt_predict(bc->trees[t], x);
        if (pred >= 0 && pred < bc->n_classes)
            votes[pred]++;
    }
    return majority(votes, bc->n_classes);
}

// доля правильных предсказаний на датасете
double bag_score(const BaggingClassifier *bc, const Dataset *ds)
{
    assert(bc && ds);
    if (ds->n_samples == 0) return 0.0;
    int correct = 0;
    for (size_t i = 0; i < ds->n_samples; i++) {
        if (bag_predict(bc, ds->X[i]) == ds->y[i]) correct++;
    }
    return (double)correct / (double)ds->n_samples;
}
