// decision_tree.h — публичный API дерева решений с бэггингом

#ifndef DECISION_TREE_H
#define DECISION_TREE_H

#include <stddef.h>

//  Ограничения 
#define DT_MAX_CLASSES  16
#define DT_MAX_FEATURES 64
#define DT_LABEL_LEN    64

//  Датасет 
typedef struct {
    double **X;        // [n_samples][n_features]
    int     *y;        // [n_samples] — индексы классов
    size_t   n_samples;
    size_t   n_features;
    int      n_classes;
    char     class_names[DT_MAX_CLASSES][DT_LABEL_LEN];
} Dataset;

Dataset *dataset_alloc(size_t n_samples, size_t n_features, int n_classes);
Dataset *dataset_load_csv(const char *path, int has_header);
void     dataset_free(Dataset *ds);

//  Узел дерева 
typedef struct DTNode {
    int is_leaf;

    // внутренний узел
    int    feature_idx;
    double threshold;
    struct DTNode *left;   // X[f] <= threshold
    struct DTNode *right;  // X[f] >  threshold

    // лист
    int class_label;
    int class_counts[DT_MAX_CLASSES];
    int n_classes;
} DTNode;

//  Дерево решений 
typedef struct {
    DTNode      *root;
    size_t       max_depth;
    int          min_samples_split;
    int          n_features_split;  // кол-во признаков на узел
    unsigned int seed;
} DecisionTree;

DecisionTree *dt_create(size_t max_depth, int min_samples_split,
                         int n_features_split, unsigned int seed);
void          dt_free(DecisionTree *dt);
void          dt_fit(DecisionTree *dt, const Dataset *ds,
                     const int *indices, size_t n_idx);
int           dt_predict(const DecisionTree *dt, const double *x);

//  Бэггинг 
typedef struct {
    DecisionTree **trees;
    size_t         n_trees;
    int            n_classes;
    unsigned int   seed;
    size_t         max_depth;
    int            min_samples_split;
} BaggingClassifier;

BaggingClassifier *bag_create(size_t n_trees, size_t max_depth,
                               int min_samples_split, unsigned int seed);
void               bag_free(BaggingClassifier *bc);
void               bag_fit(BaggingClassifier *bc, const Dataset *ds);
int                bag_predict(const BaggingClassifier *bc, const double *x);
double             bag_score(const BaggingClassifier *bc, const Dataset *ds);

//  Утилиты (экспортируются для тестов) 
double dt_gini(const int *counts, int n_classes, int total);
double dt_entropy(const int *counts, int n_classes, int total);

#endif 
