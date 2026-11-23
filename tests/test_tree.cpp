#include "mlengine/models/DecisionTree.hpp"
#include <cassert>
#include <iostream>

using namespace veloce;

void run_tree_tests() {
    std::cout << "[TEST] Running Decision Tree tests...\n";

    Matrix X = {
        { 1.0,  1.0}, // 0: Class 1
        { 1.5,  0.8}, // 1: Class 1
        {-1.0,  1.0}, // 2: Class 0
        {-0.8,  1.2}, // 3: Class 0
        {-1.0, -1.0}, // 4: Class 1
        {-1.2, -0.9}, // 5: Class 1
        { 1.0, -1.0}, // 6: Class 0
        { 0.9, -1.1}  // 7: Class 0
    };
    Matrix y = {
        {1.0},
        {1.0},
        {0.0},
        {0.0},
        {1.0},
        {1.0},
        {0.0},
        {0.0}
    };

    DecisionTreeConfig cfg;
    cfg.max_depth = 6;
    cfg.min_samples_split = 2;
    cfg.min_samples_leaf = 1;
    DecisionTree tree(cfg);
    tree.fit(X, y);

    assert(tree.node_count() > 1);
    assert(tree.max_tree_depth() >= 2);

    Matrix preds = tree.predict(X);
    for (size_t i = 0; i < X.rows(); ++i) {
        assert(preds(i, 0) == y(i, 0));
    }

    // Test serialization & deserialization
    std::string test_tree_path = "test_tree_model.bin";
    tree.save(test_tree_path);

    DecisionTree loaded_tree;
    loaded_tree.load(test_tree_path);
    Matrix loaded_preds = loaded_tree.predict(X);
    for (size_t i = 0; i < X.rows(); ++i) {
        assert(loaded_preds(i, 0) == y(i, 0));
    }

    std::cout << "[PASS] Decision Tree tests passed (100% accuracy, nodes: "
              << tree.node_count() << ", depth: " << tree.max_tree_depth() << ")!\n";
}
