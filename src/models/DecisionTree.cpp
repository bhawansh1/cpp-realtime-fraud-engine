#include "mlengine/models/DecisionTree.hpp"
#include <algorithm>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <iostream>

namespace veloce {

DecisionTree::DecisionTree(DecisionTreeConfig config) : config_(config), root_(nullptr) {}

DecisionTree::DecisionTree(const DecisionTree& other) : config_(other.config_), root_(clone_node(other.root_.get())) {}

DecisionTree& DecisionTree::operator=(const DecisionTree& other) {
    if (this != &other) {
        config_ = other.config_;
        root_ = clone_node(other.root_.get());
    }
    return *this;
}

std::unique_ptr<TreeNode> DecisionTree::clone_node(const TreeNode* node) {
    if (!node) return nullptr;
    auto copy = std::make_unique<TreeNode>();
    copy->is_leaf = node->is_leaf;
    copy->feature_idx = node->feature_idx;
    copy->threshold = node->threshold;
    copy->class_label = node->class_label;
    copy->probability = node->probability;
    copy->left = clone_node(node->left.get());
    copy->right = clone_node(node->right.get());
    return copy;
}

double DecisionTree::calculate_gini(const Matrix& y, const std::vector<size_t>& indices) const {
    if (indices.empty()) return 0.0;
    size_t count_1 = 0;
    for (size_t idx : indices) {
        if (y(idx, 0) >= 0.5) count_1++;
    }
    double p1 = static_cast<double>(count_1) / static_cast<double>(indices.size());
    double p0 = 1.0 - p1;
    return 1.0 - (p0 * p0 + p1 * p1);
}

std::unique_ptr<TreeNode> DecisionTree::build_tree(const Matrix& X, const Matrix& y,
                                                   const std::vector<size_t>& sample_indices,
                                                   size_t current_depth) {
    auto node = std::make_unique<TreeNode>();
    const size_t N = sample_indices.size();

    // Compute node statistics
    size_t count_1 = 0;
    for (size_t idx : sample_indices) {
        if (y(idx, 0) >= 0.5) count_1++;
    }
    double p1 = static_cast<double>(count_1) / static_cast<double>(N);
    node->probability = p1;
    node->class_label = (p1 >= 0.5) ? 1.0 : 0.0;

    double base_gini = 1.0 - ((1.0 - p1) * (1.0 - p1) + p1 * p1);

    // Check base stopping criteria
    if (base_gini <= 1e-9 || current_depth >= config_.max_depth || N < config_.min_samples_split) {
        node->is_leaf = true;
        return node;
    }

    const size_t D = X.cols();
    double best_gain = -1.0;
    size_t best_feature = 0;
    double best_threshold = 0.0;

    // Fast O(D * N log N) greedy split search
    struct ValueTarget {
        double val;
        double target;
        size_t original_idx;
    };
    std::vector<ValueTarget> feat_values(N);

    for (size_t f = 0; f < D; ++f) {
        for (size_t i = 0; i < N; ++i) {
            size_t idx = sample_indices[i];
            feat_values[i] = {X(idx, f), y(idx, 0), idx};
        }

        std::sort(feat_values.begin(), feat_values.end(),
                  [](const ValueTarget& a, const ValueTarget& b) { return a.val < b.val; });

        // Running counts for left and right partitions
        size_t left_count = 0;
        size_t left_1 = 0;
        size_t right_count = N;
        size_t right_1 = count_1;

        for (size_t i = 0; i < N - 1; ++i) {
            left_count++;
            right_count--;
            if (feat_values[i].target >= 0.5) {
                left_1++;
                right_1--;
            }

            // Check if adjacent values are distinct (potential split threshold)
            if (feat_values[i].val == feat_values[i + 1].val) {
                continue;
            }

            if (left_count < config_.min_samples_leaf || right_count < config_.min_samples_leaf) {
                continue;
            }

            double p1_left = static_cast<double>(left_1) / static_cast<double>(left_count);
            double p0_left = 1.0 - p1_left;
            double gini_left = 1.0 - (p0_left * p0_left + p1_left * p1_left);

            double p1_right = static_cast<double>(right_1) / static_cast<double>(right_count);
            double p0_right = 1.0 - p1_right;
            double gini_right = 1.0 - (p0_right * p0_right + p1_right * p1_right);

            double weighted_gini = (static_cast<double>(left_count) / static_cast<double>(N)) * gini_left +
                                   (static_cast<double>(right_count) / static_cast<double>(N)) * gini_right;

            double gain = base_gini - weighted_gini;
            if (gain > best_gain) {
                best_gain = gain;
                best_feature = f;
                best_threshold = 0.5 * (feat_values[i].val + feat_values[i + 1].val);
            }
        }
    }

    if (best_gain < config_.min_impurity_decrease) {
        node->is_leaf = true;
        return node;
    }

    // Split samples into left and right subsets
    std::vector<size_t> left_indices;
    std::vector<size_t> right_indices;
    left_indices.reserve(N);
    right_indices.reserve(N);

    for (size_t idx : sample_indices) {
        if (X(idx, best_feature) <= best_threshold) {
            left_indices.push_back(idx);
        } else {
            right_indices.push_back(idx);
        }
    }

    if (left_indices.empty() || right_indices.empty()) {
        node->is_leaf = true;
        return node;
    }

    node->is_leaf = false;
    node->feature_idx = best_feature;
    node->threshold = best_threshold;
    node->left = build_tree(X, y, left_indices, current_depth + 1);
    node->right = build_tree(X, y, right_indices, current_depth + 1);

    return node;
}

void DecisionTree::fit(const Matrix& X, const Matrix& y) {
    if (X.rows() != y.rows()) {
        throw std::invalid_argument("Features X and target y row count mismatch.");
    }
    const size_t N = X.rows();
    std::vector<size_t> all_indices(N);
    std::iota(all_indices.begin(), all_indices.end(), 0);

    root_ = build_tree(X, y, all_indices, 0);
}

std::pair<double, double> DecisionTree::predict_sample(const TreeNode* node, const Matrix& X, size_t row_idx) const {
    if (!node) return {0.0, 0.0};
    if (node->is_leaf) {
        return {node->class_label, node->probability};
    }
    if (X(row_idx, node->feature_idx) <= node->threshold) {
        return predict_sample(node->left.get(), X, row_idx);
    } else {
        return predict_sample(node->right.get(), X, row_idx);
    }
}

Matrix DecisionTree::predict_proba(const Matrix& X) const {
    if (!root_) throw std::runtime_error("Cannot predict on untrained DecisionTree.");
    Matrix probas(X.rows(), 1);
    for (size_t i = 0; i < X.rows(); ++i) {
        auto [label, prob] = predict_sample(root_.get(), X, i);
        probas(i, 0) = prob;
    }
    return probas;
}

Matrix DecisionTree::predict(const Matrix& X) const {
    if (!root_) throw std::runtime_error("Cannot predict on untrained DecisionTree.");
    Matrix preds(X.rows(), 1);
    for (size_t i = 0; i < X.rows(); ++i) {
        auto [label, prob] = predict_sample(root_.get(), X, i);
        preds(i, 0) = label;
    }
    return preds;
}

size_t DecisionTree::count_nodes(const TreeNode* node) {
    if (!node) return 0;
    return 1 + count_nodes(node->left.get()) + count_nodes(node->right.get());
}

size_t DecisionTree::compute_depth(const TreeNode* node) {
    if (!node || node->is_leaf) return 0;
    return 1 + std::max(compute_depth(node->left.get()), compute_depth(node->right.get()));
}

void DecisionTree::serialize_node(std::ostream& os, const TreeNode* node) const {
    if (!node) {
        os << "NULL\n";
        return;
    }
    if (node->is_leaf) {
        os << "LEAF " << node->class_label << " " << node->probability << "\n";
    } else {
        os << "SPLIT " << node->feature_idx << " " << node->threshold << " " << node->probability << "\n";
        serialize_node(os, node->left.get());
        serialize_node(os, node->right.get());
    }
}

std::unique_ptr<TreeNode> DecisionTree::deserialize_node(std::istream& is) {
    std::string type;
    if (!(is >> type) || type == "NULL") {
        return nullptr;
    }
    auto node = std::make_unique<TreeNode>();
    if (type == "LEAF") {
        node->is_leaf = true;
        is >> node->class_label >> node->probability;
    } else if (type == "SPLIT") {
        node->is_leaf = false;
        is >> node->feature_idx >> node->threshold >> node->probability;
        node->left = deserialize_node(is);
        node->right = deserialize_node(is);
    }
    return node;
}

void DecisionTree::save(const std::string& filepath) const {
    std::ofstream out(filepath);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open file for saving tree: " + filepath);
    }
    out << "VELOCEML_DECISION_TREE_V1\n";
    serialize_node(out, root_.get());
}

void DecisionTree::load(const std::string& filepath) {
    std::ifstream in(filepath);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open file for loading tree: " + filepath);
    }
    std::string header;
    in >> header;
    if (header != "VELOCEML_DECISION_TREE_V1") {
        throw std::runtime_error("Invalid tree model format: " + header);
    }
    root_ = deserialize_node(in);
}

} // namespace veloce
