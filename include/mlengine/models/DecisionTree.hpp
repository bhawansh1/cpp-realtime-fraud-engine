#pragma once

#include "mlengine/models/IModel.hpp"
#include <memory>
#include <vector>

namespace veloce {

struct DecisionTreeConfig {
    size_t max_depth{10};
    size_t min_samples_split{2};
    size_t min_samples_leaf{1};
    double min_impurity_decrease{0.0};
};

struct TreeNode {
    bool is_leaf{false};
    size_t feature_idx{0};
    double threshold{0.0};
    double class_label{0.0};
    double probability{0.0}; // Probability of class 1

    std::unique_ptr<TreeNode> left{nullptr};
    std::unique_ptr<TreeNode> right{nullptr};
};

class DecisionTree : public IModel {
public:
    explicit DecisionTree(DecisionTreeConfig config = {});
    ~DecisionTree() override = default;

    // Rule of 5: move-only or custom deep copy
    DecisionTree(const DecisionTree& other);
    DecisionTree& operator=(const DecisionTree& other);
    DecisionTree(DecisionTree&& other) noexcept = default;
    DecisionTree& operator=(DecisionTree&& other) noexcept = default;

    void fit(const Matrix& X, const Matrix& y) override;
    [[nodiscard]] Matrix predict(const Matrix& X) const override;
    [[nodiscard]] Matrix predict_proba(const Matrix& X) const override;

    void save(const std::string& filepath) const override;
    void load(const std::string& filepath) override;

    [[nodiscard]] std::string name() const override { return "DecisionTree"; }
    [[nodiscard]] size_t node_count() const noexcept { return count_nodes(root_.get()); }
    [[nodiscard]] size_t max_tree_depth() const noexcept { return compute_depth(root_.get()); }

private:
    DecisionTreeConfig config_;
    std::unique_ptr<TreeNode> root_{nullptr};

    std::unique_ptr<TreeNode> build_tree(const Matrix& X, const Matrix& y,
                                         const std::vector<size_t>& sample_indices,
                                         size_t current_depth);

    [[nodiscard]] double calculate_gini(const Matrix& y, const std::vector<size_t>& indices) const;

    [[nodiscard]] std::pair<double, double> predict_sample(const TreeNode* node,
                                                           const Matrix& X,
                                                           size_t row_idx) const;

    static size_t count_nodes(const TreeNode* node);
    static size_t compute_depth(const TreeNode* node);
    static std::unique_ptr<TreeNode> clone_node(const TreeNode* node);

    void serialize_node(std::ostream& os, const TreeNode* node) const;
    std::unique_ptr<TreeNode> deserialize_node(std::istream& is);
};

} // namespace veloce
