#include "mlengine/math/Matrix.hpp"
#include "mlengine/data/Dataset.hpp"
#include "mlengine/models/LogisticRegression.hpp"
#include "mlengine/models/DecisionTree.hpp"
#include "mlengine/metrics/Metrics.hpp"
#include <iostream>
#include <fstream>
#include <filesystem>

using namespace veloce;

int main() {
    std::cout << "========================================================\n";
    std::cout << "  VeloceML: Real-Time Financial Fraud Detection Pipeline \n";
    std::cout << "========================================================\n\n";

    std::filesystem::create_directories("data");
    std::filesystem::create_directories("models");

    // 1. Generate or load realistic fraud transaction dataset
    const size_t num_samples = 5000;
    const size_t num_features = 6;
    const double fraud_ratio = 0.15; // 15% fraud prevalence
    std::cout << "[1/5] Generating synthetic financial fraud dataset ("
              << num_samples << " transactions, " << num_features << " features)...\n";

    auto [X, y] = Dataset::generate_synthetic_fraud_dataset(num_samples, num_features, fraud_ratio, 42);

    // Export dataset to CSV for verification and Python benchmarking
    std::string csv_path = "data/fraud_transactions.csv";
    {
        std::ofstream csv(csv_path);
        csv << "amount,dist_home,dist_last_tx,freq_24h,device_risk,pin_retries,is_fraud\n";
        for (size_t r = 0; r < X.rows(); ++r) {
            for (size_t c = 0; c < X.cols(); ++c) {
                csv << X(r, c) << ",";
            }
            csv << y(r, 0) << "\n";
        }
    }
    std::cout << "      Dataset exported to: " << csv_path << "\n";

    // 2. Train / Test Split (80% train, 20% test)
    std::cout << "[2/5] Partitioning data: 80% Train, 20% Test...\n";
    auto split = Dataset::train_test_split(X, y, 0.20, true, 42);
    std::cout << "      Train set: " << split.X_train.rows() << " rows | "
              << "Test set: " << split.X_test.rows() << " rows\n";

    // 3. Feature Scaling (Z-Score normalization)
    std::cout << "[3/5] Fitting StandardScaler on training set and scaling...\n";
    StandardScaler scaler;
    Matrix X_train_scaled = scaler.fit_transform(split.X_train);
    Matrix X_test_scaled = scaler.transform(split.X_test);
    scaler.save("models/fraud_scaler.bin");
    std::cout << "      Scaler parameters saved to: models/fraud_scaler.bin\n";

    // 4. Model Training: Logistic Regression (Adam Optimizer)
    std::cout << "\n[4/5] Training Model 1: Logistic Regression (Adam, L2 Regularized)...\n";
    LogisticConfig log_cfg;
    log_cfg.learning_rate = 0.05;
    log_cfg.max_epochs = 120;
    log_cfg.batch_size = 64;
    log_cfg.l2_lambda = 0.01;
    log_cfg.use_adam = true;
    log_cfg.verbose = false;

    LogisticRegression log_model(log_cfg);
    log_model.fit(X_train_scaled, split.y_train);
    log_model.save("models/fraud_logistic.bin");
    std::cout << "      Logistic Regression training complete. Model saved to: models/fraud_logistic.bin\n";

    // Evaluate Logistic Regression on Test Set
    Matrix log_test_preds = log_model.predict(X_test_scaled);
    Matrix log_test_proba = log_model.predict_proba(X_test_scaled);
    Metrics::print_report(split.y_test, log_test_preds, "Logistic Regression (Adam)");
    std::cout << "  Test Log Loss: " << Metrics::log_loss(split.y_test, log_test_proba) << "\n";

    // 5. Model Training: Decision Tree Classifier
    std::cout << "\n[5/5] Training Model 2: Decision Tree Classifier (Gini Impurity)...\n";
    DecisionTreeConfig tree_cfg;
    tree_cfg.max_depth = 8;
    tree_cfg.min_samples_split = 10;
    tree_cfg.min_samples_leaf = 4;
    tree_cfg.min_impurity_decrease = 0.0;

    DecisionTree tree_model(tree_cfg);
    // Decision trees are invariant to monotonic feature scaling, but we pass scaled for consistency
    tree_model.fit(X_train_scaled, split.y_train);
    tree_model.save("models/fraud_tree.bin");
    std::cout << "      Decision Tree training complete (" << tree_model.node_count()
              << " nodes, depth " << tree_model.max_tree_depth()
              << "). Model saved to: models/fraud_tree.bin\n";

    // Evaluate Decision Tree on Test Set
    Matrix tree_test_preds = tree_model.predict(X_test_scaled);
    Metrics::print_report(split.y_test, tree_test_preds, "Decision Tree Classifier");

    std::cout << ">>> PIPELINE COMPLETED SUCCESSFULLY! <<<\n";
    return 0;
}
