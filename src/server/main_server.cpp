#include "mlengine/server/HttpServer.hpp"
#include "mlengine/server/EmbeddedDashboard.hpp"
#include "mlengine/models/LogisticRegression.hpp"
#include "mlengine/models/DecisionTree.hpp"
#include "mlengine/data/Dataset.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <filesystem>

using namespace veloce;

static std::string read_file_to_string(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) return "";
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

int main(int argc, char* argv[]) {
    int port = 8080;
    if (const char* env_p = std::getenv("PORT")) {
        port = std::stoi(env_p);
    } else if (argc > 1) {
        port = std::stoi(argv[1]);
    }

    std::cout << "========================================================\n";
    std::cout << "        VeloceML High-Performance C++ Server           \n";
    std::cout << "========================================================\n\n";

    // Locate models directory (supports running from repo root or build/)
    std::string models_dir = "models";
    if (!std::filesystem::exists("models") && std::filesystem::exists("../models")) {
        models_dir = "../models";
    }
    std::filesystem::create_directories(models_dir);

    std::string log_model_path = models_dir + "/fraud_logistic.bin";
    std::string tree_model_path = models_dir + "/fraud_tree.bin";
    std::string scaler_path = models_dir + "/fraud_scaler.bin";

    auto scaler = std::make_shared<StandardScaler>();
    auto log_model = std::make_shared<LogisticRegression>();
    auto tree_model = std::make_shared<DecisionTree>();

    // Self-healing: if models are not on disk, train them automatically
    if (!std::filesystem::exists(log_model_path) ||
        !std::filesystem::exists(tree_model_path) ||
        !std::filesystem::exists(scaler_path)) {
        std::cout << "[*] Models not found on disk. Initializing and training models now...\n";
        auto [X, y] = Dataset::generate_synthetic_fraud_dataset(4000, 6, 0.15, 42);
        Matrix X_scaled = scaler->fit_transform(X);
        scaler->save(scaler_path);

        LogisticConfig log_cfg;
        log_cfg.learning_rate = 0.05;
        log_cfg.max_epochs = 100;
        log_cfg.use_adam = true;
        log_model = std::make_shared<LogisticRegression>(log_cfg);
        log_model->fit(X_scaled, y);
        log_model->save(log_model_path);

        DecisionTreeConfig tree_cfg;
        tree_cfg.max_depth = 8;
        tree_cfg.min_samples_split = 5;
        tree_model = std::make_shared<DecisionTree>(tree_cfg);
        tree_model->fit(X_scaled, y);
        tree_model->save(tree_model_path);
        std::cout << "[*] Initial training complete. Models saved to " << models_dir << "/\n";
    } else {
        std::cout << "[*] Loading pre-trained models from " << models_dir << "/...\n";
        scaler->load(scaler_path);
        log_model->load(log_model_path);
        tree_model->load(tree_model_path);
    }

    std::cout << "[*] Models successfully loaded into memory.\n";

    // Initialize HTTP server
    HttpServer server(port);
    server.set_scaler(scaler);
    server.register_model("logistic", log_model);
    server.register_model("tree", tree_model);

    // Load static HTML UI dashboard (with fallback to embedded binary string)
    std::string html_content = read_file_to_string("web/index.html");
    if (html_content.empty()) {
        html_content = read_file_to_string("../web/index.html");
    }
    if (html_content.empty()) {
        html_content = std::string(EMBEDDED_DASHBOARD_HTML);
        std::cout << "[*] Web dashboard loaded from built-in binary asset (EmbeddedDashboard.hpp)\n";
    } else {
        std::cout << "[*] Web dashboard loaded from disk\n";
    }

    server.set_static_html(html_content);

    std::cout << "\n>>> SERVER READY: Open http://localhost:" << port << " in your browser! <<<\n";
    std::cout << "Endpoints active:\n";
    std::cout << "  - GET  /                  (Interactive Web UI Dashboard)\n";
    std::cout << "  - POST /api/v1/predict    (Sub-microsecond REST Inference)\n";
    std::cout << "  - GET  /health            (Service health status)\n";
    std::cout << "  - GET  /metrics           (Prometheus performance counters)\n\n";

    server.start();

    return 0;
}
