#pragma once

#include "mlengine/models/IModel.hpp"
#include "mlengine/data/Dataset.hpp"
#include <string>
#include <memory>
#include <functional>
#include <unordered_map>
#include <atomic>
#include <thread>
#include <vector>

namespace veloce {

struct HttpRequest {
    std::string method;
    std::string path;
    std::string query_string;
    std::unordered_map<std::string, std::string> headers;
    std::string body;
};

struct HttpResponse {
    int status_code{200};
    std::string content_type{"application/json; charset=utf-8"};
    std::unordered_map<std::string, std::string> headers;
    std::string body;

    static HttpResponse json(int code, const std::string& json_str);
    static HttpResponse html(int code, const std::string& html_str);
    static HttpResponse text(int code, const std::string& text_str);
};

using HttpHandler = std::function<HttpResponse(const HttpRequest&)>;

class HttpServer {
public:
    HttpServer(int port = 8080, size_t num_threads = 4);
    ~HttpServer();

    void route(const std::string& method, const std::string& path, HttpHandler handler);

    void start();
    void stop();

    // Model Registry
    void register_model(const std::string& name, std::shared_ptr<IModel> model);
    void set_scaler(std::shared_ptr<StandardScaler> scaler);
    void set_static_html(const std::string& html_content);

    // Metrics counters
    [[nodiscard]] size_t total_requests() const noexcept { return total_requests_.load(); }
    [[nodiscard]] size_t fraud_detected_count() const noexcept { return fraud_detected_.load(); }
    void record_request(bool is_fraud, double latency_us);

private:
    int port_;
    int server_fd_{-1};
    std::atomic<bool> is_running_{false};
    std::thread listener_thread_;

    std::unordered_map<std::string, HttpHandler> routes_;
    std::unordered_map<std::string, std::shared_ptr<IModel>> models_;
    std::shared_ptr<StandardScaler> scaler_{nullptr};
    std::string static_html_;

    std::atomic<size_t> total_requests_{0};
    std::atomic<size_t> fraud_detected_{0};
    std::atomic<double> total_latency_us_{0.0};
    std::chrono::steady_clock::time_point start_time_;

    void listen_loop();
    void handle_client(int client_fd);
    [[nodiscard]] HttpRequest parse_request(const std::string& raw_http);
    void send_response(int client_fd, const HttpResponse& resp);

    void setup_default_routes();
};

} // namespace veloce
