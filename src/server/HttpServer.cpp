#include "mlengine/server/HttpServer.hpp"
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <unistd.h>
#include <fcntl.h>
#include <arpa/inet.h>
#include <sstream>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <regex>

namespace veloce {

HttpResponse HttpResponse::json(int code, const std::string& json_str) {
    HttpResponse res;
    res.status_code = code;
    res.content_type = "application/json; charset=utf-8";
    res.body = json_str;
    return res;
}

HttpResponse HttpResponse::html(int code, const std::string& html_str) {
    HttpResponse res;
    res.status_code = code;
    res.content_type = "text/html; charset=utf-8";
    res.body = html_str;
    return res;
}

HttpResponse HttpResponse::text(int code, const std::string& text_str) {
    HttpResponse res;
    res.status_code = code;
    res.content_type = "text/plain; charset=utf-8";
    res.body = text_str;
    return res;
}

HttpServer::HttpServer(int port, size_t /*num_threads*/)
    : port_(port), start_time_(std::chrono::steady_clock::now()) {
    setup_default_routes();
}

HttpServer::~HttpServer() {
    stop();
}

void HttpServer::route(const std::string& method, const std::string& path, HttpHandler handler) {
    routes_[method + ":" + path] = std::move(handler);
}

void HttpServer::register_model(const std::string& name, std::shared_ptr<IModel> model) {
    models_[name] = model;
}

void HttpServer::set_scaler(std::shared_ptr<StandardScaler> scaler) {
    scaler_ = scaler;
}

void HttpServer::set_static_html(const std::string& html_content) {
    static_html_ = html_content;
}

void HttpServer::record_request(bool is_fraud, double latency_us) {
    total_requests_++;
    if (is_fraud) fraud_detected_++;
    double current = total_latency_us_.load();
    while (!total_latency_us_.compare_exchange_weak(current, current + latency_us)) {}
}

void HttpServer::start() {
    if (is_running_.load()) return;

    server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd_ < 0) {
        throw std::runtime_error("Failed to create socket");
    }

    int opt = 1;
    setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port_);

    if (bind(server_fd_, (struct sockaddr*)&address, sizeof(address)) < 0) {
        close(server_fd_);
        throw std::runtime_error("Socket bind failed on port " + std::to_string(port_));
    }

    if (listen(server_fd_, 128) < 0) {
        close(server_fd_);
        throw std::runtime_error("Socket listen failed");
    }

    is_running_.store(true);
    std::cout << "[VeloceML Server] Listening on http://0.0.0.0:" << port_ << "\n";

    listen_loop();
}

void HttpServer::stop() {
    if (is_running_.exchange(false)) {
        if (server_fd_ >= 0) {
            close(server_fd_);
            server_fd_ = -1;
        }
        if (listener_thread_.joinable()) {
            listener_thread_.join();
        }
    }
}

void HttpServer::listen_loop() {
    while (is_running_.load()) {
        sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(server_fd_, (struct sockaddr*)&client_addr, &client_len);
        if (client_fd < 0) {
            if (!is_running_.load()) break;
            continue;
        }

        // Disable Nagle's algorithm for instant packet dispatch
        int nodelay_flag = 1;
        setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, (char*)&nodelay_flag, sizeof(int));

        // Handle connection in a dedicated thread or thread pool
        std::thread([this, client_fd]() {
            handle_client(client_fd);
        }).detach();
    }
}

void HttpServer::handle_client(int client_fd) {
    std::vector<char> buffer(8192);
    ssize_t bytes_read = read(client_fd, buffer.data(), buffer.size() - 1);
    if (bytes_read <= 0) {
        close(client_fd);
        return;
    }
    buffer[bytes_read] = '\0';

    std::string raw_http(buffer.data(), bytes_read);
    HttpRequest req = parse_request(raw_http);

    HttpResponse res;

    // Handle CORS pre-flight
    if (req.method == "OPTIONS") {
        res.status_code = 204;
        res.body = "";
    } else {
        std::string route_key = req.method + ":" + req.path;
        auto it = routes_.find(route_key);
        if (it != routes_.end()) {
            try {
                res = it->second(req);
            } catch (const std::exception& e) {
                res = HttpResponse::json(500, "{\"error\": \"" + std::string(e.what()) + "\"}");
            }
        } else {
            res = HttpResponse::json(404, "{\"error\": \"Route not found: " + req.path + "\"}");
        }
    }

    send_response(client_fd, res);
    close(client_fd);
}

HttpRequest HttpServer::parse_request(const std::string& raw_http) {
    HttpRequest req;
    std::istringstream stream(raw_http);
    std::string line;

    if (std::getline(stream, line)) {
        std::istringstream line_stream(line);
        line_stream >> req.method >> req.path;

        size_t q_pos = req.path.find('?');
        if (q_pos != std::string::npos) {
            req.query_string = req.path.substr(q_pos + 1);
            req.path = req.path.substr(0, q_pos);
        }
    }

    while (std::getline(stream, line) && line != "\r" && !line.empty()) {
        size_t colon = line.find(':');
        if (colon != std::string::npos) {
            std::string key = line.substr(0, colon);
            std::string val = line.substr(colon + 1);
            val.erase(0, val.find_first_not_of(" \t\r\n"));
            val.erase(val.find_last_not_of(" \t\r\n") + 1);
            req.headers[key] = val;
        }
    }

    std::stringstream body_ss;
    body_ss << stream.rdbuf();
    req.body = body_ss.str();

    return req;
}

void HttpServer::send_response(int client_fd, const HttpResponse& resp) {
    std::string status_msg = (resp.status_code == 200) ? "OK" :
                             (resp.status_code == 204) ? "No Content" :
                             (resp.status_code == 404) ? "Not Found" : "Internal Server Error";

    std::ostringstream ss;
    ss << "HTTP/1.1 " << resp.status_code << " " << status_msg << "\r\n";
    ss << "Content-Type: " << resp.content_type << "\r\n";
    ss << "Content-Length: " << resp.body.size() << "\r\n";
    ss << "Access-Control-Allow-Origin: *\r\n";
    ss << "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n";
    ss << "Access-Control-Allow-Headers: Content-Type, Authorization\r\n";
    ss << "Connection: close\r\n\r\n";
    ss << resp.body;

    std::string msg = ss.str();
    write(client_fd, msg.c_str(), msg.size());
}

// Lightweight JSON helper to extract features array and model choice
static std::vector<double> extract_json_features(const std::string& body) {
    std::vector<double> features;
    size_t feat_pos = body.find("\"features\"");
    if (feat_pos == std::string::npos) return features;

    size_t start_bracket = body.find('[', feat_pos);
    size_t end_bracket = body.find(']', start_bracket);
    if (start_bracket == std::string::npos || end_bracket == std::string::npos) return features;

    std::string arr_str = body.substr(start_bracket + 1, end_bracket - start_bracket - 1);
    std::stringstream ss(arr_str);
    std::string token;
    while (std::getline(ss, token, ',')) {
        token.erase(0, token.find_first_not_of(" \t\r\n"));
        token.erase(token.find_last_not_of(" \t\r\n") + 1);
        if (!token.empty()) {
            features.push_back(std::stod(token));
        }
    }
    return features;
}

static std::string extract_json_string_field(const std::string& body, const std::string& field_name) {
    std::string target = "\"" + field_name + "\"";
    size_t pos = body.find(target);
    if (pos == std::string::npos) return "";
    size_t colon = body.find(':', pos);
    if (colon == std::string::npos) return "";
    size_t first_quote = body.find('"', colon);
    if (first_quote == std::string::npos) return "";
    size_t second_quote = body.find('"', first_quote + 1);
    if (second_quote == std::string::npos) return "";
    return body.substr(first_quote + 1, second_quote - first_quote - 1);
}

void HttpServer::setup_default_routes() {
    // 1. Root dashboard UI
    route("GET", "/", [this](const HttpRequest&) {
        if (!static_html_.empty()) {
            return HttpResponse::html(200, static_html_);
        }
        return HttpResponse::text(200, "VeloceML Embedded C++ Inference Engine Active.");
    });

    // 2. Health Check
    route("GET", "/health", [this](const HttpRequest&) {
        auto now = std::chrono::steady_clock::now();
        double uptime_sec = std::chrono::duration<double>(now - start_time_).count();

        std::ostringstream json;
        json << "{\n"
             << "  \"status\": \"healthy\",\n"
             << "  \"engine\": \"VeloceML C++20 Core\",\n"
             << "  \"uptime_seconds\": " << std::fixed << std::setprecision(1) << uptime_sec << ",\n"
             << "  \"total_inferences\": " << total_requests_.load() << ",\n"
             << "  \"fraud_detected\": " << fraud_detected_.load() << ",\n"
             << "  \"models_registered\": [";
        bool first = true;
        for (const auto& [name, _] : models_) {
            if (!first) json << ", ";
            json << "\"" << name << "\"";
            first = false;
        }
        json << "]\n}";
        return HttpResponse::json(200, json.str());
    });

    // Alias for /api/v1/health
    route("GET", "/api/v1/health", routes_["GET:/health"]);

    // 3. Prometheus / OpenMetrics Endpoint
    route("GET", "/metrics", [this](const HttpRequest&) {
        size_t total = total_requests_.load();
        double total_lat = total_latency_us_.load();
        double avg_lat = total > 0 ? (total_lat / static_cast<double>(total)) : 0.0;

        std::ostringstream prom;
        prom << "# HELP veloce_inferences_total Total transactions evaluated\n"
             << "# TYPE veloce_inferences_total counter\n"
             << "veloce_inferences_total " << total << "\n"
             << "# HELP veloce_fraud_flagged_total Fraudulent transactions detected\n"
             << "# TYPE veloce_fraud_flagged_total counter\n"
             << "veloce_fraud_flagged_total " << fraud_detected_.load() << "\n"
             << "# HELP veloce_avg_latency_microseconds Average inference latency\n"
             << "# TYPE veloce_avg_latency_microseconds gauge\n"
             << "veloce_avg_latency_microseconds " << std::fixed << std::setprecision(3) << avg_lat << "\n";
        return HttpResponse::text(200, prom.str());
    });

    // 4. REST Inference Endpoint: POST /api/v1/predict
    route("POST", "/api/v1/predict", [this](const HttpRequest& req) {
        std::string model_name = extract_json_string_field(req.body, "model");
        if (model_name.empty()) model_name = "logistic"; // default

        auto it = models_.find(model_name);
        if (it == models_.end()) {
            return HttpResponse::json(400, "{\"error\": \"Model not found: " + model_name + "\"}");
        }

        std::vector<double> raw_feats = extract_json_features(req.body);
        if (raw_feats.empty()) {
            return HttpResponse::json(400, "{\"error\": \"Missing or invalid 'features' array\"}");
        }

        Matrix input(1, raw_feats.size(), raw_feats);
        Matrix scaled_input = scaler_ ? scaler_->transform(input) : input;

        auto t0 = std::chrono::high_resolution_clock::now();
        Matrix prob_mat = it->second->predict_proba(scaled_input);
        Matrix pred_mat = it->second->predict(scaled_input);
        auto t1 = std::chrono::high_resolution_clock::now();

        double elapsed_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        double prob = prob_mat(0, 0);
        bool is_fraud = (pred_mat(0, 0) >= 0.5);

        record_request(is_fraud, elapsed_us);

        std::string risk_tier = (prob >= 0.85) ? "CRITICAL" :
                                (prob >= 0.50) ? "HIGH" :
                                (prob >= 0.20) ? "MODERATE" : "LOW";

        std::ostringstream json;
        json << std::fixed << std::setprecision(4);
        json << "{\n"
             << "  \"model\": \"" << model_name << "\",\n"
             << "  \"is_fraud\": " << (is_fraud ? "true" : "false") << ",\n"
             << "  \"fraud_probability\": " << prob << ",\n"
             << "  \"risk_tier\": \"" << risk_tier << "\",\n"
             << "  \"latency_microseconds\": " << elapsed_us << ",\n"
             << "  \"features_received\": " << raw_feats.size() << "\n"
             << "}";

        return HttpResponse::json(200, json.str());
    });
}

} // namespace veloce
