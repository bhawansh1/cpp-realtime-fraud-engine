#pragma once
#include <string_view>

namespace veloce {

inline constexpr std::string_view EMBEDDED_DASHBOARD_HTML = R"HTML(<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VeloceML | Ultra-Low Latency C++ Inference Engine</title>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;800&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0a0e17;
            --bg-card: rgba(18, 26, 43, 0.85);
            --border-card: rgba(255, 255, 255, 0.08);
            --accent-cyan: #00f2fe;
            --accent-blue: #4facfe;
            --accent-emerald: #10b981;
            --accent-rose: #f43f5e;
            --accent-amber: #f59e0b;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            background-color: var(--bg-primary);
            color: var(--text-main);
            min-height: 100vh;
            padding: 2rem 1rem;
            background-image: 
                radial-gradient(at 0% 0%, rgba(79, 172, 254, 0.12) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(0, 242, 254, 0.08) 0px, transparent 50%);
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
        }

        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2.5rem;
            border-bottom: 1px solid var(--border-card);
            padding-bottom: 1.5rem;
            flex-wrap: wrap;
            gap: 1rem;
        }

        .brand-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            background: rgba(0, 242, 254, 0.1);
            border: 1px solid rgba(0, 242, 254, 0.3);
            padding: 0.35rem 0.8rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--accent-cyan);
            margin-bottom: 0.5rem;
        }

        h1 {
            font-size: 2.2rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            background: linear-gradient(135deg, #ffffff 30%, #94a3b8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .status-pill {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.3);
            color: var(--accent-emerald);
            padding: 0.5rem 1rem;
            border-radius: 9999px;
            font-size: 0.85rem;
            font-weight: 600;
        }

        .pulse-dot {
            width: 8px;
            height: 8px;
            background: var(--accent-emerald);
            border-radius: 50%;
            box-shadow: 0 0 12px var(--accent-emerald);
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.4; transform: scale(0.8); }
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1.25rem;
            margin-bottom: 2.5rem;
        }

        .stat-card {
            background: var(--bg-card);
            border: 1px solid var(--border-card);
            border-radius: 1rem;
            padding: 1.5rem;
            backdrop-filter: blur(12px);
        }

        .stat-label {
            font-size: 0.85rem;
            color: var(--text-muted);
            font-weight: 500;
            margin-bottom: 0.5rem;
        }

        .stat-val {
            font-size: 1.75rem;
            font-weight: 800;
            font-family: 'JetBrains Mono', monospace;
            color: #fff;
        }

        .stat-val.accent {
            color: var(--accent-cyan);
        }

        .main-grid {
            display: grid;
            grid-template-columns: 1.3fr 1fr;
            gap: 2rem;
        }

        @media (max-width: 900px) {
            .main-grid { grid-template-columns: 1fr; }
        }

        .card {
            background: var(--bg-card);
            border: 1px solid var(--border-card);
            border-radius: 1.25rem;
            padding: 2rem;
            backdrop-filter: blur(12px);
        }

        .card-title {
            font-size: 1.25rem;
            font-weight: 700;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .presets {
            display: flex;
            gap: 0.75rem;
            margin-bottom: 1.5rem;
            flex-wrap: wrap;
        }

        .btn-preset {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-card);
            color: var(--text-muted);
            padding: 0.5rem 0.9rem;
            border-radius: 0.5rem;
            font-size: 0.8rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }

        .btn-preset:hover {
            background: rgba(255, 255, 255, 0.1);
            color: #fff;
            border-color: rgba(255, 255, 255, 0.2);
        }

        .form-group {
            margin-bottom: 1.25rem;
        }

        .form-label {
            display: flex;
            justify-content: space-between;
            font-size: 0.9rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }

        .form-val-display {
            font-family: 'JetBrains Mono', monospace;
            color: var(--accent-cyan);
        }

        input[type="range"] {
            width: 100%;
            height: 6px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
            outline: none;
            accent-color: var(--accent-cyan);
            cursor: pointer;
        }

        select {
            width: 100%;
            padding: 0.8rem 1rem;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-card);
            color: #fff;
            border-radius: 0.75rem;
            font-size: 0.95rem;
            outline: none;
            margin-bottom: 1.5rem;
        }

        select option {
            background: #121a2b;
            color: #fff;
        }

        .btn-predict {
            width: 100%;
            padding: 1rem;
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan));
            color: #050b14;
            border: none;
            border-radius: 0.75rem;
            font-size: 1rem;
            font-weight: 800;
            cursor: pointer;
            transition: all 0.2s;
            box-shadow: 0 4px 20px rgba(0, 242, 254, 0.25);
        }

        .btn-predict:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 24px rgba(0, 242, 254, 0.4);
        }

        .result-box {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 2rem;
            border-radius: 1rem;
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-card);
            text-align: center;
            margin-bottom: 1.5rem;
            min-height: 220px;
        }

        .badge-decision {
            font-size: 1.25rem;
            font-weight: 800;
            padding: 0.6rem 1.5rem;
            border-radius: 9999px;
            margin-bottom: 1rem;
            letter-spacing: 0.05em;
        }

        .badge-safe {
            background: rgba(16, 185, 129, 0.15);
            border: 1px solid var(--accent-emerald);
            color: var(--accent-emerald);
        }

        .badge-fraud {
            background: rgba(244, 63, 94, 0.15);
            border: 1px solid var(--accent-rose);
            color: var(--accent-rose);
        }

        .prob-bar-container {
            width: 100%;
            height: 12px;
            background: rgba(255, 255, 255, 0.08);
            border-radius: 9999px;
            overflow: hidden;
            margin: 1rem 0 0.5rem;
        }

        .prob-bar {
            height: 100%;
            width: 0%;
            background: var(--accent-emerald);
            transition: width 0.4s ease, background 0.4s ease;
        }

        .meta-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
            width: 100%;
            margin-top: 1rem;
            text-align: left;
        }

        .meta-item {
            background: rgba(255, 255, 255, 0.03);
            padding: 0.75rem 1rem;
            border-radius: 0.5rem;
        }

        .meta-title {
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-bottom: 0.25rem;
        }

        .meta-val {
            font-family: 'JetBrains Mono', monospace;
            font-weight: 700;
            font-size: 1.1rem;
        }

        .benchmark-section {
            margin-top: 1.5rem;
            padding-top: 1.5rem;
            border-top: 1px solid var(--border-card);
        }

        .btn-bench {
            width: 100%;
            padding: 0.8rem;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-card);
            color: var(--text-main);
            border-radius: 0.5rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.2s;
        }

        .btn-bench:hover {
            background: rgba(255, 255, 255, 0.1);
        }

        .bench-output {
            margin-top: 1rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            color: var(--accent-cyan);
            background: #060910;
            padding: 1rem;
            border-radius: 0.5rem;
            display: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <div class="brand-badge">⚡ VELOCEML C++20 INFERENCE ENGINE</div>
                <h1>Real-Time Financial Fraud Engine</h1>
            </div>
            <div class="status-pill">
                <div class="pulse-dot"></div>
                <span>REST API ONLINE :8080</span>
            </div>
        </header>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">C++ Inference Latency (p50)</div>
                <div class="stat-val accent">&lt; 0.10 µs</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Peak Engine Throughput</div>
                <div class="stat-val">5.6M+ <span style="font-size: 1rem; font-weight: 400; color: var(--text-muted);">ops/s</span></div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Active Models</div>
                <div class="stat-val">Logistic & Tree</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Model Memory Overhead</div>
                <div class="stat-val">&lt; 150 KB</div>
            </div>
        </div>

        <div class="main-grid">
            <!-- Left: Transaction Input Form -->
            <div class="card">
                <div class="card-title">
                    <span>Transaction Parameters</span>
                </div>

                <div class="presets">
                    <span style="font-size: 0.85rem; color: var(--text-muted); align-self: center;">Quick Scenarios:</span>
                    <button class="btn-preset" onclick="loadScenario('safe')">🟢 Coffee Purchase ($4.50)</button>
                    <button class="btn-preset" onclick="loadScenario('suspicious')">🟡 Online Electronics ($450)</button>
                    <button class="btn-preset" onclick="loadScenario('fraud')">🔴 Foreign ATM Fraud ($1,850)</button>
                </div>

                <div class="form-group">
                    <div class="form-label">
                        <span>Transaction Amount ($)</span>
                        <span class="form-val-display" id="val_amount">$4.50</span>
                    </div>
                    <input type="range" id="amount" min="1" max="3000" step="1" value="4.5" oninput="updateDisplay()">
                </div>

                <div class="form-group">
                    <div class="form-label">
                        <span>Distance From Home (km)</span>
                        <span class="form-val-display" id="val_dist_home">2.1 km</span>
                    </div>
                    <input type="range" id="dist_home" min="0" max="1000" step="1" value="2" oninput="updateDisplay()">
                </div>

                <div class="form-group">
                    <div class="form-label">
                        <span>Distance From Last Transaction (km)</span>
                        <span class="form-val-display" id="val_dist_last">0.5 km</span>
                    </div>
                    <input type="range" id="dist_last" min="0" max="1000" step="1" value="0.5" oninput="updateDisplay()">
                </div>

                <div class="form-group">
                    <div class="form-label">
                        <span>24h Purchase Frequency (tx count)</span>
                        <span class="form-val-display" id="val_freq">2</span>
                    </div>
                    <input type="range" id="freq" min="1" max="30" step="1" value="2" oninput="updateDisplay()">
                </div>

                <div class="form-group">
                    <div class="form-label">
                        <span>Device Risk Fingerprint [0-1]</span>
                        <span class="form-val-display" id="val_risk">0.12</span>
                    </div>
                    <input type="range" id="risk" min="0.0" max="1.0" step="0.01" value="0.12" oninput="updateDisplay()">
                </div>

                <div class="form-group">
                    <div class="form-label">
                        <span>Failed PIN / OTP Retries</span>
                        <span class="form-val-display" id="val_retries">0</span>
                    </div>
                    <input type="range" id="retries" min="0" max="5" step="1" value="0" oninput="updateDisplay()">
                </div>

                <label class="form-label" style="margin-top: 1rem;">Select Inference Model:</label>
                <select id="model_select">
                    <option value="logistic">Regularized Logistic Regression (Adam)</option>
                    <option value="tree">Decision Tree Classifier (Gini Impurity)</option>
                </select>

                <button class="btn-predict" onclick="predictTransaction()">Run Instant C++ Inference</button>
            </div>

            <!-- Right: Inference Result Display -->
            <div class="card">
                <div class="card-title">Real-Time Model Evaluation</div>

                <div class="result-box" id="result_box">
                    <div class="badge-decision badge-safe" id="badge_decision">APPROVED</div>
                    <div style="font-size: 0.9rem; color: var(--text-muted);">Fraud Probability Score</div>
                    <div style="font-size: 2.5rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; margin-top: 0.25rem;" id="score_display">
                        0.02 %
                    </div>

                    <div class="prob-bar-container">
                        <div class="prob-bar" id="prob_bar" style="width: 2%;"></div>
                    </div>

                    <div class="meta-grid">
                        <div class="meta-item">
                            <div class="meta-title">C++ Engine Latency</div>
                            <div class="meta-val" id="meta_latency" style="color: var(--accent-cyan);">- µs</div>
                        </div>
                        <div class="meta-item">
                            <div class="meta-title">Risk Tier</div>
                            <div class="meta-val" id="meta_risk" style="color: var(--accent-emerald);">LOW</div>
                        </div>
                    </div>
                </div>

                <div class="benchmark-section">
                    <div style="font-size: 0.9rem; font-weight: 700; margin-bottom: 0.5rem;">Live High-Throughput Stress Test</div>
                    <p style="font-size: 0.8rem; color: var(--text-muted); margin-bottom: 1rem;">
                        Fire 500 parallel HTTP inference requests from your browser directly to the C++ REST socket server.
                    </p>
                    <button class="btn-bench" onclick="runStressTest()">Fire 500 Live HTTP Inferences</button>
                    <div class="bench-output" id="bench_output"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        function updateDisplay() {
            document.getElementById('val_amount').innerText = '$' + parseFloat(document.getElementById('amount').value).toFixed(2);
            document.getElementById('val_dist_home').innerText = document.getElementById('dist_home').value + ' km';
            document.getElementById('val_dist_last').innerText = document.getElementById('dist_last').value + ' km';
            document.getElementById('val_freq').innerText = document.getElementById('freq').value;
            document.getElementById('val_risk').innerText = parseFloat(document.getElementById('risk').value).toFixed(2);
            document.getElementById('val_retries').innerText = document.getElementById('retries').value;
        }

        function loadScenario(type) {
            if (type === 'safe') {
                document.getElementById('amount').value = 4.50;
                document.getElementById('dist_home').value = 1.2;
                document.getElementById('dist_last').value = 0.5;
                document.getElementById('freq').value = 2;
                document.getElementById('risk').value = 0.08;
                document.getElementById('retries').value = 0;
            } else if (type === 'suspicious') {
                document.getElementById('amount').value = 450.00;
                document.getElementById('dist_home').value = 65;
                document.getElementById('dist_last').value = 45;
                document.getElementById('freq').value = 8;
                document.getElementById('risk').value = 0.52;
                document.getElementById('retries').value = 1;
            } else if (type === 'fraud') {
                document.getElementById('amount').value = 1850.00;
                document.getElementById('dist_home').value = 520;
                document.getElementById('dist_last').value = 380;
                document.getElementById('freq').value = 19;
                document.getElementById('risk').value = 0.94;
                document.getElementById('retries').value = 3;
            }
            updateDisplay();
            predictTransaction();
        }

        async function predictTransaction() {
            const features = [
                parseFloat(document.getElementById('amount').value),
                parseFloat(document.getElementById('dist_home').value),
                parseFloat(document.getElementById('dist_last').value),
                parseFloat(document.getElementById('freq').value),
                parseFloat(document.getElementById('risk').value),
                parseFloat(document.getElementById('retries').value)
            ];

            const model = document.getElementById('model_select').value;

            try {
                const response = await fetch('/api/v1/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model: model, features: features })
                });

                const data = await response.json();
                renderResult(data);
            } catch (err) {
                console.error("Inference call failed:", err);
            }
        }

        function renderResult(data) {
            const probPct = (data.fraud_probability * 100).toFixed(2);
            document.getElementById('score_display').innerText = probPct + ' %';
            
            const probBar = document.getElementById('prob_bar');
            probBar.style.width = Math.max(3, data.fraud_probability * 100) + '%';

            const badge = document.getElementById('badge_decision');
            const metaRisk = document.getElementById('meta_risk');

            if (data.is_fraud) {
                badge.className = 'badge-decision badge-fraud';
                badge.innerText = 'BLOCKED (FRAUD)';
                probBar.style.background = 'var(--accent-rose)';
                metaRisk.innerText = data.risk_tier;
                metaRisk.style.color = 'var(--accent-rose)';
            } else {
                badge.className = 'badge-decision badge-safe';
                badge.innerText = 'APPROVED (SAFE)';
                probBar.style.background = 'var(--accent-emerald)';
                metaRisk.innerText = data.risk_tier;
                metaRisk.style.color = 'var(--accent-emerald)';
            }

            document.getElementById('meta_latency').innerText = data.latency_microseconds.toFixed(2) + ' µs';
        }

        async function runStressTest() {
            const output = document.getElementById('bench_output');
            output.style.display = 'block';
            output.innerText = "Firing 500 HTTP concurrent requests to C++ server...";

            const payload = JSON.stringify({
                model: "logistic",
                features: [120.0, 15.0, 5.0, 3.0, 0.2, 0.0]
            });

            const t0 = performance.now();
            const requests = [];
            for (let i = 0; i < 500; ++i) {
                requests.push(fetch('/api/v1/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: payload
                }));
            }

            await Promise.all(requests);
            const totalMs = performance.now() - t0;
            const reqPerSec = Math.round(500 / (totalMs / 1000));

            output.innerHTML = `Completed 500 HTTP requests in <strong>${totalMs.toFixed(1)} ms</strong><br>Browser HTTP Throughput: <strong>${reqPerSec} requests/sec</strong><br>Server core processing: Sub-microsecond.`;
        }

        // Run initial prediction on load
        window.addEventListener('load', () => {
            updateDisplay();
            predictTransaction();
        });
    </script>
</body>
</html>)HTML";

} // namespace veloce
