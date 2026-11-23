# =========================================================================
# Stage 1: Build the C++20 VeloceML Engine
# =========================================================================
FROM alpine:3.20 AS builder

RUN apk add --no-cache \
    build-base \
    cmake \
    git \
    linux-headers

WORKDIR /app

# Copy CMakeLists and source code
COPY CMakeLists.txt ./
COPY include/ ./include/
COPY src/ ./src/
COPY tests/ ./tests/
COPY benchmarks/ ./benchmarks/
COPY web/ ./web/

# Build optimized Release binaries
RUN mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release .. && \
    make -j$(nproc) veloce_ml veloce_train veloce_server veloce_tests veloce_benchmark && \
    ./veloce_tests

# =========================================================================
# Stage 2: Ultra-Minimal Production Runtime Image (< 25 MB)
# =========================================================================
FROM alpine:3.20 AS runtime

RUN apk add --no-cache libstdc++ libgcc

WORKDIR /app

# Copy compiled binaries from builder
COPY --from=builder /app/build/veloce_server /app/veloce_server
COPY --from=builder /app/build/veloce_train /app/veloce_train
COPY --from=builder /app/build/veloce_benchmark /app/veloce_benchmark

# Copy static frontend web dashboard
COPY web/ /app/web/

# Create data and models directory
RUN mkdir -p /app/data /app/models

ENV PORT=8080
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget -qO- http://127.0.0.1:${PORT}/health || exit 1

ENTRYPOINT ["/app/veloce_server"]
CMD ["8080"]
