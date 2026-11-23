#pragma once

#include "mlengine/math/Matrix.hpp"
#include <string>
#include <memory>

namespace veloce {

class IModel {
public:
    virtual ~IModel() = default;

    // Train the model on feature matrix X and target labels y
    virtual void fit(const Matrix& X, const Matrix& y) = 0;

    // Predict discrete binary or multi-class labels (0 or 1)
    [[nodiscard]] virtual Matrix predict(const Matrix& X) const = 0;

    // Predict continuous probability scores in range [0, 1]
    [[nodiscard]] virtual Matrix predict_proba(const Matrix& X) const = 0;

    // Serialize model weights and parameters to disk
    virtual void save(const std::string& filepath) const = 0;

    // Load model weights and parameters from disk
    virtual void load(const std::string& filepath) = 0;

    // Get model identifier name
    [[nodiscard]] virtual std::string name() const = 0;
};

} // namespace veloce
