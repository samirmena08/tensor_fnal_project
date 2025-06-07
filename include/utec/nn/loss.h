//
// Created by esaua on 7/06/2025.
//

#ifndef PROG3_TENSOR_FINAL_PROJECT_V2025_01_LOSS_H
#define PROG3_TENSOR_FINAL_PROJECT_V2025_01_LOSS_H

#include "utec/algebra/tensor.h"

namespace utec::nn {

    template <typename T>
    class MSELoss {
    private:
        utec::algebra::Tensor<T, 2> last_pred;
        utec::algebra::Tensor<T, 2> last_target;

    public:
        // Guarda predicciones y valores objetivo
        T forward(const utec::algebra::Tensor<T, 2>& pred, const utec::algebra::Tensor<T, 2>& target) {
            last_pred = pred;
            last_target = target;

            T sum = static_cast<T>(0);
            for (size_t i = 0; i < pred.size(); ++i) {
                T diff = pred.data()[i] - target.data()[i];
                sum += diff * diff;
            }

            return sum / static_cast<T>(pred.size());
        }

        // Derivada: dL/dpred = 2/N * (pred - target)
        utec::algebra::Tensor<T, 2> backward() {
            utec::algebra::Tensor<T, 2> grad(last_pred.shape());
            T scale = static_cast<T>(2) / static_cast<T>(last_pred.size());

            for (size_t i = 0; i < grad.size(); ++i)
                grad.data()[i] = scale * (last_pred.data()[i] - last_target.data()[i]);

            return grad;
        }
    };

} // namespace utec::nn

#endif //PROG3_TENSOR_FINAL_PROJECT_V2025_01_LOSS_H
