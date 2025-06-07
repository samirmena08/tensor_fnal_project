//
// Created by esaua on 7/06/2025.
//

#ifndef PROG3_TENSOR_FINAL_PROJECT_V2025_01_ACTIVATION_H
#define PROG3_TENSOR_FINAL_PROJECT_V2025_01_ACTIVATION_H

#include "layer.h"
#include "utec/algebra/tensor.h"

namespace utec::nn {

    template <typename T>
    class ReLU : public ILayer<T> {
    private:
        utec::algebra::Tensor<T, 2> mask;  // 1 donde x > 0, 0 en otro caso

    public:
        utec::algebra::Tensor<T, 2> forward(const utec::algebra::Tensor<T, 2>& x) override {
            mask = utec::algebra::Tensor<T, 2>(x.shape());
            utec::algebra::Tensor<T, 2> result(x.shape());

            for (size_t i = 0; i < x.size(); ++i) {
                if (x.data()[i] > static_cast<T>(0)) {
                    result.data()[i] = x.data()[i];
                    mask.data()[i] = static_cast<T>(1);
                } else {
                    result.data()[i] = static_cast<T>(0);
                    mask.data()[i] = static_cast<T>(0);
                }
            }

            return result;
        }

        utec::algebra::Tensor<T, 2> backward(const utec::algebra::Tensor<T, 2>& grad) override {
            utec::algebra::Tensor<T, 2> result(grad.shape());

            for (size_t i = 0; i < grad.size(); ++i)
                result.data()[i] = grad.data()[i] * mask.data()[i];

            return result;
        }
    };

} // namespace utec::nn

#endif //PROG3_TENSOR_FINAL_PROJECT_V2025_01_ACTIVATION_H
