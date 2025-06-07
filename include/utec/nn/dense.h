//
// Created by esaua on 7/06/2025.
//

#ifndef PROG3_TENSOR_FINAL_PROJECT_V2025_01_DENSE_H
#define PROG3_TENSOR_FINAL_PROJECT_V2025_01_DENSE_H

#include "layer.h"
#include "utec/algebra/tensor.h"

namespace utec::nn {

    template <typename T>
    class Dense : public ILayer<T> {
    private:
        utec::algebra::Tensor<T, 2> W, dW;  // pesos y su gradiente
        utec::algebra::Tensor<T, 1> b, db;  // bias y su gradiente
        utec::algebra::Tensor<T, 2> last_input;  // cache para backward

    public:
        Dense(size_t in_feats, size_t out_feats)
                : W(std::array<size_t, 2>{in_feats, out_feats}),
                  dW(std::array<size_t, 2>{in_feats, out_feats}),
                  b(std::array<size_t, 1>{out_feats}),
                  db(std::array<size_t, 1>{out_feats})
        {
            // Inicializa pesos pequeños
            for (auto& val : W.data())
                val = static_cast<T>(0.01);

            b.fill(static_cast<T>(0));
        }

        utec::algebra::Tensor<T, 2> forward(const utec::algebra::Tensor<T, 2>& x) override {
            last_input = x;
            auto out = utec::algebra::matrix_product(x, W);
            for (size_t i = 0; i < out.shape()[0]; ++i)
                for (size_t j = 0; j < out.shape()[1]; ++j)
                    out(i, j) += b(j);
            return out;
        }

        utec::algebra::Tensor<T, 2> backward(const utec::algebra::Tensor<T, 2>& grad) override {
            auto x_T = utec::algebra::transpose_2d(last_input);
            dW = utec::algebra::matrix_product(x_T, grad);

            db.fill(static_cast<T>(0));
            for (size_t i = 0; i < grad.shape()[0]; ++i)
                for (size_t j = 0; j < grad.shape()[1]; ++j)
                    db(j) += grad(i, j);

            auto W_T = utec::algebra::transpose_2d(W);
            return utec::algebra::matrix_product(grad, W_T);
        }

        // Métodos auxiliares para optimización
        utec::algebra::Tensor<T, 2>& weights() { return W; }
        utec::algebra::Tensor<T, 2>& grad_weights() { return dW; }
        utec::algebra::Tensor<T, 1>& bias() { return b; }
        utec::algebra::Tensor<T, 1>& grad_bias() { return db; }
    };

} // namespace utec::nn

#endif //PROG3_TENSOR_FINAL_PROJECT_V2025_01_DENSE_H
