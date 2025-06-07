//
// Created by esaua on 7/06/2025.
//

#ifndef PROG3_TENSOR_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H
#define PROG3_TENSOR_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H

#include <vector>
#include <memory>
#include "dense.h"
#include "layer.h"
#include "loss.h"

namespace utec::nn {

    template <typename T>
    class NeuralNetwork {
    private:
        std::vector<std::unique_ptr<ILayer<T>>> layers;
        MSELoss<T> criterion;

    public:
        void add_layer(std::unique_ptr<ILayer<T>> layer) {
            layers.push_back(std::move(layer));
        }

        utec::algebra::Tensor<T, 2> forward(const utec::algebra::Tensor<T, 2>& x) {
            auto out = x;
            for (auto& layer : layers)
                out = layer->forward(out);
            return out;
        }

        void backward(const utec::algebra::Tensor<T, 2>& grad) {
            auto g = grad;
            for (auto it = layers.rbegin(); it != layers.rend(); ++it)
                g = (*it)->backward(g);
        }

        void optimize(T lr) {
            // Recorremos todas las capas y actualizamos si tienen pesos
            for (auto& layer : layers) {
                // Dense tiene métodos para acceder a sus parámetros
                auto* dense = dynamic_cast<Dense<T>*>(layer.get());
                if (dense) {
                    auto& W = dense->weights();
                    auto& dW = dense->grad_weights();
                    for (size_t i = 0; i < W.size(); ++i)
                        W.data()[i] -= lr * dW.data()[i];

                    auto& b = dense->bias();
                    auto& db = dense->grad_bias();
                    for (size_t i = 0; i < b.size(); ++i)
                        b.data()[i] -= lr * db.data()[i];
                }
            }
        }

        T train(const utec::algebra::Tensor<T, 2>& X, const utec::algebra::Tensor<T, 2>& Y,
                size_t epochs, T lr) {
            T loss = static_cast<T>(0);

            for (size_t e = 0; e < epochs; ++e) {
                auto pred = forward(X);
                loss = criterion.forward(pred, Y);
                auto grad = criterion.backward();
                backward(grad);
                optimize(lr);
            }

            return loss;
        }
    };

} // namespace utec::nn

#endif //PROG3_TENSOR_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H
