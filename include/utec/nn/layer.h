//
// Created by esaua on 7/06/2025.
//

#ifndef PROG3_TENSOR_FINAL_PROJECT_V2025_01_LAYER_H
#define PROG3_TENSOR_FINAL_PROJECT_V2025_01_LAYER_H

#include "utec/algebra/tensor.h"

namespace utec::nn {

    template <typename T>
    class ILayer {
    public:
        virtual ~ILayer() = default;

        // Forward: recibe un batch de datos (batch x input_feats)
        virtual utec::algebra::Tensor<T, 2> forward(const utec::algebra::Tensor<T, 2>& x) = 0;

        // Backward: recibe gradiente de salida (batch x output_feats) y devuelve gradiente de entrada
        virtual utec::algebra::Tensor<T, 2> backward(const utec::algebra::Tensor<T, 2>& grad) = 0;
    };

} // namespace utec::nn

#endif //TENSOR_FNAL_PROJECT_LAYER_H
