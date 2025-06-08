//
// Created by Intel on 8/06/2025.
//

#ifndef PONG_AGENT_H
#define PONG_AGENT_H

#include "utec/nn/layer.h"
#include "utec/nn/neural_network.h"
#include "utec/algebra/tensor.h"
#include <memory>
#include <cmath>

namespace utec::nn {

    struct State {
        float ball_x, ball_y;
        float paddle_y;
    };

    template <typename T>
    class PongAgent {
        std::unique_ptr<ILayer<T>> model;

    public:
        PongAgent(std::unique_ptr<ILayer<T>> m) : model(std::move(m)) {}

        int act(const State& s) {
            typename utec::algebra::Tensor<T, 2> input(std::array<size_t, 2>{1, 3});
            input(0, 0) = s.ball_x;
            input(0, 1) = s.ball_y;
            input(0, 2) = s.paddle_y;

            typename utec::algebra::Tensor<T, 2> output = model->forward(input);
            T val = output(0, 0);

            if (val < -0.33) return -1;
            if (val > 0.33) return +1;
            return 0;
        }


    };

} // namespace utec::nn

#endif // PONG_AGENT_H
