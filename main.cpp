#include <iostream>
#include "utec/nn/neural_network.h"
#include "utec/nn/dense.h"
#include "utec/nn/activation.h"
#include "utec/agent/PongAgent.h"
#include "utec/agent/EnvGym.h"

using namespace utec::algebra;
using namespace utec::nn;

int main() {
    using T = float;

    // Dataset XOR
    Tensor<T, 2> X(std::array<size_t, 2>{4, 2});
    X(0, 0) = 0; X(0, 1) = 0;
    X(1, 0) = 0; X(1, 1) = 1;
    X(2, 0) = 1; X(2, 1) = 0;
    X(3, 0) = 1; X(3, 1) = 1;

    Tensor<T, 2> Y(std::array<size_t, 2>{4, 1});
    Y(0, 0) = 0;
    Y(1, 0) = 1;
    Y(2, 0) = 1;
    Y(3, 0) = 0;

    // Red neuronal
    NeuralNetwork<T> net;
    net.add_layer(std::make_unique<Dense<T>>(2, 4));
    net.add_layer(std::make_unique<ReLU<T>>());
    net.add_layer(std::make_unique<Dense<T>>(4, 1));

    // Entrenamiento
    T final_loss = net.train(X, Y, 1000, 0.1f);
    std::cout << "Final loss: " << final_loss << "\n";

    // Predicción
    auto pred = net.forward(X);
    const auto& shape = pred.shape();

    std::cout << "Pred shape: {"
              << shape[0] << ", " << shape[1] << "}\n";

    if (shape[1] == 1) {
        for (size_t i = 0; i < shape[0]; ++i) {
            std::cout << "Entrada: (" << X(i, 0) << ", " << X(i, 1) << ") -> "
                      << "Predicción: " << pred(i, 0) << "\n";
        }
    } else {
        std::cout << "Error: pred no tiene forma [n,1].\n";
    }
    // Crear agente Pong con el modelo entrenado
    auto agent = PongAgent<T>(std::make_unique<Dense<T>>(3, 1));

    // Estado de ejemplo: bola arriba de la paleta
    State test_state{0.5f, 0.8f, 0.3f};

    // Ejecutar acción
    int action = agent.act(test_state);
    std::cout << "Acción tomada por el agente: " << action << "\n";

    // Simular un paso
    EnvGym env;
    float reward;
    bool done;

    State s0 = env.reset();
    int a0 = agent.act(s0);
    State s1 = env.step(a0, reward, done);
    std::cout << "Después de un paso -> Bola en: (" << s1.ball_x << ", " << s1.ball_y << ")"
              << " | Paleta: " << s1.paddle_y
              << " | Recompensa: " << reward
              << " | Terminado: " << done << "\n";

    return 0;
}