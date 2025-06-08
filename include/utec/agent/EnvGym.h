#ifndef ENV_GYM_H
#define ENV_GYM_H

#include <random>
#include "PongAgent.h"

namespace utec::nn {

    class EnvGym {
        float ball_x{}, ball_y{};
        float paddle_y{};
        float ball_vx{}, ball_vy{};
        std::default_random_engine rng;

    public:
        EnvGym() : rng(std::random_device{}()) {}

        State reset() {
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            ball_x = 0.5f;
            ball_y = 0.5f;
            paddle_y = 0.5f;
            ball_vx = (dist(rng) > 0.5f) ? 0.03f : -0.03f;
            ball_vy = (dist(rng) - 0.5f) * 0.06f;
            return {ball_x, ball_y, paddle_y};
        }

        State step(int action, float& reward, bool& done) {
            // Mover la paleta
            paddle_y += action * 0.04f;
            paddle_y = std::max(0.0f, std::min(1.0f, paddle_y));

            // Mover la bola
            ball_x += ball_vx;
            ball_y += ball_vy;

            // Rebote en techo o suelo
            if (ball_y < 0.0f || ball_y > 1.0f) ball_vy *= -1;

            done = false;
            reward = 0.0f;

            // Gol o pérdida
            if (ball_x < 0.0f) {
                done = true;
                reward = -1.0f; // falló
            } else if (ball_x > 1.0f) {
                // ¿La paleta estaba en la posición correcta?
                if (std::abs(paddle_y - ball_y) < 0.1f) {
                    reward = +1.0f; // éxito
                } else {
                    reward = -1.0f; // falló
                }
                done = true;
            }

            return {ball_x, ball_y, paddle_y};
        }
    };

} // namespace utec::nn

#endif // ENV_GYM_H
