import numpy as np
from scipy.optimize import minimize_scalar

target_value = 30
coefficient_a = 2
coefficient_b = 3


def objective_function(x):
    """
    Функция, которую необходимо минимизировать.
    """
    return (
        target_value
        - (x[0] - coefficient_a) * np.exp(-(x[0] - coefficient_a))
        - (x[1] - coefficient_b) * np.exp(-(x[1] - coefficient_b))
    )


def gradient_function(x):
    """
    Градиент функции.
    """
    dfdx = (x[0] - coefficient_a) * np.exp(-(x[0] - coefficient_a)) + 1
    dfdy = (x[1] - coefficient_b) * np.exp(-(x[1] - coefficient_b)) + 1
    return np.array([dfdx, dfdy])


def main():
    # Начальное приближение
    initial_point = np.array([0.0, 0.0])

    # Точность сходимости
    convergence_threshold = 1e-6

    x = initial_point

    while True:
        # Градиент функции в текущей точке
        gradient = gradient_function(x)

        # Проверка условия сходимости
        if np.linalg.norm(gradient) < convergence_threshold:
            break

        # Направление спуска
        descent_direction = -gradient / np.linalg.norm(gradient)

        # Минимизация функции по направлению спуска
        res = minimize_scalar(
            lambda alpha: objective_function(x + alpha * descent_direction)
        )
        optimal_step_size = res.x

        # Обновление текущей точки
        x = x + optimal_step_size * descent_direction

        # Обновление порога сходимости (опционально)
        convergence_threshold = np.linalg.norm(gradient)

    # Вывод результата
    print("Минимум функции найден в точке x = {:.2f} и y = {:.2f}".format(x[0], x[1]))


if __name__ == "__main__":
    main()
