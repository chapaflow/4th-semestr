import numpy as np
from scipy.optimize import minimize_scalar


def objective_function(x):
    """
    Функция, которую необходимо минимизировать.
    """
    target_value = 30
    coefficient_a = 2
    coefficient_b = 3
    return (
        target_value
        - (x[0] - coefficient_a) * np.exp(-(x[0] - coefficient_a))
        - (x[1] - coefficient_b) * np.exp(-(x[1] - coefficient_b))
    )


def main():
    # Точность сходимости
    convergence_threshold = 1e-6

    # Начальное приближение
    initial_approximation = np.array([0, 0])

    # Размерность пространства
    dimension = len(initial_approximation)

    # Инициализация переменных
    current_point = np.copy(initial_approximation)
    iteration_count = 1
    direction_index = 1

    # Основной цикл
    while True:
        # Одномерный поиск
        res = minimize_scalar(
            lambda lmbd: objective_function(
                current_point + lmbd * np.eye(dimension)[direction_index - 1]
            )
        )
        step_size = res.x

        # Обновление текущей точки
        current_point = (
            current_point + step_size * np.eye(dimension)[direction_index - 1]
        )

        # Обновление индекса направления
        if direction_index < dimension:
            direction_index += 1
        else:
            # Проверка условия сходимости
            if (
                np.linalg.norm(current_point - initial_approximation)
                < convergence_threshold
            ):
                break
            else:
                initial_approximation = current_point
                direction_index = 1
                iteration_count += 1

    print(
        "Минимум функции найден в точке x = {:.2f} и y = {:.2f}".format(
            current_point[0], current_point[1]
        )
    )


if __name__ == "__main__":
    main()
