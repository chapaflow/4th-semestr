import numpy as np

# Параметры функции
target_value = 30
a_coefficient = 2
b_coefficient = 3


# Функция, которую нужно оптимизировать
def objective_function(x):
    """
    Функция, которую необходимо минимизировать.
    """
    return (
        target_value
        - (x[0] - a_coefficient) * np.exp(-(x[0] - a_coefficient))
        - (x[1] - b_coefficient) * np.exp(-(x[1] - b_coefficient))
    )


def optimize_rosenbrock(x, alpha, epsilon):
    """
    Метод оптимизации Розенброка.
    """
    while True:
        x_prev = x.copy()

        # Поиск вдоль оси x
        while True:
            f_prev = objective_function(x)
            x[0] -= alpha * (
                (
                    2
                    * (x[0] - a_coefficient)
                    * np.exp(-(x[0] - a_coefficient))
                    * (1 + (x[1] - a_coefficient))
                )
                + ((x[1] - b_coefficient) * np.exp(-(x[0] - b_coefficient)))
            )
            if abs(objective_function(x) - f_prev) < epsilon:
                break

        # Поиск вдоль оси y
        while True:
            f_prev = objective_function(x)
            x[1] -= alpha * (x[1] - b_coefficient) * np.exp(-(x[0] - b_coefficient))
            if abs(objective_function(x) - f_prev) < epsilon:
                break

        # Проверка критерия остановки
        if np.linalg.norm(x - x_prev) < epsilon:
            break

    return x


def main():
    # Шаг оптимизации
    alpha = 0.001

    # Критерий остановки
    epsilon = 1e-6

    # Начальное приближение
    initial_guess = np.array([0.0, 0.0])

    # Оптимизация методом Розенброка
    optimal_solution = optimize_rosenbrock(initial_guess, alpha, epsilon)

    print("Оптимальное значение x:", optimal_solution[0])
    print("Оптимальное значение y:", optimal_solution[1])


if __name__ == "__main__":
    main()
