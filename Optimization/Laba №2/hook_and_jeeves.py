import numpy as np

target_value = 30
a_coefficient = 2
b_coefficient = 3


def objective_function(x):
    """
    Функция, которую необходимо минимизировать.
    """
    return (
        target_value
        - (x[0] - a_coefficient) * np.exp(-(x[0] - a_coefficient))
        - (x[1] - b_coefficient) * np.exp(-(x[1] - b_coefficient))
    )


def main():
    initial_point = np.array([0.0, 0.0])
    step_size = 0.01
    convergence_threshold = 1e-6

    current_point = initial_point.copy()
    new_point = initial_point.copy()

    while True:
        # Исследовательский поиск
        for i in range(len(current_point)):
            next_point_plus = current_point.copy()
            next_point_plus[i] += step_size
            next_point_minus = current_point.copy()
            next_point_minus[i] -= step_size
            if objective_function(next_point_plus) < objective_function(current_point):
                new_point[i] = next_point_plus[i]
            elif objective_function(next_point_minus) < objective_function(
                current_point
            ):
                new_point[i] = next_point_minus[i]
        # Узловой поиск
        pattern_point = 2 * new_point - current_point
        if objective_function(pattern_point) < objective_function(current_point):
            current_point = pattern_point.copy()
        else:
            if np.linalg.norm(step_size) < convergence_threshold:
                break
            step_size /= 2
            new_point = current_point.copy()

    print(
        "Минимум функции найден в точке x = {:.2f} и y = {:.2f}".format(
            current_point[0], current_point[1]
        )
    )


if __name__ == "__main__":
    main()
