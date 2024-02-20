import argparse

def function(x):
    return x ** 3 - x

def golden_section_search(a:int, b:int, e=1e-3) -> float:
    '''
    Получает:
    a - значение коэффициента a (тип int)
    b - значение коэффициента b (тип int)
    e - точность (тип float)
     
    Возваращает:
    result - значение функции в точках a и b
    '''
    # Константа
    t = 0.618

    while abs(b - a) > e:
        x_1, x_2 = a + (1 - t) * (b - a), a + t * (b - a)
        f_1, f_2 = function(x_1), function(x_2)

        if f_1 > f_2:
            a, f_1, x_1 = x_1, f_2, x_2
            x_2 = b - (b - a) * t
            f_2 = function(x_2)
        else:
            b, f_2, x_2 = x_2, f_1, x_1
            x_1 = a + (b - a) * t
            f_1 = function(x_1)

    result = (a + b) / 2

    return result

def main(a, b, e):
    optimal_point = golden_section_search(a, b, e)
    result = function(optimal_point)
    print("Минимум достигается при x = {:.5f} со значением функции = {:.5f}".format(result, function(result)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--a', help = 'Значение коэффициента a', type=int)
    parser.add_argument('--b', help = 'Значение коэффициента b', type=int)
    parser.add_argument('--e', help = 'Точность', type=float)
    args = parser.parse_args()
    main(args.a, args.b, args.e)
