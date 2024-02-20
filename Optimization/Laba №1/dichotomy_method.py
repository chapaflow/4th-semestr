import argparse

def function(x):
    return x ** 3 - x

def main(a:int, b:int, e=1e-3) -> float:
    '''
    Получает:
    a - значение коэффициента a (тип int)
    b - значение коэффициента b (тип int)
    e - точность (тип float)

    Возваращает:
    result - значение функции в точках a и b
    '''
    e = 1e-3

    while (b-a) > 2*e:
        x = (a + b) / 2
        x_1, x_2 = x - (e / 2), x + (e / 2)
        f_1, f_2 = function(x_1), function(x_2)
        if f_1 > f_2:
            a = x_1
        else:
            b = x_2 

    result = function((a + b)/ 2)
    print(f"Значение функции: {result}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--a', help = 'Значение коэффициента a', type=int)
    parser.add_argument('--b', help = 'Значение коэффициента b', type=int)
    parser.add_argument('--e', help = 'Точность', type=float)
    args = parser.parse_args()
    main(args.a, args.b)
