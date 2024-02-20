import argparse

def function(x):
    return x ** 3 - x

def fibonacci_numbers(n):
    fib = [0, 1]
    for i in range(2, n + 1):
        fib.append(fib[-1] + fib[-2])
    return fib

def fibonacci_search(a, b, e):
    L = b - a
    n = 0  
    
    while fibonacci_numbers(n)[-1] < L / e:
        n += 1
    fib = fibonacci_numbers(n)  

    for i in range(n, 2, -1):
        x1 = a + (fib[i - 2] / fib[i]) * L
        x2 = b - (fib[i - 2] / fib[i]) * L
        f1 = function(x1)
        f2 = function(x2)
        
        if f1 > f2:
            a = x1
            f1 = f2
            x1 = x2
            L = b - a
            x2 = b - (fib[i - 3] / fib[i - 1]) * L
            f2 = function(x2)
        else:
            b = x2
            f2 = f1
            x2 = x1
            L = b - a
            x1 = a + (fib[i - 3] / fib[i - 1]) * L
            f1 = function(x1)

    x_min = (a + b) / 2
    f_min = function(x_min)

    return x_min, f_min


def main(a, b, e):
    x_min, f_min = fibonacci_search(a, b, e)
    print(f"Минимум функции находится в x = {x_min} с значением f(x) = {f_min}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--a', help = 'Значение коэффициента a', type=int)
    parser.add_argument('--b', help = 'Значение коэффициента b', type=int)
    parser.add_argument('--e', help = 'Точность', type=float)
    args = parser.parse_args()
    main(args.a, args.b, args.e)
