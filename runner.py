from barrier_method import BarrierOptimization

alpha = 0.2
beta = 0.5
mu = 10
t = 5
eps = 0.0001


def main():
    # q1()   # Simple quadratic w/ ineqs
    # q2()   # Bad LP?
    # q3()   # Simple LP w/ ineqs
    q4()   # Quadractic w/ 2 vars and ineqs
    # q5()   # Weighted sum of log functions


# Exercise 11.1
# A simple problem
# min   x^2 +1
# s.t.  2 <= x <= 4
def q1():
    objective = 'x**2 + 1'
    constraints = ['2 - x', 'x - 4']
    starting_points = [3, 3.5, 2.1]

    problem = BarrierOptimization(objective, constraints)
    problem.optimize(starting_points, alpha, beta, mu, t, eps)


# Exercise 11.2
# min x_2
# s.t. x1 < x2, 0 < x2
def q2():
    objective = 'x2'
    constraints = ['x1 - x2', '-x2']
    starting_points = [[1, 5], [2, 4], [3, 10]]

    problem = BarrierOptimization(objective, constraints)
    problem.optimize(starting_points, alpha, beta, mu, t, eps)


# Question 3
# min 20x + 5y
# s.t. x - y >= 5
#      x > 100
#      y > 100
def q3():
    objective = '20*x + 5*y'
    constraints = ['5 - x + y', '100 - x', '100 - y']
    starting_points = [[120, 104], [5000, 101]]

    problem = BarrierOptimization(objective, constraints)
    problem.optimize(starting_points, alpha, beta, mu, t, eps)


# Question 4
# min x1^4 - 10x1^2  + 5x2^2 + x3^2 + x1 + x2 - x3
# s.t. x1 < 5
#      x2 > 3
#      x3 > 0
def q4():
    objective = 'x1 ** 4 - 10*x1**2 + 5*x2**2 + x3**2  + x1 + x2 - x3'
    constraints = ['x1 - 5', '3 - x2', '-x3']
    starting_points = [[4, 4, 1], [1, 6, 3], [4, 5, 10]]

    problem = BarrierOptimization(objective, constraints, )
    problem.optimize(starting_points, alpha, beta, mu, t, eps)


# Question 5
# min 6 log(x) + 8 log(y)
# s.t. x > 10, y > 15

def q5():
    objective = '6*log(x) + 8*log(y)'
    constraints = ['10 - x', '15 - y']
    starting_points = [[20, 40], [100, 200], [53, 43]]

    problem = BarrierOptimization(objective, constraints, )
    problem.optimize(starting_points, alpha, beta, mu, t, eps)


if __name__ == '__main__':
    main()
