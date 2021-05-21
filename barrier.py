import cvxpy as cp


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
    # Parameters
    mu = 10
    t = 5
    eps = 0.0001
    m = 2

    # Variables
    x = cp.Variable(1)

    # Ineq. constraints
    f1 = 2 - x
    f2 = x - 4
    constraints = [f1 <= 0, f2 <= 0]

    # Objective function and phi (barrier) function
    f0 = x ** 2 + 1
    phi = - cp.log(-f1) - cp.log(-f2)

    optimize(f0, phi, constraints, t, x, m, mu, eps)


# Exercise 11.2
# min x_2
# s.t. x1 <= x2, 0 <= x2
def q2():
    # Parameters
    mu = 10
    t = 5
    eps = 0.0001
    m = 2

    # Variables
    x1 = cp.Variable(1)
    x2 = cp.Variable(1)

    # Ineq. constraints
    f1 = x1 - x2
    f2 = -x2
    constraints = [f1 <= 0, f2 <= 0]

    # Objective function and phi (barrier) function
    f0 = x2
    phi = - cp.log(-f1) - cp.log(-f2)

    optimize(f0, phi, constraints, t, (x1, x2), m, mu, eps)


# Question 3
# min 20x + 5y
# s.t. x - y >= 5
#      x >= 100
#      y >= 100
def q3():
    # Parameters
    mu = 10
    t = 5
    eps = 0.0001
    m = 3

    # Variables
    x = cp.Variable(1)
    y = cp.Variable(1)

    # Ineq. constraints
    f1 = 5 - x + y
    f2 = 100 - x
    f3 = 100 - y
    constraints = [f1 <= 0, f2 <= 0, f3 <= 0]

    # Objective function and phi (barrier) function
    f0 = 20*x + 5*y

    phi = - cp.log(-f1) - cp.log(-f2) - cp.log(-f3)

    optimize(f0, phi, constraints, t, (x, y), m, mu, eps)


# Question 4
# min 3x^3 + 4y^2
# s.t. x - y >= 5
#      x >= 0
#      y <= 0
def q4():
    # Parameters
    mu = 5
    t = 20
    eps = 0.0001
    m = 3

    # Variables
    x = cp.Variable(1)
    y = cp.Variable(1)

    # Ineq. constraints
    f1 = 5 - x + y
    f2 = -x
    f3 = y
    constraints = [f1 <= 0, f2 <= 0, f3 <= 0]

    # Objective function and phi (barrier) function
    f0 = 3*x**3 + 4*y**2

    phi = - cp.log(-f1) - cp.log(-f2) - cp.log(-f3)

    optimize(f0, phi, constraints, t, (x, y), m, mu, eps)


# Question 5
# min 6 log(x) + 12 log(y)
# s.t. x >= 10, y >= 15

def q5():
    # Parameters
    mu = 10
    t = 5
    eps = 0.0001
    m = 1

    # Variables
    x = cp.Variable(1)
    y = cp.Variable(1)

    # Ineq. constraints
    f1 = 10 - x
    f2 = 15 - y

    constraints = [f1 <= 0, f2 <= 0]

    # Objective function and phi (barrier) function
    f0 = 6 * cp.log(x) + 12 * cp.log(y)

    phi = - cp.log(-f1) - cp.log(-f2)

    optimize(f0, phi, constraints, t, (x, y), m, mu, eps)


def optimize(f0, phi, constraints, t, x, m, mu, eps):
    count = 1
    # Main Barrier method loop
    while True:
        # Centering step (outer loop)
        # Assemble the objective function (given the new t)
        objective = t * f0 + phi


        # Newton step (inner loop)
        # Determine the new newton step Delta x_nt
        v = cp.Variable(1)
        f_nt = v

        # objective = cp.Minimize(t * f0 + phi)
        # problem = cp.Problem(objective, constraints)
        # print(f'\nRound {count}, t = {t}')
        # print(f'Objective: {objective}')
        #
        # # Solve the problem
        # solution = problem.solve()
        # print(f'Problem status: {problem.status}, current solution: {solution / t}')
        for var in x:
            print(f'{var.name()} value: {var.value}')

        # Check stopping condition
        if m / t < eps:
            break

        # Increase t
        t = mu * t
        count += 1


if __name__ == '__main__':
    main()