import sympy as sp


def tst():
    objective = 'x1 ** 4 - 10*x1**2 + 5*x2**2 + x3**2  + x1 + x2 - x3'
    # x1 < 5, x2 > 3, x3 > 0
    constraints = ['x1 - 5', '3 - x2', '-x3']
    starting_points = [[4, 4, 1], [1, 2, 3], [5, 5, 5]]
    alpha = 0.2
    beta = 0.5
    mu = 10
    t = 5
    eps = 0.0001
    m = len(constraints)

    # Create the objective/constraint function and obtain its variables
    f0: sp.Function = sp.parse_expr(objective)
    fi = [sp.parse_expr(func) for func in constraints]
    variables = sorted(f0.free_symbols, key=lambda var: var.name)
    num_variables = len(variables)
    x = sp.Matrix(starting_points[0])

    # Barrier method (outer loop)
    while True:
        # Centering step
        # Assemble the barrier function
        obj_phi = t * f0 - sum([sp.log(-func) for func in fi])

        # Calculate the delta x n nt (newton step)
        obj_phi_grad = sp.Matrix([obj_phi.diff(var) for var in variables])
        obj_phi_hess: sp.MutableDenseMatrix = obj_phi_grad.jacobian(variables)
        delta_x: sp.MutableDenseMatrix = -obj_phi_hess.inv() * obj_phi_grad

        # Newton decrement
        lambda_x = sp.sqrt(obj_phi_grad.transpose() * obj_phi_hess.inv() * obj_phi_grad)

        # Newton's method (inner loop)
        # Compute the Newton step and decrement
        while True:
            newton_step: sp.MutableDenseMatrix = eval_fn(delta_x, variables, x)
            gradient = eval_fn(obj_phi_grad, variables, x)
            newton_dec = eval_fn(lambda_x, variables, x)

            # Calculate stopping criterion
            # newton_dec.base = newton_dec^2
            if newton_dec.base[0] / 2 <= eps:
                break

            # Perform line search
            inner_t = 1
            # While the search function value lies above the tangent line (of factor alpha), make t smaller
            while True:
                val1 = eval_fn(obj_phi, variables, x + inner_t * newton_step)
                val2 = eval_fn(obj_phi, variables, x) + (alpha * inner_t * gradient.transpose() * newton_step)[0]
                # If val1 is complex number, reduce t as it falls outside the feasible region
                if not val1.is_Number or val1 > val2:
                    inner_t = beta * inner_t
                else:
                    break

            # Update x
            x = x + inner_t * newton_step

        print(f'x = {x}')

        # Check stopping criterion
        if m / t < eps:
            break

        # Increase t
        t = mu * t


def eval_fn(f, variables, values):
    return f.subs(dict(zip(variables, values)))


if __name__ == '__main__':
    tst()
