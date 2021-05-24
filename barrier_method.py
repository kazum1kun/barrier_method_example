import sympy as sp


class BarrierOptimization:
    def __init__(self, objective, constraints, need_parse):
        # Create the objective/constraint function and obtain its variables
        if need_parse:
            self.f0: sp.Function = sp.parse_expr(objective)
            self.fi = [sp.parse_expr(func) for func in constraints]
        else:
            self.f0 = objective
            self.fi = constraints

        self.variables = set(self.f0.free_symbols)
        for cons in self.fi:
            self.variables.update(cons.free_symbols)
        self.variables = sorted(list(self.variables), key=lambda var: var.name)

        self.num_variables = len(self.variables)
        self.m = len(constraints)

    def optimize(self, starting_points, alpha, beta, mu, t, eps):
        orig_t = t

        for x in starting_points:
            print(f'\nInitial x = {x}')
            x = sp.Matrix(x)
            t = orig_t

            bm_round = 1
            # Barrier method (outer loop)
            while True:
                print(f'\nBarrier method round {bm_round}')
                print(f'Current t: {t}')

                # Centering step
                # Assemble the barrier function
                obj_phi = t * self.f0 - sum([sp.log(-func) for func in self.fi])

                # Calculate the delta x n nt (newton step)
                obj_phi_grad = sp.Matrix([obj_phi.diff(var) for var in self.variables])
                obj_phi_hess: sp.MutableDenseMatrix = obj_phi_grad.jacobian(self.variables)
                delta_x: sp.MutableDenseMatrix = -obj_phi_hess.inv() * obj_phi_grad

                # Newton decrement
                lambda_x = sp.sqrt(obj_phi_grad.transpose() * obj_phi_hess.inv() * obj_phi_grad)

                # Newton's method (inner loop)
                # Compute the Newton step and decrement
                nm_count = 1
                while True:
                    newton_step: sp.MutableDenseMatrix = self.eval_fn(delta_x, x)
                    gradient = self.eval_fn(obj_phi_grad, x)
                    newton_dec = self.eval_fn(lambda_x, x)

                    # Calculate stopping criterion
                    # newton_dec.base = newton_dec^2
                    if newton_dec.base[0] / 2 <= eps:
                        break

                    nm_count += 1
                    # Perform line search
                    inner_t = 1
                    # While the search function value lies above the tangent line (of factor alpha), make t smaller
                    while True:
                        val1 = self.eval_fn(obj_phi, x + inner_t * newton_step)
                        val2 = self.eval_fn(obj_phi, x) + (alpha * inner_t * gradient.transpose() * newton_step)[0]
                        # If val1 is complex number, reduce t as it falls outside the feasible region
                        if not val1.is_Number or val1 > val2:
                            inner_t = beta * inner_t
                        else:
                            break

                    # Update x
                    x = x + inner_t * newton_step

                print(f'Inner loop finished, took {nm_count} NM rounds')
                print(f'x = {x}')
                print(f'f(x) = {self.eval_fn(self.f0, x).evalf()}')

                # Check stopping criterion
                if self.m / t < eps:
                    break

                bm_round += 1
                # Increase t
                t = mu * t

    def phase_i(self, starting_points):
        for x in starting_points:
            # Choose an appropriate starting s value
            constr = sp.Matrix(self.fi)
            s = max(self.eval_fn(constr, x)) + 0.000001
            print(f'current value of s is {s}')

            # If s < 0, the problem is already strictly feasible, and no further action are needed
            # Otherwise, construct an an optimization problem that finds x that makes s less than zero
            if s >= 0:
                # New objective
                p1_objective = sp.parse_expr('s')
                p1_constraints = [cons - p1_objective for cons in self.fi]

                p1_problem = BarrierOptimization(p1_objective, p1_constraints, False)
                p1_problem.optimize([x], 0.2, 0.9, 10, 5, 0.1)

    def eval_fn(self, f, values):
        return f.subs(dict(zip(self.variables, values)))
