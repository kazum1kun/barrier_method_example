import numpy as np
import sympy as sp


class Optimizer:
    def __init__(self, obj_function: str):
        # Create the objective function and obtain its variables
        f: sp.Function = sp.parse_expr(obj_function)
        self.variables = sorted(f.free_symbols, key=lambda var: var.name)
        self.num_variables = len(self.variables)

        # Create partial derivatives (gradients) based on the function
        self.orig_gradient = [f.diff(var) for var in self.variables]
        self.gradient = [sp.lambdify(var, f.diff(var), modules='numpy') for var in self.variables]

        # Create lambda-fied version of f so it can be used as an function
        self.f = sp.lambdify(self.variables, f, modules='numpy')
        self.orig_f = f

    '''
    Perform the optimization
    
    On every iteration, the gradient at the current location is calculated and used to determine
    the search direction. Then, using backtracking line search, a new location is calculated
    which is strictly better than the current location. This process continues until the norm
    of the gradient is smaller or equal to the stop condition.
    
    If the objective is convex, this method guarantees to find a solution with error no greater 
    than the stopping condition. If the objective is non-convex, it may instead find local optima.
    
    Parameters:
        starting_point : [float]
            An vector that defines the starting position of the search
            Must be of the same length as the variable vector
        stop_condition : float 
            Once the norm of the gradient at current location is lesser or equal to this number, 
            the optimization/search will terminate
        alpha: float
            A value used to find step sizes. Larger alpha results in longer steps and vice versa
            Accepted values: (0, 1), recommended values: (0.2, 0.5)
        beta: float
            Another value used to find step sizes. Larger value results in a finer search, and 
            vice versa
            Accepted values: (0, 1), recommended values: ~0.5
        search_strategy: str
            Defines the strategy to search for descent direction. Valid options are:
                steepest: use the negative of gradient, -∇f(x) as the next descent direction (default)
                newton: use the newton's method, -H^(-1)∇f(x), where H is the Hessian of the function
                
    Return:
        Step-by-step result, including function values, x values, step sizes, and more
    '''

    def optimize(self, starting_point, stop_condition, alpha, beta, search_strategy='steepest'):
        self._check_params(starting_point, stop_condition, alpha, beta, search_strategy)
        output_log = []

        temp = ('Optimization starting!\n'
                f'The objective function is: {self.orig_f}\n'
                f'The free variable(s) are: {self.variables}\n')
        print(temp)
        output_log.append(temp)

        # Initialize output data structure
        output_data = []

        # Initialize gradient at the starting point and calculate the norm of the gradient
        gradient_x = np.array([x_prime(x_val) for x_prime, x_val in zip(self.gradient, starting_point)])
        gradient_norm = gradient_x.dot(gradient_x)
        x = starting_point

        # Initialize the counter and update output with data from iteration 0
        itr = 0
        output_data.append([itr, *x, self.f(*x), gradient_norm, 'N/A'])

        # Main optimization routine
        while gradient_norm > stop_condition:
            itr += 1
            # Update the search direction
            search_dir = self._calc_descent_dir(gradient_x, x, search_strategy)

            # Calculate step size using Armijo rule
            t = 1
            # While the search function value lies above the tangent line (of factor alpha), make t smaller
            while self.f(*(x + t * search_dir)) > self.f(*x) + alpha * t * gradient_x.dot(search_dir):
                t = beta * t
            x = x + t * search_dir

            # Gradient that's evaluated at a particular (x1, x2)
            gradient_x = np.array([x_prime(x_val) for x_prime, x_val in zip(self.gradient, x)])
            # Calculate the norm of the function
            gradient_norm = gradient_x.dot(gradient_x)

            output_data.append([itr, *x, self.f(*x), gradient_norm, t])

        temp = (f'Optimizer successfully finished after {itr} iterations.\n'
                f'alpha: {alpha}, beta: {beta}, strategy: {search_strategy}\n'
                f'starting point: {starting_point}, stopping condition: norm <= {stop_condition}\n'
                f'The result is {self.f(*x)} at {x}\n')
        print(temp)
        output_log.append(temp)

        return output_data, output_log

    # Return the descent direction based on the given strategy
    def _calc_descent_dir(self, gradient, x, strategy):
        # The steepest descent is simply the negative of its gradient at current location
        if strategy == 'steepest':
            return -gradient
        elif strategy == 'newton':
            # Create Hessian of the objective function. The Hessian is simply the Jacobian of the gradient
            hessian = np.array(sp.Matrix(self.orig_gradient).jacobian(self.variables))

            # Evaluate the hessian, if applicable
            for i in range(0, self.num_variables):
                hessian.flat[i] = hessian.flat[i].subs(self.variables[i], x[i])
            hessian = hessian.astype('float64')

            return -np.linalg.inv(hessian).__matmul__(gradient)

    # Check the parameters passed into the optimize function and make sure they are legal
    def _check_params(self, starting_point, stop_condition, alpha, beta, search_strategy):
        if len(starting_point) != len(self.variables):
            raise ValueError(
                f'Expecting starting point vector of size {len(self.variables)}, got {len(starting_point)}')
        if stop_condition < 0:
            raise ValueError('The stopping condition must be greater or equal to zero')
        if alpha <= 0 or alpha >= 1 or beta <= 0 or beta >= 1:
            raise ValueError('Alpha and beta values must be in range (0, 1)')
        if search_strategy != 'steepest' and search_strategy != 'newton':
            raise ValueError('Please specify a valid search strategy: either \'steepest\' or \'newton\'')
