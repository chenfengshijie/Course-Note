import numpy as np

def simplex_method(constraint_coeffs, constraints, objective_coeffs):
    num_constraints, num_vars = constraint_coeffs.shape
    
    constraint_coeffs = np.hstack([constraint_coeffs, np.eye(num_constraints)])
    objective_coeffs = np.hstack([objective_coeffs, np.zeros(num_constraints)])
    basis = np.arange(num_vars, num_vars + num_constraints)
    epsilon = 1e-6
    
    # Initialize simplex tableau
    tableau = np.zeros((num_constraints + 1, num_vars + num_constraints + 1))
    tableau[:-1, :-1] = constraint_coeffs
    tableau[:-1, -1] = constraints
    tableau[-1, :-1] = objective_coeffs
    reduced_costs = objective_coeffs 
    
    while reduced_costs.max() > 0:
        # Choose entering variable
        entering = np.argmax(reduced_costs)
        
        if tableau[:-1, entering].max() <= 0:
            return None, np.inf
        
        # Choose leaving variable
        ratios = tableau[:-1, -1] / (tableau[:-1, entering] + epsilon)
        ratios[ratios <= 0] = np.inf
        leaving = np.argmin(ratios)
        basis[leaving] = entering
        
        pivot = tableau[leaving, entering]
        tableau[leaving] /= pivot
        for i in range(num_constraints + 1):
            if i != leaving:
                tableau[i] -= tableau[i, entering] * tableau[leaving]
        
        # Compute new objective function value
        optimal_value = -tableau[-1, -1]
        optimal_solution = np.zeros(num_vars + num_constraints)
        optimal_solution[basis] = tableau[:-1, -1]
        reduced_costs = tableau[-1, :-1]
    
    return optimal_solution[:num_vars], optimal_value

if __name__ == "__main__":
    A = np.array([[1, 4, 2], [1, 2, 4]])
    b = np.array([48, 60])
    c = np.array([6, 14, 13])
    
    optimal_solution, optimal_value = simplex_method(A, b, c)
    print("Optimal solution X:", optimal_solution)
    print("Maximum optimal value Zmax:", optimal_value)