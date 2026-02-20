

class Simplex():
    def __init__(self, objective, constraints):
        self.objective = objective
        self.constraints = constraints

        #convert the problem to standard form
        self._convert_to_standard_form()
    
    def _convert_to_standard_form(self):
        pass

    def solve(self):
        pass
    
    def solve_with_visualization(self):
        """solve LP problem using simplex method with visualizing the steps"""
        pass

if __name__ == "__main__":
    #todo

