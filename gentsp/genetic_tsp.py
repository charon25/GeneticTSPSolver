from typing import List


class TSPSolver:
    
    def __init__(self, distance_matrix: List[List[int]], **kwargs) -> None:
        """Initiate a TSP solver with a matrix containing the distance between every pair of points and optional parameters.

        Args:
            distance_matrix (List[List[int]]): A n*n matric which contains the distance between every pair of points. As such, M[i][j] should be equal to M[j][i].
        
        Keyword args:
            population_size (int): Size of the population used for the algorithm (default : 100).
            mutation_rate (float): Probability of mutation for any individual (default: 0.01).
            new_individuals (int): Number of new random individual to add at each generation (default: 0).
        
        Raises:
            TypeError, ValueError
        """

        self.distances = distance_matrix
        self.population_size = 100
        self.mutation_rate = 0.01
        self.new_individuals = 0

        for arg in kwargs:
            if arg == 'population_size':
                self.population_size = int(kwargs[arg])
            elif arg == 'mutation_rate':
                self.mutation_rate = float(kwargs[arg])
            elif arg == 'new_individuals':
                self.new_individuals = int(kwargs[arg])
        
        if self.population_size <= 0:
            raise ValueError(f'Population size ({self.population_size}) cannot be zero or negative.')
        if self.new_individuals > self.population_size:
            raise ValueError(f'There cannot be more new individuals ({self.new_individuals}) than the population size ({self.population_size}).')
        if not (1.0 >= self.mutation_rate >= 0.0):
            raise ValueError(f'Mutation rate ({self.mutation_rate}) should be between 0.0 and 1.0 (inclusive).')


    def _compute_fitness(self, individual: List[int]):
        """Compute the fitness of one individual based on the distance matrix.

        Args:
            individual (List[int]): Individual (ordered list of nodes).

        Returns:
            fitness (float): Inverse of the total distance multiplied by 1000.
        """

        total_distance = 0
        for node1, node2 in zip(individual, individual[1:]):
            total_distance += self.distances[node1][node2]
        
        return 1000 / total_distance

    def _sort_by_fitness(self, individuals: List[List[int]]):
        """Compute the fitness of every individuals and sort them in a decreasing order.

        Args:
            individuals (List[List[int]]): List of individuals (each individual is an ordered list of nodes).

        Returns:
            fitnesses (List[Tuple[float, List[int]]]): List of the individuals sort by decreasing fitness. The list contains tuple of the form (fitness, individual).
        """

        fitnesses = list()
        for individual in individuals:
            fitnesses.append((self._compute_fitness(individual), individual))

        fitnesses.sort(key=lambda couple: couple[0], reverse=True)

        return fitnesses
