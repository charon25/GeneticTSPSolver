from random import choices, random, shuffle
from typing import Dict, List, Tuple, Union

from numpy.random import choice
from numpy import std
from tqdm import trange


class TSPSolver:
    
    def __init__(self, distance_matrix: List[List[int]], **kwargs) -> None:
        """Initiate a TSP solver with a matrix containing the distance between every pair of points and optional parameters.

        Args:
            distance_matrix (List[List[int]]): A n*n matric which contains the distance between every pair of points. As such, M[i][j] should be equal to M[j][i].
        
        Keyword args:
            population_size (int): Size of the population used for the algorithm (default : 100).
            
            mutation_rate (float): Probability of mutation for any individual (default: 0.05).
            
            new_individuals (int): Number of new random individual to add at each generation (default: 0).
            
            elitism (int): Number of individual to carry over each generation (default: 0).
            
            selection (str): Method to select which individuals can breed. Should be "best" or "weighted" (default: "weighted").
            "best" selects the individuals with the best fitness. "weighted" picks the individuals with probability proportional to their fitness.
            
            breeder_count (int): Number of individual which will breed (default: population_size / 2).
        
        Raises:
            TypeError, ValueError
        """

        self.distances = distance_matrix
        if any(len(self.distances) != len(row) for row in self.distances):
            raise ValueError(f'Distance matrix should be square.')
        self.node_count = len(self.distances)
        self.population_size = 100
        self.mutation_rate = 0.05
        self.new_individuals = 0
        self.elitism = 0
        self.selection = 'weighted'
        self.breeder_count = None

        for arg in kwargs:
            if arg == 'population_size':
                self.population_size = int(kwargs[arg])
            elif arg == 'mutation_rate':
                self.mutation_rate = float(kwargs[arg])
            elif arg == 'new_individuals':
                self.new_individuals = int(kwargs[arg])
            elif arg == 'elitism':
                self.elitism = int(kwargs[arg])
            elif arg == 'selection':
                self.selection = kwargs[arg]
            elif arg == 'breeder_count':
                self.breeder_count = int(kwargs[arg])

        if self.breeder_count is None:
            self.breeder_count = int(self.population_size // 2)

        if self.population_size <= 0:
            raise ValueError(f'Population size ({self.population_size}) cannot be zero or negative.')
        if not (1.0 >= self.mutation_rate >= 0.0):
            raise ValueError(f'Mutation rate ({self.mutation_rate}) should be between 0.0 and 1.0 (inclusive).')
        if self.new_individuals > self.population_size:
            raise ValueError(f'There cannot be more new individuals ({self.new_individuals}) than the population size ({self.population_size}).')
        if self.new_individuals < 0:
            raise ValueError(f'New individuals count ({self.new_individuals}) should be non negative.')
        if self.elitism < 0:
            raise ValueError(f'Elitism ({self.elitism}) should be non negative.')
        if self.elitism > self.population_size:
            raise ValueError(f'Elitism ({self.elitism}) cannot be greater than the population size ({self.population_size}).')
        if self.selection not in ('weighted', 'best'):
            raise ValueError(f'Selection method should be "weighted" or "best".')
        if self.breeder_count <= 0:
            raise ValueError(f'Breeder count ({self.breeder_count}) cannot be zero or negative.')
        if self.breeder_count > self.population_size:
            raise ValueError(f'There cannot be more new breeder ({self.breeder_count}) than the population size ({self.population_size}).')

        self._initialise_population()


    def _compute_fitness(self, individual: List[int]) -> float:
        """Compute the fitness of one individual based on the distance matrix.

        Args:
            individual (List[int]): Individual (ordered list of nodes).

        Returns:
            fitness (float): Inverse of the total distance.
        """

        total_distance = 0
        for node1, node2 in zip(individual, individual[1:]):
            total_distance += self.distances[node1][node2]
        
        return 1 / total_distance


    def _sort_by_fitness(self, individuals: List[List[int]]) -> Tuple[List[Tuple[float, List[int]]], float]:
        """Compute the fitness of every individuals and sort them in a decreasing order. Also return the total fitness (for the normalisation).

        Args:
            individuals (List[List[int]]): List of individuals (each individual is an ordered list of nodes).

        Returns:
            fitnesses (Tuple[List[Tuple[float, List[int]]], float]): A tuple containing the list of the individuals sort by decreasing 
            fitness as well as the total fitness. The list contains tuple of the form (fitness, individual).
        """

        fitnesses = list()
        total_fitness = 0.0

        for individual in individuals:
            fitness = self._compute_fitness(individual)
            fitnesses.append((fitness, individual))
            total_fitness += fitness

        fitnesses.sort(key=lambda couple: couple[0], reverse=True)

        return (fitnesses, total_fitness)

    def _create_random_individual(self) -> List[int]:
        nodes = list(range(self.node_count))
        shuffle(nodes)
        return nodes

    def _initialise_population(self) -> None:
        """Create an initial random population of the right size."""
        self.population: List[List[int]] = []

        for _ in range(self.population_size):
            self.population.append(self._create_random_individual())


    def _get_mating_pool_best(self, fitnesses: List[Tuple[float, List[int]]]) -> List[List[int]]:
        """Return the individuals with the best fitness. The number is dependant on the breeder_count parameter.

        Args:
            fitnesses (List[Tuple[float, List[int]]]): List of tuples of (fitness, individual), as returned by the _sort_by_fitness function.

        Returns:
            List[List[int]]: The <breeder_count> individuals with the best fitness.
        """

        # Second element is the individual
        return [couple[1] for couple in fitnesses[:self.breeder_count]]

    def _get_mating_pool_weighted(self, fitnesses: List[Tuple[float, List[int]]], total_fitness: float) -> List[List[int]]:
        """Return a random selection of individual where the probability of being chosen is proportional to the fitness.
        The number is dependant on the breeder_count parameter.

        Args:
            fitnesses (List[Tuple[float, List[int]]]): List of tuples with fitness and individuals, as returned by the _sort_by_fitness function.

            total_fitness (float): Sum of all fitnesses.

        Returns:
            List[List[int]]: <breeder_count> individuals.
        """

        # First element is the fitness
        probabilities = [couple[0] / total_fitness for couple in fitnesses]
        indexes = choice(self.population_size, size=self.breeder_count, replace=False, p=probabilities)

        return [fitnesses[index][1] for index in indexes]

    def _get_mating_pool(self, fitnesses: List[Tuple[float, List[int]]], total_fitness: float) -> List[List[int]]:
        """Return a list of individual chosen based on the selection method.

        Args:
            fitnesses (List[Tuple[float, List[int]]]): List of tuples with fitness and individuals, as returned by the _sort_by_fitness function.

            total_fitness (float): Sum of all fitnesses.

        Raises:
            ValueError: if the selection method is not 'best' or 'weighted'.

        Returns:
            List[List[int]]: <breeder_count> individuals.
        """
        if self.selection == 'best':
            return self._get_mating_pool_best(fitnesses)
        elif self.selection == 'weighted':
            return self._get_mating_pool_weighted(fitnesses, total_fitness)
        else:
            raise ValueError(f"Unknown selection method ('{self.selection}').")


    def _breed(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Breed the two parents randomly into a new individual.

        Args:
            parent1 (List[int])

            parent2 (List[int])

        Returns:
            List[int]: new individual.
        """
        child: List[int] = [-1 for _ in range(self.node_count)]

        start = int(random() * self.node_count)
        end = int(random() * self.node_count)

        if start > end:
            start, end = end, start
        
        child[start:end] = parent1[start:end]

        for i in range(self.node_count):
            if start <= i < end:
                continue

            if child[i] == -1:
                di = 0
                while parent2[(i + di) % self.node_count] in child:
                    di += 1
                child[i] = parent2[(i + di) % self.node_count]
        
        return child


    def _breed_population(self, fitnesses: List[Tuple[float, List[int]]], mating_pool: List[List[int]]) -> List[List[int]]:
        """Create the new generation based on the precedent one and simulation parameters. Will first add the elite
        (<elitism> best from the previous generation) then reproduce the individuals in the mating pool,
        and then add <new_individuals> new random individuals.

        Args:
            fitnesses (List[Tuple[float, List[int]]]): List of tuples with fitness and individuals, as returned by the _sort_by_fitness function.

            mating_pool (List[List[int]]): List of individuals which will be used for breeding.

        Returns:
            List[List[int]]: The population of the new generation.
        """
        new_population: List[List[int]] = []

        # Elitism
        new_population.extend([couple[1] for couple in fitnesses[:self.elitism]])

        # Breeding
        breed_count = self.population_size - self.elitism - self.new_individuals
        indexes = choices(mating_pool, k=breed_count)

        for i in range(breed_count):
            parent1 = indexes[i]
            parent2 = indexes[breed_count - i - 1]
            new_population.append(self._breed(parent1, parent2))

        # Random new individuals
        new_population.extend([self._create_random_individual() for _ in range(self.new_individuals)])

        return new_population


    def _mutate(self, individual: List[int]) -> List[int]:
        """With the given mutation rate, mutate an individual by swapping two of its nodes.

        Args:
            individual (List[int])

        Returns:
            List[int]: possibly mutated individual.
        """
        if random() < self.mutation_rate:
            index1 = int(random() * self.node_count)
            index2 = int(random() * self.node_count)

            individual[index1], individual[index2] = individual[index2], individual[index1]

        return individual


    def _mutate_population(self, population: List[List[int]]) -> List[List[int]]:
        """Apply mutate to all the population.

        Args:
            new_population (List[List[int]])

        Returns:
            List[List[int]]: Mutated population
        """
        for i in range(self.population_size):
            population[i] = self._mutate(population[i])
        
        return population


    def _get_population_stats(self, fitnesses: List[Tuple[float, List[int]]], total_fitness: float) -> Dict[str, Union[float, List[int]]]:
        """Return some statistics about the population.

        Args:
            fitnesses (List[Tuple[float, List[int]]]): List of tuples with fitness and individuals, as returned by the _sort_by_fitness function.
            total_fitness (float): Sum of all fitnesses

        Returns:
            Dict[str, Union[float, List[int]]]: Dictionary containing some statistics about the population.
        """

        stats = {}

        stats['best_fitness'] = fitnesses[0][0]
        stats['best_distance'] = 1 / fitnesses[0][0]
        stats['best_individual'] = fitnesses[0][1]
        stats['worst_fitness'] = fitnesses[-1][0]
        stats['worst_distance'] = 1 / fitnesses[-1][0]
        stats['worst_individual'] = fitnesses[-1][1]
        stats['average_fitness'] = total_fitness / self.population_size
        stats['average_distance'] = 1 / stats['average_fitness']

        stats['fitness_stdev'] = std([couple[0] for couple in fitnesses])

        stats['average_individual'] = fitnesses[0][1]
        min_diff_to_average = 10**99
        for fitness, individual in fitnesses:
            if abs(fitness - stats['average_fitness']) < min_diff_to_average:
                min_diff_to_average = abs(fitness - stats['average_fitness'])
                stats['average_individual'] = individual

        return stats


    def pass_one_generation(self) -> Dict[str, Union[float, List[int]]]:
        """Apply all the steps of one generation, and returns the stats of this generation.

        Returns:
            Dict[str, Union[float, List[int]]]: The statisticss of the generation.
        """

        fitnesses, total_fitness = self._sort_by_fitness(self.population)
        mating_pool = self._get_mating_pool(fitnesses, total_fitness)
        new_population = self._breed_population(fitnesses, mating_pool)
        self.population = self._mutate_population(new_population)

        return self._get_population_stats(fitnesses, total_fitness)


    def pass_generations(self, n_generations: int) -> Dict[str, List[Union[float, List[int]]]]:
        stats: Dict[str, List[Union[float, List[int]]]] = {}

        for _ in trange(n_generations, desc='Generations'):
            generation_stats = self.pass_one_generation()

            for key, value in generation_stats.items():
                if not key in stats:
                    stats[key] = []

                stats[key].append(value)

        return stats


if __name__ == '__main__':

    # Minimum distance in this example is 9
    distances = [
        [0, 7, 9, 6, 1],
        [7, 0, 3, 4, 9],
        [9, 3, 0, 2, 5],
        [6, 4, 2, 0, 3],
        [1, 9, 5, 3, 0]
    ]

    solver = TSPSolver(distances)

    stats = solver.pass_generations(100)

    print(f'Average distance after 100 generations : {stats["average_distance"][-1]:.1f}')
    print(f'Best distance after 100 generations : {stats["best_distance"][-1]:.1f}')
    print(f'Best individual : {stats["best_individual"][-1]}')
