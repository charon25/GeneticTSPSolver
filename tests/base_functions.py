import unittest

from gentsp import TSPSolver

class TestDamages(unittest.TestCase):

    def test_creation(self):
        distances = [
            [1, 3],
            [3, 5]
        ]

        TSPSolver(distances)
    
    def test_creation_with_parameters(self):
        distances = [
            [1, 3],
            [3, 5]
        ]

        TSPSolver(distances, population_size=1000, mutation_rate=0.5, new_individuals=10)
    
    def test_creation_with_wrong_parameters(self):
        distances = [
            [1, 3],
            [3, 5]
        ]

        with self.assertRaises(ValueError):
            TSPSolver([[1, 3], [1]])
        with self.assertRaises(ValueError):
            TSPSolver(distances, population_size=-1)
        with self.assertRaises(ValueError):
            TSPSolver(distances, mutation_rate=1.5)
        with self.assertRaises(ValueError):
            TSPSolver(distances, mutation_rate=-0.2)
        with self.assertRaises(ValueError):
            TSPSolver(distances, population_size=100, new_individuals=101)
        with self.assertRaises(ValueError):
            TSPSolver(distances, new_individuals=-1)
        
    def test_fitness_calculation(self):
        distances = [
            [0, 1, 2],
            [1, 0, 3],
            [2, 3, 0]
        ]

        solver = TSPSolver(distances)
        fitness1 = solver._compute_fitness([0, 2, 1])
        fitness2 = solver._compute_fitness([2, 1, 0])

        self.assertAlmostEqual(fitness1, 1000 / (3 + 2))
        self.assertAlmostEqual(fitness2, 1000 / (3 + 1))

    def test_fitness_sorting(self):
        distances = [
            [0, 1, 2, 3],
            [1, 0, 4, 5],
            [2, 4, 0, 6],
            [3, 5, 6, 0]
        ]

        solver = TSPSolver(distances)

        individuals = [
            [0, 1, 2, 3],
            [0, 3, 1, 2],
            [2, 3, 0, 1]
        ]

        correct_ordering = [[2, 3, 0, 1], [0, 1, 2, 3], [0, 3, 1, 2]]
        correct_fitnesses = [1000 / (6 + 3 + 1), 1000 / (1 + 4 + 6), 1000 / (3 + 5 + 4)]

        sorted_individuals = solver._sort_by_fitness(individuals)

        for k, (fitness, individual) in enumerate(sorted_individuals):
            self.assertAlmostEqual(fitness, correct_fitnesses[k])
            self.assertListEqual(individual, correct_ordering[k])

if __name__ == '__main__':
    unittest.main()
