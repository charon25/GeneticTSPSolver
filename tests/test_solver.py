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

        TSPSolver(distances, population_size=1000, mutation_rate=0.5, new_individuals=10, selection='best', breeder_count=100)
    
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
        with self.assertRaises(ValueError):
            TSPSolver(distances, elitism=-1)
        with self.assertRaises(ValueError):
            TSPSolver(distances, population_size=100, elitism=101)
        with self.assertRaises(ValueError):
            TSPSolver(distances, selection='string')
        with self.assertRaises(ValueError):
            TSPSolver(distances, breeder_count=-1)
        with self.assertRaises(ValueError):
            TSPSolver(distances, population_size=100, breeder_count=101)
        
    def test_fitness_calculation(self):
        distances = [
            [0, 1, 2],
            [1, 0, 3],
            [2, 3, 0]
        ]

        solver = TSPSolver(distances)
        fitness1 = solver._compute_fitness([0, 2, 1])
        fitness2 = solver._compute_fitness([2, 1, 0])

        self.assertAlmostEqual(fitness1, 1 / (3 + 2))
        self.assertAlmostEqual(fitness2, 1 / (3 + 1))

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
        correct_fitnesses = [1 / (6 + 3 + 1), 1 / (1 + 4 + 6), 1 / (3 + 5 + 4)]

        sorted_individuals, total_fitness = solver._sort_by_fitness(individuals)

        self.assertAlmostEqual(total_fitness, 1 / (6 + 3 + 1) + 1 / (1 + 4 + 6) + 1 / (3 + 5 + 4))

        for k, (fitness, individual) in enumerate(sorted_individuals):
            self.assertAlmostEqual(fitness, correct_fitnesses[k])
            self.assertListEqual(individual, correct_ordering[k])

    def test_create_initial_population(self):
        NODE_COUNT = 4
        distances = [[0 for _ in range(NODE_COUNT)] for _ in range(NODE_COUNT)]
        POPULATION_SIZE = 50

        solver = TSPSolver(distances, population_size=POPULATION_SIZE)

        self.assertEqual(len(solver.population), POPULATION_SIZE)

        for individual in solver.population:
            self.assertEqual(len(individual), NODE_COUNT)
            self.assertEqual(list(sorted(individual)), list(range(NODE_COUNT)))

    def test_select_mating_pool_method_best(self):
        BREEDER_COUNT = 2
        NODE_COUNT = 3
        distances = [[0 for _ in range(NODE_COUNT)] for _ in range(NODE_COUNT)]

        fitnesses = [
            (0.8, [0, 1, 2]),
            (0.5, [1, 2, 0]),
            (0.2, [2, 0, 1]),
            (0.1, [1, 0, 2])
        ]
        TOTAL_FITNESS = sum(couple[0] for couple in fitnesses)

        solver = TSPSolver(distances, population_size=4, selection='best', breeder_count=BREEDER_COUNT)

        mating_pool = solver._get_mating_pool(fitnesses, TOTAL_FITNESS)

        self.assertListEqual(mating_pool, [[0, 1, 2], [1, 2, 0]])

    def test_select_mating_pool_method_weighted(self):
        BREEDER_COUNT = 2
        NODE_COUNT = 3
        distances = [[0 for _ in range(NODE_COUNT)] for _ in range(NODE_COUNT)]

        fitnesses = [
            (0.8, [0, 1, 2]),
            (0.5, [1, 2, 0]),
            (0.2, [2, 0, 1]),
            (0.1, [1, 0, 2])
        ]
        TOTAL_FITNESS = sum(couple[0] for couple in fitnesses)

        solver = TSPSolver(distances, population_size=4, selection='weighted', breeder_count=BREEDER_COUNT)

        mating_pool = solver._get_mating_pool(fitnesses, TOTAL_FITNESS)

        self.assertEqual(len(mating_pool), BREEDER_COUNT)

        for i in range(BREEDER_COUNT):
            for j in range(BREEDER_COUNT):
                if i != j:
                    self.assertNotEqual(mating_pool[i], mating_pool[j])


if __name__ == '__main__':
    unittest.main()
