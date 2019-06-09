from collections import namedtuple
from tqdm import tqdm
import numpy as np
import random
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt


class Genetic():
    """ Genetic algorithm solving optimalization task of given
        two variable function
    """

    def __init__(self, func, epochs=500, pop_size=100, mutation_chance=0.1):
        self.xrange = -3/2, 4
        self.yrange = -3, 4
        self.func = func
        self.pop_size = pop_size
        self.epochs = epochs
        self.mutation_chance = mutation_chance
        self.population = self.initialize_population()
        self.evaluate_population()

    @staticmethod
    def generate_random(a, b):
        return a + np.random.random() * (b - a)

    @staticmethod
    def float_to_binary(point):
        """ Convert (X, Y) point with floating numbers
            to binary.
        """
        x_whole, x_dec = str(point[0]).split(".")
        y_whole, y_dec = str(point[1]).split(".")

        x_whole, x_dec = int(x_whole), int(x_dec)
        y_whole, y_dec = int(y_whole), int(y_dec)

        return Point((format(x_whole, '010b'), format(x_dec, '010b')),
                     (format(y_whole, '010b'), format(y_dec, '010b')))

    @staticmethod
    def binary_to_float(point):
        """ Convert (X, Y) point with binary numbers
            to floating.
        """
        x, y = point
        x_whole, x_dec = int(x[0], 2), int(x[1], 2)
        y_whole, y_dec = int(y[0], 2), int(y[1], 2)
        return Point(float('.'.join([str(x_whole), str(x_dec)])),
                     float('.'.join([str(y_whole), str(y_dec)])))

    @staticmethod
    def swap_string(string, i, j):
        """ Swap i-th and j-th chars in a string """
        lst = list(string)
        lst[i], lst[j] = lst[j], lst[i]
        return ''.join(lst)

    def generate_winner(self):
        """ Randomly choose two points and choose a winner.
            Point is a winner if its value is lower than
            the opponent's, because we're minimizing the
            function.
        """
        fighter1, fighter2 = random.choice(self.population), \
                             random.choice(self.population)
        result1, result2 = self.func(*fighter1), self.func(*fighter2)
        if result1 < result2:
            return fighter1
        return fighter2

    def combine(self, point1, point2):
        """ Combine 2 points (X, Y) taking random whole and decimal
            from each into a new child.
        """
        bin1 = Genetic.float_to_binary(point1)
        bin2 = Genetic.float_to_binary(point2)
        x_wholes, y_wholes = (bin1[0][0], bin2[0][0]), (bin1[1][0], bin2[1][0])
        x_decs, y_decs = (bin1[0][1], bin2[0][1]), (bin1[1][1], bin2[1][1])

        child_x_whole = np.random.choice(x_wholes)
        child_y_whole = np.random.choice(y_wholes)

        child_x_dec = np.random.choice(x_decs)
        child_y_dec = np.random.choice(y_decs)

        if np.random.random() < self.mutation_chance:
            child_x_dec, child_y_dec = self.mutate(child_x_dec, child_y_dec)

        return Point((child_x_whole, child_x_dec),
                     (child_y_whole, child_y_dec))

    @classmethod
    def mutate(cls, x_dec, y_dec):
        """ Mutate given x decimal part or y decimal part, randomly """
        gene1, gene2 = 0, 0
        while gene1 == gene2:
            gene1 = np.random.randint(0, 11)
            gene2 = np.random.randint(0, 11)
        choice = np.random.random()

        if choice < 0.5:
            new_x_dec = Genetic.swap_string(x_dec, gene1, gene2)
            new_y_dec = y_dec
        if choice >= 0.5:
            new_x_dec = x_dec
            new_y_dec = Genetic.swap_string(y_dec, gene1, gene2)

        return new_x_dec, new_y_dec

    def initialize_population(self):
        """ Create a population of points """
        return [(self.generate_random(self.xrange[0], self.xrange[1]),
                 self.generate_random(self.yrange[0], self.yrange[1]))
                for i in range(self.pop_size)]

    def evaluate_population(self):
        """ Count the func of pairs X, Y in population """
        return [self.func(*i) for i in self.population]

    def next_generation(self):
        children = []
        while len(children) < self.pop_size:
            p1 = self.generate_winner()
            p2 = self.generate_winner()
            children.append(Genetic.binary_to_float(self.combine(p1, p2)))
        self.population = children

    def run(self):
        for _ in tqdm(range(self.epochs), desc='Evolution progress', unit='gen'):
            self.next_generation()
        return self.population[0]


if __name__ == '__main__':
    def plot3d():
        """ 3D Plot of F(x, y) and minimum found """
        ax = Axes3D(plt.figure())

        x = np.linspace(gen.xrange[0], gen.xrange[1], 1000)
        y = np.linspace(gen.yrange[0], gen.yrange[1], 1000)

        x1, x2 = np.meshgrid(x, y)
        Z = f(x1, x2)

        ax.plot_surface(x1, x2, Z, cmap=cm.get_cmap('rainbow'), alpha=0.6)
        ax.plot([min_x], [min_y], min_z, markerfacecolor='red', marker='o',
                markersize=10)

        plt.show()

    def plot2d():
        """ 2D plots for F(x, 0) and F(0, y) """
        x = np.linspace(gen.xrange[0], gen.xrange[1], 1000)
        y = np.linspace(gen.yrange[0], gen.yrange[1], 1000)

        plt.figure()
        plt.subplots_adjust(hspace=0.6)

        plt.subplot(211)
        plt.title('F(x, 0)')

        plt.plot(x, f(x, 0), color='green')
        plt.scatter(min_x, f(min_x, 0), color='red', label='Found minimum')

        plt.xlabel('x')
        plt.ylabel('z')
        plt.legend(loc='best')

        plt.subplot(212)
        plt.title('F(0, y)')

        plt.plot(y, f(0, y), color='blue')
        plt.scatter(min_y, f(0, min_y), color='red', label='Found minimum')

        plt.xlabel('y')
        plt.ylabel('z')
        plt.legend(loc='best')

        plt.show()

    Point = namedtuple('Point', ['x', 'y'])
    f = lambda x, y: np.sin(x + y) + (x + y)**2 - 3/2 * x + 5/2 * y + 1
    gen = Genetic(f, epochs=500, mutation_chance=0.2)

    minim = gen.run()

    min_x, min_y = minim
    min_z = f(min_x, min_y)

    print('Minimum found!', ' X =', min_x, ', Y =', min_y)
    print('F(min_x, min_y) =', min_z)

    # plot2d()
    plot3d()
