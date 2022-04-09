
# Importing the required libraries for the project
import retro
import neat
import cv2
import pickle
import os
import numpy as np


class Thread(object):
    def __init__(self, genome, config):
        self.genome = genome
        self.config = config

    def process(self):
        self.env = retro.make('SonicTheHedgehog2-Genesis')
        observation = self.env.reset()
        action = self.env.action_space.sample()

        input_x, input_y, input_colors = self.env.observation_space.shape

        input_x = int(input_x/8)
        input_y = int(input_y/8)

        NeuralNet = neat.nn.RecurrentNetwork.create(self.genome, self.config)

        fitness = 0
        max_fitness = 0
        x_position = 0
        #x_position_max = 0
        x_position_end = 0
        counter = 0
        frame = 0
        ImageArray = []

        done = False
        while not done:

            # Resizing, reshaping, and converting the image into grayscale
            observation = cv2.resize(observation, (input_x, input_y))
            observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
            observation = np.reshape(observation, (input_x, input_y))

            # Flattening the 2d image into a single array/vector
            ImageArray = np.ndarray.flatten(observation)
            # for x in observation:
            #     for y in x:
            #         ImageArray.append(y)

            actions = NeuralNet.activate(ImageArray)

            observation, reward, done, info = self.env.step(actions)
            # ImageArray.clear()

            x_position = info['x']
            x_position_end = info['screen_x_end']

            if x_position >= 10500:
                fitness += 1000000
                done = True

            fitness += reward

            if fitness > max_fitness:
                max_fitness = fitness
                counter = 0
            else:
                counter += 1

            if done or counter > 250:
                done = True

            self.genome.fitness = fitness

        fitness = int(fitness)
        print(fitness)

        return fitness


def eval_genomes(genome, config):
    Processing = Thread(genome, config)
    return Processing.process()


def run(config_file):

    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)

    population = neat.Population(config)

    # Adding Statistics to the cli(Command line interface)
    population.add_reporter(neat.StdOutReporter(True))
    statistics = neat.StatisticsReporter()
    population.add_reporter(statistics)
    # This saves a checkpoint every 10 generations
    population.add_reporter(neat.Checkpointer(10))

    # this line makes use of 6 threads of the cpu
    ParrEval = neat.ParallelEvaluator(6, eval_genomes)

    ParrWinner = population.run(ParrEval.evaluate, 50)

    # Saving the trained model using pickle
    with open('SonicParrallelWinner1.pkl', 'wb') as Outputfile:
        pickle.dump(ParrWinner, Outputfile, 1)


if __name__ == '__main__':

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-SonicNEAT.txt')

    run(config_path)
