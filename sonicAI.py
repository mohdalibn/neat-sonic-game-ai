# The libraries for the project are imported below
from neat import statistics
import retro
import neat
import cv2
import pickle
import os
import numpy as np


# Making Our OpenAI Retro Environment
# the parameters are environment and state
env = retro.make('SonicTheHedgehog2-Genesis')

image_array = []

# THe eval_genomes function


def eval_genomes(genomes, config):
    global image_array

    for genome_id, genome in genomes:

        observation = env.reset()  # This gives us the first image of the game

        action = env.action_space.sample()

        # Creating variables for our inputs - y first and then x, followed by c
        input_y, input_x, input_colors = env.observation_space.shape

        # dividing the variables by 8 for our input layer - Remember that the number of inputs are 1120
        input_x = int(input_x/8)
        input_y = int(input_y/8)

        NeuralNet = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

        current_max_fitness = 0
        fitness_current = 0
        frame = 0
        counter = 0
        x_position = 0
        x_position_max = 0  # this variable will be used to check whether sonic is at the end of the level and terminate the neural network
        x_position_end = 0

        done = False

        # Visuals on what the neural network is seeing
        # cv2.namedWindow("main",cv2.WINDOW_NORMAL)

        while not done:
            env.render()
            frame += 1

            # THe following 2 lines are for the window which displays what the neural network actually sees.
            # scaledimg = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB)
            # scaledimg = cv2.resize(scaledimg, (input_x, input_y))

            # downsizing our current image/screenshot to the size for our neural network
            observation = cv2.resize(observation, (input_x, input_y))

            # turning it to grayscale
            observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)

            # Reshaping
            observation = np.reshape(observation, (input_x, input_y))

            # Neural Network vision
            # cv2.imshow("main", scaledimg)
            # cv2.waitKey(1)

            # Flattening/Compressing the 2d image into a single array
            for x in observation:
                for y in x:
                    image_array.append(y)

            netOutput = NeuralNet.activate(image_array)

            # print(netOutput)
            observation, reward, done, info = env.step(netOutput)

            image_array.clear()  # clearing the list each time

            # x_position = info['x']
            # x_position_end = info['screen_x_end']

            # if x_position > x_position_max:
            #     fitness_current += 1
            #     x_position_max = x_position

            # # checks whether sonic has reached the end of the level
            # if x_position == x_position_end:
            #     fitness_current += 100000 # 100000 is the fitness threshold in the config file. It is not likely to reach that score so we add 100000 if sonic reaches the end of the level
            #     done = True

            # FOR A MORE GENERIC APPROACH, WE WILL USE THE FOLLOWING LINE OF CODE TO INCREMENT THE REWARD AND COMMENT OUT THE LINES ABOVE
            fitness_current += reward

            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                counter = 0
            else:
                counter += 1

            if done or counter == 250:
                done = True
                print(genome_id, fitness_current)

            genome.fitness = fitness_current


def run(config_file):
    # Variable to store the name of our config file
    #config_file = 'config-SonicNEAT'

    # Setting up our NEAT Config file
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)

    # Creating the population
    population = neat.Population(config)

    # Adding Statistics to the cli(Command line interface)
    population.add_reporter(neat.StdOutReporter(True))
    statistics = neat.StatisticsReporter()
    population.add_reporter(statistics)
    # This saves a checkpoint every 10 generations
    population.add_reporter(neat.Checkpointer(10))

    # # Using threads for Parallelization
    # ParrEval = neat.ParallelEvaluator(6, eval_genomes)

    # Final result
    winner = population.run(eval_genomes)

    # Saving the trained model using pickle
    with open('SonicWinner.pkl', 'wb') as Outputfile:
        pickle.dump(winner, Outputfile, 1)


if __name__ == '__main__':

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-SonicNEAT.txt')

    run(config_path)
