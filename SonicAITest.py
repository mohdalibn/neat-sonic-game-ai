
# Importing the required libraries for the project
from sys import flags
import retro
import os
import pickle
import neat
import cv2
import numpy as np


# Loading saved model
with open('SonicParrallelWinner1.pkl', 'rb') as file:
    genome = pickle.load(file)

local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config-SonicNEAT.txt')

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)

NeuralNet = neat.nn.RecurrentNetwork.create(genome, config)

env = retro.make('SonicTheHedgehog2-Genesis')
observation = env.reset()
action = env.action_space.sample()

ImageArray = []

input_x, input_y, input_colors = env.observation_space.shape

input_x = int(input_x/8)
input_y = int(input_y/8)


done = False
while not done:

    env.render()

    observation = cv2.resize(observation, (input_x, input_y))
    observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
    observation = np.reshape(observation, (input_x, input_y))

    ImageArray = np.ndarray.flatten(observation)

    actions = NeuralNet.activate(ImageArray)

    observation, reward, done, info = env.step(actions)
