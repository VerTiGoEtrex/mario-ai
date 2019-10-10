#!/usr/bin/env python

from multiprocessing import Pool
import json
import retro
import numpy as np  # For image-matrix/vector operations
import cv2  # For image reduction
import neat
import pickle

# env = retro.make(game='SuperMarioBros-Nes',
#  state='Level1-1', record=False)
oned_image = []
genome_to_replay_info = []
fileid = 0
generation = 1


def setupEnv():
    global env
    env = retro.make(game='SuperMarioBros-Nes',
                     state='Level1-1', record=False)


class ParallelEvaluator(object):
    def __init__(self, num_workers, eval_function, timeout=None):
        """
        eval_function should take one argument, a tuple of
        (genome object, config object), and return
        a single float (the genome's fitness).
        """
        self.num_workers = num_workers
        self.eval_function = eval_function
        self.timeout = timeout
        self.pool = Pool(num_workers, initializer=setupEnv)

    def __del__(self):
        self.pool.close()  # should this be terminate?
        self.pool.join()

    def evaluate(self, genomes, config):
        jobs = []
        for genome_id, genome in genomes:
            jobs.append(self.pool.apply_async(
                self.eval_function, (genome_id, genome, config)))

        # assign the fitness back to each genome
        for job, (genome_id, genome) in zip(jobs, genomes):
            genome.fitness = job.get(timeout=self.timeout)


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome_id, genome, config)


def eval_genome(genome_id, genome, config):
    ob = env.reset()  # First image
    inx, iny, inc = env.observation_space.shape  # inc = color
    # image reduction for faster processing
    inx = int(inx/8)
    iny = int(iny/8)
    # 20 Networks
    net = neat.nn.RecurrentNetwork.create(genome, config)
    current_max_fitness = 0
    fitness_current = 0
    frame = 0
    counter = 0

    done = False

    while not done:
        env.render()  # Optional
        frame += 1

        ob = cv2.resize(ob, (inx, iny))  # Ob is the current frame
        # Colors are not important for learning
        ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
        ob = np.reshape(ob, (inx, iny))

        oned_image = np.ndarray.flatten(ob)
        # Give an output for current frame from neural network
        neuralnet_output = net.activate(oned_image)
        # Try given output from network in the game
        ob, rew, done, info = env.step(neuralnet_output)

        fitness_current += rew
        if fitness_current > current_max_fitness:
            current_max_fitness = fitness_current
            counter = 0
        else:
            counter += 1
            # count the frames until it successful

        # Train mario for max 250 frames
        if done or counter == 250:
            done = True
            print(genome_id, fitness_current)
    return fitness_current


config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')
p = neat.Population(config)

p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
# Save the process after each 10 frames
p.add_reporter(neat.Checkpointer(10))

evaluator = ParallelEvaluator(5, eval_genome)
winner = p.run(evaluator.evaluate)

with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)

with open('info.json', 'w') as infoFile:
    json.dump(genome_to_replay_info, infoFile)

print("DONE!")
