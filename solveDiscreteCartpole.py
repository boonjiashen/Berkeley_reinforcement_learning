"""Use Berkeley interface in Gym"""

import gym
import logging
import os
import sys
import itertools
import random
import numpy as np
import matplotlib.pyplot as plt

import MultiDimGrid
import gymUtil
import qlearningAgents
import featureExtractors 
import util
import Experiment

numTrainEpisodes = 50000
numEpisodesPerCheck = 100
numPolicyChecks = 30
trainRender = False
#trainRender = True
testRender = False

# Define environment
env = gym.make('CartPole-v0')
gymUtil.insertRewardAccumulator(env)
lims = (0.866, 2.55, .26, 3.2) # Define grid to discretize state space
#lims = (1.5, 2.55, .26, 3.2)
num_intervals = 10
ndgrid = MultiDimGrid.MultiDimGrid(
        [MultiDimGrid.SingleDimGrid(high=lim, low=-lim, num_intervals=num_intervals)
        for lim in lims])
gymUtil.transformEnvState(env, lambda x: ndgrid.discretize(x))

# Define agent
#env = gym.make('Acrobot-v0')
kwargs = {
        'epsilon': 0.01,
        'gamma': 1,
        'alpha': 0.2, 
        }
agent = qlearningAgents.QLearningAgent(**kwargs)
agent.getLegalActions = lambda x: range(env.action_space.n)

def make_experiment():
    kwargs = {
            'agent': agent,
            'environment': env,
            'numPolicyChecks': numPolicyChecks,
            'numEpisodesPerCheck': numEpisodesPerCheck,
            'numTrainEpisodes': numTrainEpisodes,
            }
    for key, value in sorted(kwargs.items(), key=lambda x: x[0]):
        logging.info('\t%s = %s' % (str(key,), str(value)))
    experiment = Experiment.Experiment(**kwargs)

    return experiment


if __name__ == '__main__':

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    exp = make_experiment()
    #outdir = '/tmp/random-agent-results'
    #env.monitor.start(outdir, force=True)

    exp.run()
    exp.plot()
    plt.show()

    # You can optionally set up the logger. Also fine to set the level
    # to logging.DEBUG or logging.WARN if you want to change the
    # amount of outut.

    ### Train linear model
    ##def features_and_q(qvalues):
        ##for (state, action), qvalue in qvalues.iteritems():
            ##yield state + tuple([action, 1]), qvalue
    ##A, b = zip(*list(features_and_q(agent.qvalues)))
    ##A = np.vstack(A)
    ##w, _, _, _ = np.linalg.lstsq(A, b)

    ### Dump result info to disk
    #env.monitor.close()

    ### Upload to the scoreboard. We could also do this from another
    ### process if we wanted.
    ##logger.info("Successfully ran RandomAgent. Now trying to upload results to the scoreboard. If it breaks, you can always just try re-uploading the same results.")
    ##gym.upload(outdir, algorithm_id='random')
