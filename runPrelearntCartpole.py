"""Runs prelearnt weights on qlearning"""

import qlearningAgents
import featureExtractors 
import util, gymUtil
import gym
import logging
import os
import sys
import itertools
import random
import numpy as np

numTestEpisodes = 100
testRender = True

# Define environment
env = gym.make('CartPole-v0')
gymUtil.transformEnvState(env, lambda x: tuple(x))

# Define agent
#env = gym.make('Acrobot-v0')
kwargs = {
        'epsilon': 0.01,
        'gamma': 0.5,
        'alpha': 0.5, 
        }
agent = qlearningAgents.ApproximateQAgent(**kwargs)
agent.featExtrator = featureExtractors.StateToFeatures(bias=True)
agent.getLegalActions = lambda x: range(env.action_space.n)

# Learnt from QLearning
weights = [0.8827211 ,   0.69630211,   0.99550439,  -0.57754716,
         1.36921167,  13.72946649]
agent.weights = util.Counter(enumerate(weights))


if __name__ == '__main__':
    # You can optionally set up the logger. Also fine to set the level
    # to logging.DEBUG or logging.WARN if you want to change the
    # amount of outut.
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    def testOneEpisode():
        "Turns off greediness and returns success/failure (true/false)"
        state = env.reset()
        stateIsTerminal = False

        while not stateIsTerminal:
            action = agent.getPolicy(state)
            sys.stdout.write(str(action))
            state, reward, stateIsTerminal, _ = env.step(action)
            if testRender: env.render()
        return reward > 0

    outdir = '/tmp/random-agent-results'
    env.monitor.start(outdir, force=True)

    testSuccessRate =  \
            sum(testOneEpisode() for _ in xrange(numTestEpisodes)) / \
            float(numTestEpisodes)

    print("Test success rate = %.1f%%" % (100.* testSuccessRate))

    ## Dump result info to disk
    env.monitor.close()
