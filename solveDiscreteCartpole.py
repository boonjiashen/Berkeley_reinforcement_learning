"""Use Berkeley interface in Gym"""

import qlearningAgents
import featureExtractors 
import util
import gym
import logging
import os
import sys
import itertools
import random
import numpy as np
import MultiDimGrid
import gymUtil
import matplotlib.pyplot as plt

numTrainEpisodes = 500
numTestEpisodes = 10
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


if __name__ == '__main__':
    # You can optionally set up the logger. Also fine to set the level
    # to logging.DEBUG or logging.WARN if you want to change the
    # amount of outut.
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    #def tmp(x):
        #a, b, c, d = ndgrid.discretize(x)
        #return b, c, d

    def trainOneEpisode():
        state = env.reset()
        stateIsTerminal = False
        sars = [state]

        while not stateIsTerminal:
            action = agent.getAction(state)
            #action = random.choice([0, 1])
            #actionValue = agent.getQValue(state, action)
            #sys.stdout.write('\r%s, %i, %f, %s' % (str(state), action, actionValue, str(agent.weights)))
            #sys.stdout.write('%i' % (action))
            #print(action)
            nextState, reward, stateIsTerminal, _ = env.step(action)
            agent.update(state, action, nextState, reward)
            if trainRender: env.render()
            state = nextState
            #sys.stdout.write(
            #print(reward)

            sars.extend([action, reward, nextState])

        #print('End of episode %i, last reward = %i' % (i, reward))
        #sys.stdout.write('1' if reward > 0 else '0')
        #print env.total_reward_since_reset
        return sars

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
    #env.monitor.start(outdir, force=True)

    # Training phase
    learning_curve = []
    for i in xrange(numTrainEpisodes):
        sars = trainOneEpisode() 
        undiscounted_return = sum(sars[2::3])
        learning_curve.append(undiscounted_return)
        if (i+1) % 10 == 0:
            logging.info('Completed %i training episodes' % (i+1))

    # Train linear model
    def features_and_q(qvalues):
        for (state, action), qvalue in qvalues.iteritems():
            yield state + tuple([action, 1]), qvalue
    A, b = zip(*list(features_and_q(agent.qvalues)))
    A = np.vstack(A)
    w, _, _, _ = np.linalg.lstsq(A, b)

    testSuccessRate =  \
            sum(testOneEpisode() for _ in xrange(numTestEpisodes)) / \
            float(numTestEpisodes)

    print("Test success rate = %.1f%%" % (100.* testSuccessRate))

    ## Dump result info to disk
    env.monitor.close()

    ## Upload to the scoreboard. We could also do this from another
    ## process if we wanted.
    #logger.info("Successfully ran RandomAgent. Now trying to upload results to the scoreboard. If it breaks, you can always just try re-uploading the same results.")
    #gym.upload(outdir, algorithm_id='random')

    # Plot stuff
    def mean(items):
        return float(sum(items)) / len(items)
    smoothed = [mean(chunk) for chunk in util.chunks_of_size_n(iter(learning_curve), 100)]
    #plt.plot(learning_curve)
    plt.plot(smoothed)

    plt.show()
