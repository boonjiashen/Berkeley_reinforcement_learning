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

numTrainEpisodes = 50000
numTestEpisodes = 100
trainRender = False
#trainRender = True
testRender = True

def transformEnvState(env, transform):
    """Transform an environment's state.

    Affects reset and step methods
    """
    oldStep = env.step
    oldReset = env.reset
    def newStep(*args, **kwargs):
        state, action, done, info = oldStep(*args, **kwargs)
        return transform(state), action, done, info
    def newReset(*args, **kwargs):
        return transform(oldReset(*args, **kwargs))
    env.step = newStep
    env.reset = newReset


class StateActionConcatenator(featureExtractors.FeatureExtractor):
    """Feature is the state and action concatenated.

    Assumes both state and action are numeric values,
    state is an iterable and action is a scalar
    """

    def __init__(self, bias=True):
        self.bias = bias

    def getFeatures(self, state, action):
        iterators = (
                state,  # deg 1 polynomial
                [action],
                [1] if self.bias else [],  # bias term
                )
        elements = itertools.chain(*iterators)
        feats = util.Counter(enumerate(elements))
        return feats

def getDeg2Polynomials(elements):
    return (x * y
            for x, y in itertools.combinations_with_replacement(elements, 2)
            )  # deg 2 polynomial

class TwoDegPolynomial(featureExtractors.FeatureExtractor):
    def getFeatures(self, state, action):
        iterators = (
                state,  # deg 1 polynomial
                deg2Polynomials,  # deg 2 polynomial
                [action],
                [1],  # bias term
                )
        elements = itertools.chain(*iterators)
        feats = util.Counter(enumerate(elements))
        return feats


if __name__ == '__main__':
    # You can optionally set up the logger. Also fine to set the level
    # to logging.DEBUG or logging.WARN if you want to change the
    # amount of outut.
    #logger = logging.getLogger()
    #logger.setLevel(logging.INFO)

    env = gym.make('CartPole-v0')
    transformEnvState(env, lambda x: tuple(x))

    #env = gym.make('Acrobot-v0')
    kwargs = {
            'epsilon': 0.0,
            'gamma': 0.5,
            'alpha': 0.05,
            }
    agent = qlearningAgents.ApproximateQAgent(**kwargs)
    agent.getLegalActions = lambda x: range(env.action_space.n)
    agent.featExtractor = StateActionConcatenator(bias=True)
    agent.weights = util.Counter(enumerate([100]*6))
    #agent.featExtractor = TwoDegPolynomial()
    print('epsilon = %f' % agent.epsilon)

    # You provide the directory to write to (can be an existing
    # directory, but can't contain previous monitor results. You can
    # also dump to a tempdir if you'd like: tempfile.mkdtemp().
    #outdir = '/tmp/random-agent-results'
    #env.monitor.start(outdir, force=True)

    def trainOneEpisode():
        state = env.reset()
        stateIsTerminal = False

        while not stateIsTerminal:
            action = agent.getAction(state)
            #action = random.choice([0, 1])
            actionValue = agent.getQValue(state, action)
            #sys.stdout.write('\r%s, %i, %f, %s' % (str(state), action, actionValue, str(agent.weights)))
            #sys.stdout.write('%i' % (action))
            #print(action)
            nextState, reward, stateIsTerminal, _ = env.step(action)
            agent.update(state, action, nextState, reward)
            if trainRender: env.render()
            state = nextState
            #sys.stdout.write(
            #print(reward)

        #print('End of episode %i, last reward = %i' % (i, reward))
        #sys.stdout.write('1' if reward > 0 else '0')
        return reward > 0

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

    for i in xrange(numTrainEpisodes):
        trainSuccesses = trainOneEpisode() 
        if (i+1) % 10 == 0:
            print('Completed %i training episodes' % (i+1))
    testSuccessRate =  \
            sum(testOneEpisode() for _ in xrange(numTestEpisodes)) / \
            float(numTestEpisodes)

    print("Test success rate = %.1f%%" % (100.* testSuccessRate))

    ## Dump result info to disk
    #env.monitor.close()

    ## Upload to the scoreboard. We could also do this from another
    ## process if we wanted.
    #logger.info("Successfully ran RandomAgent. Now trying to upload results to the scoreboard. If it breaks, you can always just try re-uploading the same results.")
    #gym.upload(outdir, algorithm_id='random')
