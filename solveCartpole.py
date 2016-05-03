"""Use Berkeley interface in Gym"""

import qlearningAgents
import featureExtractors 
import util
import gym
import logging
import os
import sys
import itertools

numTrainEpisodes = 10000
numTestEpisodes = 1000
trainRender = False
#trainRender = True
testRender = True

def castEnvStateToTuple(env):
    """Casts an environment's state to tuple.

    Affects reset and step methods
    """
    oldStep = env.step
    oldReset = env.reset
    def newStep(*args, **kwargs):
        state, action, done, info = oldStep(*args, **kwargs)
        return tuple(state), action, done, info
    def newReset(*args, **kwargs):
        return tuple(oldReset(*args, **kwargs))
    env.step = newStep
    env.reset = newReset

class StateActionConcatenator(featureExtractors.FeatureExtractor):
    """Feature is the state and action concatenated.

    Assumes both state and action are numeric values,
    state is an iterable and action is a scalar
    """
    def getFeatures(self, state, action):
        # 1 is the bias term
        elements = itertools.chain(state, [action, 1])
        feats = util.Counter(enumerate(elements))
        return feats

class TwoDegPolynomial(featureExtractors.FeatureExtractor):
    def getFeatures(self, state, action):
        deg2Polynomials = (
                x * y
                for x, y in itertools.combinations_with_replacement(state, 2)
                )  # deg 2 polynomial
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
    castEnvStateToTuple(env)

    #env = gym.make('Acrobot-v0')
    kwargs = {
            'epsilon': 0.05,
            'gamma': 0.8,
            'alpha': 0.2,
            }
    agent = qlearningAgents.ApproximateQAgent(**kwargs)
    agent.getLegalActions = lambda x: range(env.action_space.n)
    #agent.featExtractor = StateActionConcatenator()
    agent.featExtractor = TwoDegPolynomial()

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
            #print(action)
            nextState, reward, stateIsTerminal, _ = env.step(action)
            agent.update(state, action, nextState, reward)
            if trainRender: env.render()
            state = nextState
            #sys.stdout.write('\r%s' % (str(state)))
            #print(state)

        #print('End of episode %i, last reward = %i' % (i, reward))
        #sys.stdout.write('1' if reward > 0 else '0')
        return reward > 0

    def testOneEpisode():
        "Turns off greediness and returns success/failure (true/false)"
        state = env.reset()
        stateIsTerminal = False

        while not stateIsTerminal:
            action = agent.getPolicy(state)
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
