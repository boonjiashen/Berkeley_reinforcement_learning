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

numTrainEpisodes = 1000
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

# Define environment
env = gym.make('CartPole-v0')
transformEnvState(env, lambda x: tuple(x))

# Define agent
#env = gym.make('Acrobot-v0')
kwargs = {
        'epsilon': 0.01,
        'gamma': 0.5,
        'alpha': 0.5, 
        }
agent = qlearningAgents.ApproximateQAgent1QPerAction(
        featureExtractors.StateToFeatures(bias=True),
        **kwargs)
agent.getLegalActions = lambda x: range(env.action_space.n)
agent.weights[0] = util.Counter(enumerate([100]*5))
agent.weights[1] = util.Counter(enumerate([100]*5))
#agent.featExtractor = TwoDegPolynomial()
print('epsilon = %f' % agent.epsilon)


if __name__ == '__main__':
    # You can optionally set up the logger. Also fine to set the level
    # to logging.DEBUG or logging.WARN if you want to change the
    # amount of outut.
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    def trainOneEpisode():
        state = env.reset()
        stateIsTerminal = False
        sars = [state]

        while not stateIsTerminal:
            #action = agent.getAction(state)
            action = random.choice([0, 1])
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
    env.monitor.start(outdir, force=True)

    # Training phase
    states = []
    for i in xrange(numTrainEpisodes):
        sars = trainOneEpisode() 
        states.extend(sars[::3])
        if (i+1) % 10 == 0:
            logging.info('Completed %i training episodes' % (i+1))
    states = np.vstack(states)
    logging.info('Max of states = %s' % str(np.max(states, axis=0)))
    logging.info('Min of states = %s' % str(np.min(states, axis=0)))

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
