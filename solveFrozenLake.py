"""Use Berkeley interface in Gym"""

import logging
import os
import sys
from qlearningAgents import QLearningAgent
import gym

if __name__ == '__main__':
    # You can optionally set up the logger. Also fine to set the level
    # to logging.DEBUG or logging.WARN if you want to change the
    # amount of outut.
    #logger = logging.getLogger()
    #logger.setLevel(logging.INFO)

    #env = gym.make('CartPole-v0')
    env = gym.make('FrozenLake-v0')
    kwargs = {
            'epsilon': 0.05,
            'gamma': 0.8,
            'alpha': 0.2,
            }
    agent = QLearningAgent(**kwargs)
    agent.getLegalActions = lambda x: range(env.action_space.n)

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
            #env.render()
            state = nextState

        #print('End of episode %i, last reward = %i' % (i, reward))
        sys.stdout.write('1' if reward > 0 else '0')
        return reward > 0

    def testOneEpisode():
        "Turns off greediness and returns success/failure (true/false)"
        state = env.reset()
        stateIsTerminal = False

        while not stateIsTerminal:
            action = agent.getPolicy(state)
            state, reward, stateIsTerminal, _ = env.step(action)
        return reward > 0

    numTrainEpisodes = 10000
    numTestEpisodes = 1000

    trainSuccesses = [trainOneEpisode() for _ in xrange(numTrainEpisodes)]
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
