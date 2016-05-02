"""Use Berkeley interface in Gym"""

import logging
import os
import sys
from qlearningAgents import QLearningAgent
import gym

# The world's simplest agent!
class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

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

    episode_count = 10000
    successes = []

    for i in xrange(episode_count):
        state = env.reset()
        stateIsTerminal = False

        while not stateIsTerminal:
            action = agent.getAction(state)
            #print(action)
            nextState, reward, stateIsTerminal, _ = env.step(action)
            agent.update(state, action, nextState, reward)
            #env.render()
            state = nextState
        else:
            #print('End of episode %i, last reward = %i' % (i, reward))
            sys.stdout.write('1' if reward > 0 else '0')
            successes.append(reward > 0)

    def testOneEpisode():
        "Turns off greediness and returns success/failure (true/false)"
        state = env.reset()
        stateIsTerminal = False

        while not stateIsTerminal:
            action = agent.getPolicy(state)
            state, reward, stateIsTerminal, _ = env.step(action)
        return reward > 0

    numTestEpisodes = 1000
    numSuccesses = sum(testOneEpisode() for _ in xrange(numTestEpisodes))
    print("Test success rate = %.1f%%" % (100.*numSuccesses/numTestEpisodes))

    ## Dump result info to disk
    #env.monitor.close()

    ## Upload to the scoreboard. We could also do this from another
    ## process if we wanted.
    #logger.info("Successfully ran RandomAgent. Now trying to upload results to the scoreboard. If it breaks, you can always just try re-uploading the same results.")
    #gym.upload(outdir, algorithm_id='random')
