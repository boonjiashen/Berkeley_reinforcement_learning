import numpy as np
import logging

class Experiment(object):
    """Inspired by RLPy's Experiment class

    Experiment consists of an agent and an environment. One experiment is used
    to observe learning over time.

    Agent should be of type ReinforcmentAgent and should have methods
    `behaviorPolicy`, `estimationPolicy`, and `update`

    Environment should be of type gym.Env
    """

    def __init__(self, agent, environment, numPolicyChecks,
            numEpisodesPerCheck, numTrainEpisodes):
        self.agent = agent 
        self.environment = environment 
        self.numPolicyChecks = numPolicyChecks 
        self.numEpisodesPerCheck = numEpisodesPerCheck 
        self.numTrainEpisodes = numTrainEpisodes

    def trainOneEpisode(self):
        "Follows behavior policy, does learning, and returns sar-tuples"
        state = self.environment.reset()
        stateIsTerminal = False
        sars = [state]

        while not stateIsTerminal:
            action = self.agent.behaviorPolicy(state)
            nextState, reward, stateIsTerminal, _ = self.environment.step(action)
            self.agent.update(state, action, nextState, reward)
            state = nextState
            sars.extend([action, reward, nextState])

        return sars

    def testOneEpisode(self):
        """Follows estimation policy and returns sar-tuples

        Does no learning
        """
        state = self.environment.reset()
        stateIsTerminal = False
        sars = [state]

        while not stateIsTerminal:
            action = self.agent.estimationPolicy(state)
            nextState, reward, stateIsTerminal, _ = self.environment.step(action)
            state = nextState
            sars.extend([action, reward, nextState])

        return sars

    def check(self):
        """Returns the total reward of the estimation policy averaged over
        several episodes
        """

        def generateTotalReward():
            "Generate total reward for one test episode"
            return sum(self.testOneEpisode()[2::3])
        
        returns = []
        for i in range(self.numEpisodesPerCheck):
            returns.append(generateTotalReward())
            logging.info('Tested %i episodes' % (i+1))
        return np.mean(returns)

    def run(self):
        #TODO check for correctness
        checkInds = np.linspace(0, self.numTrainEpisodes-1,
                self.numPolicyChecks+1, dtype=int)
        checkInds = np.unique(checkInds)
        learning_curve = []
        for i in range(self.numTrainEpisodes):
            self.trainOneEpisode()
            logging.info('Completed %i training episodes' % (i+1))
            if i in checkInds:
                learning_curve.append(self.check())

        return learning_curve, checkInds
