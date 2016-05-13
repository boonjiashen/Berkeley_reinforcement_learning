# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math
import collections

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        self.qvalues = util.Counter() # A Counter is a dict with default 0

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        return self.qvalues[(state, action, )]  \
                if (state, action, ) in self.qvalues  \
                else 0.0

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        actions = self.getLegalActions(state)
        if actions:
            return max([self.getQValue(state, a) for a in actions])
        else:
            return 0.0

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        actions = self.getLegalActions(state)
        if not actions:
            return None

        # Choose a random action if best known q-value sucks
        bestQ = self.computeValueFromQValues(state)
        #if bestQ < 0: return random.choice(actions)

        # Randomly choose among optimal actions if best known q-value doesn't suck
        return random.choice([a for a in actions
                if self.getQValue(state, a) == bestQ])

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        takeRandomAction = util.flipCoin(self.epsilon)
        if not legalActions:
            return None
        if takeRandomAction:
            return random.choice(legalActions)
        return self.computeActionFromQValues(state)

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        oldQ = self.getQValue(state, action)
        nextActions = self.getLegalActions(nextState)
        if nextActions:
            newQ = reward + self.discount *   \
                    max([self.getQValue(nextState, nextAction)
                    for nextAction in nextActions])
        else:
            newQ = reward
        self.qvalues[(state, action, )] = (1 - self.alpha) * oldQ + self.alpha * newQ

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)

    # Define some synonyms which make more technical sense
    behaviorPolicy = getAction
    estimationPolicy = getPolicy



class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        qValue = self.featExtractor.getFeatures(state, action) * self.getWeights()
        #print 'in getQValue, features=\n', self.featExtractor.getFeatures(state, action)
        #print 'in getQValue, qValue = ', qValue
        return qValue

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """

        oldQ = self.getQValue(state, action)
        nextActions = self.getLegalActions(nextState)
        if nextActions:
            newQ = reward + self.discount *   \
                    max([self.getQValue(nextState, nextAction)
                    for nextAction in nextActions])
        else:
            newQ = reward
        difference = newQ - oldQ
        features = self.featExtractor.getFeatures(state, action)
        #print 'features\n', features
        #print 'weights\n', self.weights
        for key in features.keys():
            self.weights[key] += self.alpha * difference * features[key]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass

class ApproximateQAgent1QPerAction(PacmanQAgent):
    """
    Q(s,a) = w_a * f(s)

    That is we have one set of weights for each action
    """
    def __init__(self, extractor, **args):
        self.featExtractor = extractor
        PacmanQAgent.__init__(self, **args)
        self.weights = collections.defaultdict(util.Counter)

    def getWeights(self, action):
        return self.weights[action]

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        qValue = self.featExtractor.getFeatures(state) * self.getWeights(action)
        return qValue

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """

        oldQ = self.getQValue(state, action)
        nextActions = self.getLegalActions(nextState)
        if nextActions:
            newQ = reward + self.discount *   \
                    max([self.getQValue(nextState, nextAction)
                    for nextAction in nextActions])
        else:
            newQ = reward
        difference = newQ - oldQ
        features = self.featExtractor.getFeatures(state)
        #print 'features\n', features
        #print 'weights\n', self.weights
        currWeights = self.getWeights(action)
        for key in features.keys():
            currWeights[key] += self.alpha * difference * features[key]


NEGINF = -float('inf')

class UCBQLearningAgent(QLearningAgent):

    def __init__(self, damp, UCBConst, **kwargs):
    #def __init__(self, **kwargs):
        QLearningAgent.__init__(self, **kwargs)
        self.N = util.Counter()  # num visits for s, a
        self.UCB = util.Counter()  # wuh..
        self.V = util.Counter()  # variance of rewards at s, a
        self.minReward = 0  # minimum reward received in history
        self.maxReward = 0  # maximum reward received in history
        self.damp = damp  # damping factor (`r` in paper)
        self.UCBConst = UCBConst  # `C'` in paper)

        print('In UCBQ')
        print('damp = %f' % self.damp)
        print('UCBConst = %f' % self.UCBConst)


    def computeActionFromUCB(self, state):
        """Compute argmax_a UCB(s, a)

        If there are no legal actions, which is the case at the terminal state,
        you should return None.
        """
        actions = self.getLegalActions(state)
        if not actions:
            return None
        def key(action):
            if (state, action) not in self.UCB:
                return NEGINF
            else:
                return self.UCB[(state, action, )]
        return max(actions, key=key)


    def getAction(self, state):
        """Behavior policy is epsilon greedy UCB
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        takeRandomAction = util.flipCoin(self.epsilon)
        if not legalActions:
            return None
        if takeRandomAction:
            return random.choice(legalActions)
        return self.computeActionFromUCB(state)

    def updateRewardHistory(self, reward):
        if reward == 0:
            return
        self.maxReward = max(reward, self.maxReward)
        self.minReward = min(reward, self.minReward)

    def update(self, state, action, nextState, reward):
        oldQ = self.getQValue(state, action)
        oldV = self.V[(state, action, )]
        sa = (state, action, )

        # Update visit count
        self.N[sa] = self.N[sa] + 1
        newN = self.N[sa]

        # Normalize reward to be between 0 and 1 + self.discount * 
        self.updateRewardHistory(reward)
        reward = 1. * (reward - self.minReward) / (self.maxReward - self.minReward)

        # Update Q (to temporary value since we need the old Q to update V
        nextActions = self.getLegalActions(nextState)
        sample_return = reward + self.discount *   \
                max([self.getQValue(nextState, nextAction)
                for nextAction in nextActions])  \
                if nextActions else reward
        learning_rate = (1. - self.damp) / (1 - self.damp**newN)
        newQ = oldQ + learning_rate * (sample_return - oldQ)
        self.qvalues[sa] = newQ

        # Update variance
        V_numerator = (sum(self.damp**i * (oldV + oldQ**2) for i in range(1, newN+1)) + sample_return**2)
        V_denominator = sum(self.damp**i for i in range(1, newN+2))
        self.V[sa] = V_numerator / V_denominator 

        # Update UCB
        actions = self.getLegalActions(state)
        numVisitsToS = max(self.N[(state, x, )] for x in actions)  \
                if actions else 0 
        for currAction in actions:
            sca = (state, currAction, )  # Shorthand since used a lot here
            numVisitsToSA = max(1, self.N[sca])
            C = min(1./4, self.V[sca] + self.UCBConst *
                    math.sqrt(math.log(numVisitsToS)/numVisitsToSA))
            self.UCB[sca] = self.getQValue(*sca) + math.sqrt(C * math.log(numVisitsToS)) / numVisitsToSA
        

class PacmanUCBQAgent(UCBQLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2,
            numTraining=0, damp=0.5, UCBConst=0.5, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        UCBQLearningAgent.__init__(self, damp=float(damp), UCBConst=float(UCBConst), **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = UCBQLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action
