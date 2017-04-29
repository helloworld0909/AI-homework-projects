import util, math, random
from collections import defaultdict
from util import ValueIteration

############################################################
# Problem 2a

# If you decide 2a is true, prove it in blackjack.pdf and put "return None" for
# the code blocks below.  If you decide that 2a is false, construct a counterexample.
class CounterexampleMDP(util.MDP):
    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return -1
        # END_YOUR_CODE

    # Return set of actions possible from |state|.
    def actions(self, state):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return {-2: [0], -1:[-1], 0: [-1, 1], 1:[-1, 1], 2: [0]}[state]
        # END_YOUR_CODE

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        rewards = {-2: -100, -1:-5, 0:-5, 1:-5, 2:100}
        if action == 0:
            return [(state, 1.0, 0)]
        return [(state + action, 0.8, rewards[state + action]), (state - action, 0.2, rewards[state - action])]
        # END_YOUR_CODE

    def discount(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return 1.0
        # END_YOUR_CODE

############################################################
# Problem 3a

class BlackjackMDP(util.MDP):
    def __init__(self, cardValues, multiplicity, threshold, peekCost):
        """
        cardValues: array of card values for each card type
        multiplicity: number of each card type
        threshold: maximum total before going bust
        peekCost: how much it costs to peek at the next card
        """
        self.cardValues = cardValues
        self.multiplicity = multiplicity
        self.threshold = threshold
        self.peekCost = peekCost

    # Return the start state.
    # Look at this function to learn about the state representation.
    # The first element of the tuple is the sum of the cards in the player's
    # hand.
    # The second element is the index (not the value) of the next card, if the player peeked in the
    # last action.  If they didn't peek, this will be None.
    # The final element is the current deck.
    def startState(self):
        return (0, None, (self.multiplicity,) * len(self.cardValues))  # total, next card (if any), multiplicity for each card

    # Return set of actions possible from |state|.
    # You do not need to modify this function.
    # All logic for dealing with end states should be done in succAndProbReward
    def actions(self, state):
        return ['Take', 'Peek', 'Quit']

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.  Indicate a terminal state (after quitting or
    # busting) by setting the deck to None. 
    # When the probability is 0 for a particular transition, don't include that 
    # in the list returned by succAndProbReward.
    def succAndProbReward(self, state, action):

        valueInHand, nextCard, deckCount = state

        if valueInHand > self.threshold:
            return []
        if not deckCount:
            return []
        deckIndexCount = zip(range(len(deckCount)), deckCount)
        deckIndexCount = filter(lambda x: x[1] != 0, deckIndexCount)
        if not deckIndexCount:
            return []

        succ = []
        deckProb = map(lambda c: float(c) / sum(deckCount), deckCount)
        # assert abs(sum(deckProb) - 1.0) < 1e-6, 'deckProb Error {}'.format(deckProb)

        if action == 'Quit':
            return [((valueInHand, None, None), 1.0, valueInHand)]

        elif action == 'Take':
            reward = 0
            # Peeked in the last turn
            if nextCard is not None:
                newDeckCount = list(deckCount)
                newDeckCount[nextCard] -= 1
                newValue = valueInHand + self.cardValues[nextCard]
                if newValue > self.threshold:
                    newDeckCount = None
                    reward = 0
                else:
                    if not reduce(lambda a, b: a or b, newDeckCount):  # All elements in newDeckCount are zero
                        newDeckCount = None
                        reward = valueInHand + self.cardValues[nextCard]
                    else:
                        newDeckCount = tuple(newDeckCount)
                return [((newValue, None, newDeckCount), 1.0, reward)]

            # Not peeked in the last turn
            else:
                for index, count in deckIndexCount:
                    newValue = valueInHand + self.cardValues[index]
                    if newValue > self.threshold:
                        succ.append(((newValue, None, None), deckProb[index], reward))
                    else:
                        newDeckCount = list(deckCount)
                        newDeckCount[index] -= 1
                        if not reduce(lambda a,b: a or b, newDeckCount):    # All elements in newDeckCount are zero
                            newDeckCount = None
                            reward = newValue
                        else:
                            newDeckCount = tuple(newDeckCount)
                        succ.append(((newValue, None, newDeckCount), deckProb[index], reward))
                return succ

        elif action == 'Peek':
            # No double peeking
            if nextCard is not None:
                return []
            for index, count in deckIndexCount:
                succ.append(((valueInHand, index, deckCount), deckProb[index], -self.peekCost))
            return succ
        else:
            raise Exception('succ Error')

    def discount(self):
        return 1

############################################################
# Problem 3b

def peekingMDP():
    """
    Return an instance of BlackjackMDP where peeking is the optimal action at
    least 10% of the time.
    """

    return BlackjackMDP(cardValues=[16, 5, 6], multiplicity=2, threshold=20, peekCost=1)


############################################################
# Problem 4a: Q learning

# Performs Q-learning.  Read util.RLAlgorithm for more information.
# actions: a function that takes a state and returns a list of actions.
# discount: a number between 0 and 1, which determines the discount factor
# featureExtractor: a function that takes a state and action and returns a list of (feature name, feature value) pairs.
# explorationProb: the epsilon value indicating how frequently the policy
# returns a random action
class QLearningAlgorithm(util.RLAlgorithm):
    def __init__(self, actions, discount, featureExtractor, explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = defaultdict(float)
        self.numIters = 0

    # Return the Q function associated with the weights and features
    def getQ(self, state, action):
        score = 0
        for f, v in self.featureExtractor(state, action):
            score += self.weights[f] * v
        return score

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):
        self.numIters += 1
        if random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        else:
            return max((self.getQ(state, action), action) for action in self.actions(state))[1]

    # Call this function to get the step size to update the weights.
    def getStepSize(self):
        return 1.0 / math.sqrt(self.numIters)

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
    def incorporateFeedback(self, state, action, reward, newState):

        if newState is None:
            return

        stepSize = self.getStepSize()
        difference = reward + self.discount * max(self.getQ(newState, actionPrime) for actionPrime in self.actions(newState)) - self.getQ(state, action)
        for f, v in self.featureExtractor(state, action):
            self.weights[f] += stepSize * difference * v


# Return a singleton list containing indicator feature for the (state, action)
# pair.  Provides no generalization.
def identityFeatureExtractor(state, action):
    featureKey = (state, action)
    featureValue = 1
    return [(featureKey, featureValue)]

############################################################
# Problem 4b: convergence of Q-learning

# Test function
def comparePolicy(mdp, featureExtractor):
    if not hasattr(mdp, 'states'):
        mdp.computeStates()

    rl = QLearningAlgorithm(mdp.actions, mdp.discount(), featureExtractor)
    util.simulate(mdp, rl, numTrials=30000)

    # policy from Q-learning
    rl.explorationProb = 0.0
    QLearningPolicy = {}
    for state in mdp.states:
        QLearningPolicy[state] = rl.getAction(state)

    # value iteration method
    vi = ValueIteration()
    vi.solve(mdp, 0.001)

    matchCount = 0
    for state in mdp.states:
        if QLearningPolicy.get(state) == vi.pi.get(state):
            matchCount += 1

    print 'Match: {} / {}'.format(matchCount, len(mdp.states))

# Small test case
smallMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10, peekCost=1)

# Large test case
largeMDP = BlackjackMDP(cardValues=[1, 3, 5, 8, 10], multiplicity=3, threshold=40, peekCost=1)

def problem_4b():
    print '\n4-b'
    comparePolicy(smallMDP, identityFeatureExtractor)
    comparePolicy(largeMDP, identityFeatureExtractor)

############################################################
# Problem 4c: features for Q-learning.

# You should return a list of (feature key, feature value) pairs (see
# identityFeatureExtractor()).
# Implement the following features:
# - indicator on the total and the action (1 feature).
# - indicator on the presence/absence of each card and the action (1 feature).
#       Example: if the deck is (3, 4, 0 , 2), then your indicator on the presence of each card is (1,1,0,1)
#       Only add this feature if the deck != None
# - indicator on the number of cards for each card type and the action (len(counts) features).  Only add these features if the deck != None
def blackjackFeatureExtractor(state, action):
    total, nextCard, counts = state

    feature = []
    featureValue = 1
    feature.append(((total, action), featureValue))
    if counts is not None:
        feature.append(((tuple(int(bool(count)) for count in counts), action), featureValue))
        for index, count in enumerate(counts):
            feature.append((((index, count, counts[index]), action), featureValue))
    return feature

def problem_4c():
    if not hasattr(largeMDP, 'states'):
        largeMDP.computeStates()

    QL_1 = QLearningAlgorithm(largeMDP.actions, largeMDP.discount(), identityFeatureExtractor, 0.2)
    QL_2 = QLearningAlgorithm(largeMDP.actions, largeMDP.discount(), blackjackFeatureExtractor, 0.2)
    QLReward_1 = util.simulate(largeMDP, QL_1, numTrials=30000)
    QLReward_2 = util.simulate(largeMDP, QL_2, numTrials=30000)
    print '\n4-c'
    print 'QL reward using identityFeatureExtractor: {}'.format(sum(QLReward_1) * 1.0 / len(QLReward_1))
    print 'QL reward using blackjackFeatureExtractor: {}'.format(sum(QLReward_2) * 1.0 / len(QLReward_2))


############################################################
# Problem 4d: What happens when the MDP changes underneath you?!

# Original mdp
originalMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10, peekCost=1)

# New threshold
newThresholdMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=15, peekCost=1)

def problem_4d():
    if not hasattr(originalMDP, 'states'):
        originalMDP.computeStates()
    if not hasattr(newThresholdMDP, 'states'):
        newThresholdMDP.computeStates()

    VI = ValueIteration()
    VI.solve(originalMDP, 0.001)
    fixedVI = util.FixedRLAlgorithm(VI.pi)
    VIReward = util.simulate(newThresholdMDP, fixedVI, numTrials=30000)

    QL = QLearningAlgorithm(originalMDP.actions, originalMDP.discount(), blackjackFeatureExtractor, 0.2)
    QLReward = util.simulate(newThresholdMDP, QL, numTrials=30000)

    print '\n4-d'
    print 'VI reward: {}'.format(sum(VIReward)*1.0/len(VIReward))
    print 'QL reward: {}'.format(sum(QLReward)*1.0/len(QLReward))

if __name__ == '__main__':
    problem_4d()
    problem_4c()
    problem_4d()
