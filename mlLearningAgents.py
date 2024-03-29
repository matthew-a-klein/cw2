# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
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

# This template was originally adapted to KCL by Simon Parsons, but then
# revised and updated to Py3 for the 2022 course by Dylan Cope and Lin Li

from __future__ import absolute_import
from __future__ import print_function

import random

from pacman import Directions, GameState
from pacman_utils.game import Agent
from pacman_utils import util


class GameStateFeatures:
    
    """
    Wrapper class around a game state where you can extract
    useful information for your Q-learning algorithm

    WARNING: We will use this class to test your code, but the functionality
    of this class will not be tested itself
    """
    def __init__(self, state: GameState):
        self.state = state

        """
        Args:
            state: A given game state object
        """
        

    def __eq__(self, other):
        """
        Compare two GameStateFeatures objects for equality.

        Args:
            other: Another object to compare with.

        Returns:
            bool: True if both objects are equal, False otherwise.
        """
        return self.__hash__() == other.__hash__()
        
    
    def __hash__(self):
        """
        Return a hash value for the GameStateFeatures object.

        Returns:
            int: Hash value for the object.
        """
        # Hash the state attribute
        return self.state.__hash__()




class QLearnAgent(Agent):

    def __init__(self,
                 alpha: float = 0.2,
                 epsilon: float = 0.05,
                 gamma: float = 0.8,
                 maxAttempts: int = 30,
                 numTraining: int = 10):
        """
        These values are either passed from the command line (using -a alpha=0.5,...)
        or are set to the default values above.

        The given hyperparameters are suggestions and are not necessarily optimal
        so feel free to experiment with them.

        Args:
            alpha: learning rate
            epsilon: exploration rate
            gamma: discount factor
            maxAttempts: How many times to try each action in each state
            numTraining: number of training episodes
        """
        super().__init__()
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.maxAttempts = int(maxAttempts)
        self.numTraining = int(numTraining)
        self.prevState = []
        self.prevAction = []
        self.score = 0 
        # Record the state values
        self.q_values = util.Counter()
        # Record the times we have taken a particular action in each state
        self.visits = util.Counter()
        # Count the number of games we have played
        self.episodesSoFar = 0

    # Accessor functions for the variable episodesSoFar controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar += 1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value: float):
        self.epsilon = value

    def getAlpha(self) -> float:
        return self.alpha

    def setAlpha(self, value: float):
        self.alpha = value

    def getGamma(self) -> float:
        return self.gamma

    def getMaxAttempts(self) -> int:
        return self.maxAttempts

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    @staticmethod
    def computeReward(startState: GameState,
                      endState: GameState) -> float:
        """
        Args:
            startState: A starting state
            endState: A resulting state

        Returns:
            The reward assigned for the given trajectory
        """
        return endState.getScore() - startState.getScore()
        

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getQValue(self,
                  state: GameStateFeatures,
                  action: Directions) -> float:
        """
        Args:
            state: A given state
            action: Proposed action to take

        Returns:
            Q(state, action)
        """
        return self.q_values[(state,action)]

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def maxQValue(self, state: GameStateFeatures) -> float:
        """
        Args:
            state: The given state

        Returns:
            q_value: the maximum estimated Q-value attainable from the state
        """
        q_values = [self.getQValue(state, a) for a in state.state.getLegalPacmanActions()]
        return max(q_values, default=0)
    
    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def learn(self,
              state: GameStateFeatures,
              action: Directions,
              reward: float,
              nextState: GameStateFeatures):
        """
        Performs a Q-learning update

        Args: print(self.q_values)
            state: the initial state
            action: the action that was taken
            nextState: the resulting state
            reward: the reward received on this trajectory
        """
        q = self.getQValue(state,action)
        self.q_values[(state,action)] = q + self.alpha*(reward + self.gamma*self.maxQValue(nextState) - q)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def updateCount(self,
                    state: GameStateFeatures,
                    action: Directions):
        """
        Updates the stored visitation counts.

        Args:
            state: Starting state
            action: Action taken
        """
        visits = self.visits[(state, action)]
        self.visits[(state, action)] = visits + 1

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getCount(self,
                 state: GameStateFeatures,
                 action: Directions) -> int:
        """
        Args:
            state: Starting state
            action: Action taken

        Returns:
            Number of times that the action has been taken in a given state
        """
        return self.visits[(state, action)]

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def explorationFn(self,
                      utility: float,
                      counts: int) -> float:
        """
        Computes exploration function.
        Return a value based on the counts

        HINT: Do a greed-pick or a least-pick

        Args:
            utility: expected utility for taking some action a in some given state s
            counts: counts for having taken visited

        Returns:
            The exploration value
        """
        "*** YOUR CODE HERE ***"
        # Perform greed-pick and ignore the number of times we have done
        # the action before
        return utility 

        # return the action maximises Q of state
    def getBestAction(self, state):
        """
        Get the action that maximises the exploration function
        In practice, this is  the action that maximises reward
        Args:
            state: the current state

        Returns:
            The best action to take
        """
        legal = state.state.getLegalPacmanActions()
        tmp = util.Counter()
        for action in legal:
            tmp[action] = self.explorationFn(self.getQValue(state, action), self.visits[(state, action)])
        return tmp.argMax()

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getAction(self, state: GameState) -> Directions:
        """
        Choose an action to take to maximise reward while
        balancing gathering data for learning

        If you wish to use epsilon-greedy exploration, implement it in this method.
        HINT: look at pacman_utils.util.flipCoin

        Args:
            state: the current state

        Returns:
            The action to take
        """
        # The data we have about the state of the game
        legal = state.getLegalPacmanActions()
        stateFeatures = GameStateFeatures(state)

        reward = state.getScore() - self.score
        if len(self.prevState) > 0:
            last_state = self.prevState[-1]
            last_action = self.prevAction[-1]
            self.learn(last_state, last_action, reward, stateFeatures)
            
        # Pick the best action with probability 1 - epsilon
        if util.flipCoin(self.epsilon):
            action =  random.choice(legal)
        else:
            action = self.getBestAction(stateFeatures)
        self.updateCount(stateFeatures, action)   
        # update attributes
        self.score = state.getScore()
        self.prevState.append(stateFeatures)
        self.prevAction.append(action)

        # Now pick what action to take.
        return action

    def final(self, state: GameState):
        """
        Handle the end of episodes.
        This is called by the game after a win or a loss.

        Args:
            state: the final game state
        """
        # update Q-values
        reward = state.getScore()-self.score
        last_state = self.prevState[-1]
        last_action = self.prevAction[-1]
        self.learn(last_state, last_action, reward, GameStateFeatures(state))
        # reset attributes
        self.score = 0
        self.prevState = []
        self.prevAction = []
        print(f"Game {self.getEpisodesSoFar()} just ended!")
        
        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg, '-' * len(msg)))
            self.setAlpha(0)
            self.setEpsilon(0)
