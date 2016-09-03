import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.qEstimate = {}
        self.learningRate = 1
        self.discountRate = 0
        self.randomExploration = 0
        self.randomExplorationDecay = 0.00075
        #Use the following to keep track of progress over the trials
        self.randomExplorationTrack = [0]
        self.cumScore = [0]
        self.completionTime = [0]
        self.wrongMoves = [0]

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        print(str(self.cumScore[-1]) + '|' + str(self.completionTime[-1]) + '|' + str(self.wrongMoves[-1]))
        print(self.randomExploration)
        #Update our progress tracking variables
        self.cumScore.append(0)
        self.completionTime.append(0)
        self.wrongMoves.append(0)
        self.randomExplorationTrack.append(0)


    def getQEstimate(self, inputState=None, inputAction='All'):
        #Use this method to get a specific q value or initialize if nothing is recorded yet\
        if inputState is None:
            return self.qEstimate
        elif inputState not in self.qEstimate and not inputState[0] is None:
            self.qEstimate[inputState] = dict.fromkeys(self.env.valid_actions, 0)
        if inputAction is 'All':
            return self.qEstimate[inputState]
        else:
            return self.qEstimate[inputState][inputAction]

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        #Record time for this last run, will use this to track progress near the end
        self.completionTime[-1] = t
        # TODO: Update state
        #If we have reached the destination we will not update our stats
        if not self.next_waypoint is None:
            #Our state will be a tuple of input values for (not in order): Waypoint, Light, Oncoming, Left, Right
            self.state = tuple([self.next_waypoint]) + tuple([str(x) for x in inputs.values()])
            # TODO: Select action according to your policy
            #We will select randomly based on our decaying epsilon: self.randomExploration
            tempCurrent = self.getQEstimate(self.state)
            if random.randrange(0,1000,1)/1000.0 >= self.randomExploration:
                #Pick a random selection of the highest value for all possible actions given the current state
                tempCurrent = [act for act, vals in tempCurrent.iteritems() if vals==max(tempCurrent.values())]
                action = np.random.choice(tempCurrent, 1)[0]
            else:
                #Pick an entirely random selection
                action = np.random.choice(self.env.valid_actions, 1)[0]
            #Decay the random exploration parameter
            self.randomExploration = max(self.randomExploration-self.randomExplorationDecay, 0)
            self.randomExplorationTrack[-1] = self.randomExploration

            # Execute action and get reward
            reward = self.env.act(self, action)
            # Get maximum reward from the future state
            tempNewState = self.env.sense(self)
            tempNewState = tuple([self.planner.next_waypoint()]) + tuple([str(x) for x in tempNewState.values()])
            #We are keeping track of total score and wrong moves, used for progress tracking
            self.cumScore[-1] += reward
            if reward < 0:
                self.wrongMoves[-1] += 1
            # TODO: Learn policy based on state, action, reward
            self.qEstimate[self.state][action] = self.learningRate*(reward + self.discountRate*max(self.getQEstimate(tempNewState).values())) + self.getQEstimate(self.state, action)*(1-self.learningRate)
            #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment(num_dummies=10)  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=10000)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
    #Export Q estimate to a data frame so we can start looking at some stats
    finQ1 = pd.DataFrame(a.qEstimate.keys())
    finQ1.columns=['waypoint'] + a.env.sense(a).keys()
    finQ2 = pd.DataFrame(a.qEstimate.values())
    finQ2.columns = ['act_' + str(x) for x in a.env.valid_actions]
    finQ = pd.concat([finQ1, finQ2], axis=1)

    #Print some stats to check some edge cases
    print(finQ)
    print(finQ[(finQ['waypoint']=='right') & (finQ['light']=='red')])
    print(finQ[(finQ['waypoint']=='left') & (finQ['light']=='green') & ((finQ['oncoming']=='forward') | (finQ['oncoming']=='right'))])
    
    #Makes some plots
    a.cumScore = pd.Series(a.cumScore[1:], name='Total Score')
    a.wrongMoves = pd.Series(a.wrongMoves[1:], name='Wrong Moves')
    a.completionTime = pd.Series(a.completionTime[1:], name='Completion Time')
    a.randomExplorationTrack = pd.Series(a.randomExplorationTrack[1:], name='Random Parameter')

    fig, axes = plt.subplots(4, 1)
    fig.canvas.set_window_title('Final Stats for Individual Trials')
    a.cumScore.plot(ax=axes[0], color='blue', legend=True)
    a.wrongMoves.plot(ax=axes[1], color='green', legend=True)
    a.completionTime.plot(ax=axes[2], color='red', legend=True)
    a.randomExplorationTrack.plot(ax=axes[3], color='black', legend=True)
    plt.show()

if __name__ == '__main__':
    run()
