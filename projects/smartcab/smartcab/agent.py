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
        self.learningRate = 0.2
        self.discountRate = 0.2
        self.cumScore = [0]
        self.completionTime = [0]
        self.wrongMoves = [0]

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        print(str(self.cumScore[-1]) + '|' + str(self.completionTime[-1]) + '|' + str(self.wrongMoves[-1]))
        #for k, v in self.qEstimate.iteritems():
        #    print(str(k) + '|' + str(v))
        #print(self.getQEstimate(inputState=('right','red',None,'forward',None)))
        #print(self.getQEstimate(inputState=('left','green','forward',None,None)))
        #print(self.getQEstimate(inputState=('left','green','right',None,None)))
        self.cumScore.append(0)
        self.completionTime.append(0)
        self.wrongMoves.append(0)


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
        #print(inputs['left'])
        self.completionTime[-1] = t
        # TODO: Update state
        if not self.next_waypoint is None:
            self.state = tuple([self.next_waypoint]) + tuple([str(x) for x in inputs.values()])
            # TODO: Select action according to your policy
            tempCurrent = self.getQEstimate(self.state)
            tempCurrent = [act for act, vals in tempCurrent.iteritems() if vals==max(tempCurrent.values())]
            action = np.random.choice(tempCurrent, 1)[0]
            #action = self.next_waypoint


            # Execute action and get reward
            reward = self.env.act(self, action)
            tempNewState = self.env.sense(self)
            tempNewState = tuple([self.planner.next_waypoint()]) + tuple([str(x) for x in tempNewState.values()])
            self.cumScore[-1] += reward
            if reward < 0:
                self.wrongMoves[-1] += 1
            # TODO: Learn policy based on state, action, reward
            self.qEstimate[self.state][action] = self.learningRate*(reward + self.discountRate*max(self.getQEstimate(tempNewState).values())) + self.getQEstimate(self.state, action)*(1-self.learningRate)
            #print(str(self.cumScore) + '|' + str(self.qEstimate[self.state]))
            #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment(num_dummies=20)  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
    #Export Q estimate to multiindex dataframe for closer examination
    finQ = pd.DataFrame(a.qEstimate.values(), pd.MultiIndex.from_tuples(a.qEstimate.keys(), names=['waypoint'] + a.env.sense(a).keys()))
    finQ.sort_index(level=1)
    print(finQ.loc['right'].loc['red'])
    print(finQ.loc['left'].loc['green'])
    #plt.show()

if __name__ == '__main__':
    run()
