# baselineTeam.py
# ---------------
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


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
import util
from game import Directions
from util import nearestPoint
from game import Actions


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveAStar', second='Defensive', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class BaseAgent(CaptureAgent):  # agente de base

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.getAgentPosition(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        self.midWidth = game_state.data.layout.width / 2
        self.height = game_state.data.layout.height

    def get_successor(self, gameState, action):
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos is not nearestPoint(pos):
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def dist_bound(self, game_state):
        #distance to closest boundary
        my_state = game_state.get_agent_state(self.index)
        myPosition = my_state.get_position()
        boundaries = []
        if self.red:
            i = self.midWidth - 1
        else:
            i = self.midWidth + 1
        boundaries = [(i, j) for j in range(self.height)]
        validPositions = []
        for i in boundaries:
            if not game_state.has_wall(i[0], i[1]):
                validPositions.append(i)
        return validPositions

    def ghost_dist(self, game_state):
        #distance of the closest ghost
        my_pos = game_state.getAgentState(self.index).getPosition()
        enemies = [game_state.getAgentState(i) for i in self.get_opponents(game_state)]
        ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]
        if len(ghosts) > 0:
            dists = [self.get_maze_distance(my_pos, a.getPosition()) for a in ghosts]
            return min(dists)
        else:
            return None

    def bounds(self, gameState):
        ''''
        return a list of positions of boundary
        '''
        my_state = gameState.getAgentState(self.index)
        my_position = my_state.getPosition()
        boundaries = []
        if self.red:
            i = self.midWidth - 1
        else:
            i = self.midWidth + 1
        boudaries = [(i, j) for j in range(self.height)]
        valid_positions = []
        for i in boudaries:
            if not gameState.hasWall(i[0], i[1]):
                valid_positions.append(i)
        return valid_positions

    def heuristic(self, state, game_state):
        #heuristic for avoiding ghosts
        heuristic = 0
        if self.ghost_dist(game_state) != None:
            enemies = [game_state.getAgentState(i) for i in self.get_opponents(game_state)]
            # pacmans = [a for a in enemies if a.isPacman]
            ghosts = [a for a in enemies if not a.isPacman and a.scaredTimer < 2 and a.getPosition() != None]
            if ghosts != None and len(ghosts) > 0:
                ghostpos = [ghost.getPosition() for ghost in ghosts]
                # pacmanPositions = [pacman.getPosition() for pacman in pacmans]
                ghostDists = [self.get_maze_distance(state, ghostposition) for ghostposition in ghostpos]
                ghostDist = min(ghostDists)
                if ghostDist < 3:
                    # print ghostDist
                    heuristic = pow((5 - ghostDist), 5)

        return heuristic

    def nullHeuristic(self, state, problem=None):
        return 0

    def aStarSearch(self, problem, gameState, heuristic=nullHeuristic):  # TODO change for our astar
        from util import PriorityQueue
        initial_state = problem.getStartState()
        visited = []
        frontier = PriorityQueue()
        h = heuristic(initial_state, gameState)
        g = 0
        f = g + h
        start_node = (initial_state, [], g)
        frontier.push(start_node, f)

        while not frontier.isEmpty():
            current_node = frontier.pop()
            state = current_node[0]
            path = current_node[1]
            current_cost = current_node[2]
            if state not in visited:
                visited.append(state)
                if problem.isGoalState(state):
                    return path
                successors = problem.getSuccessors(state)
                for successor in successors:
                    current_path = list(path)
                    successor_state = successor[0]
                    move = successor[1]
                    g = successor[2] + current_cost
                    h = heuristic(successor_state, gameState)
                    if successor_state not in visited:
                        current_path.append(move)
                        f = g + h
                        successor_node = (successor_state, current_path, g)
                        frontier.push(successor_node, f)
        return []


class OffensiveAStar(BaseAgent):  # versión que ataca
    #agent that seeks food

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.getAgentPosition(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        self.midWidth = game_state.data.layout.width / 2
        self.height = game_state.data.layout.height

    def get_successor(self, game_state, action):
        successor = game_state.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def getg_host_distance(self, gameState, index):
        myPosition = gameState.getAgentState(self.index).getPosition()
        ghost = gameState.getAgentState(index)
        dist = self.get_maze_distance(myPosition, ghost.getPosition())
        return dist

    def chooseAction(self, gameState):
        if len(self.get_food(gameState).asList()) < 3 or gameState.data.timeleft < self.distHome(gameState) + 60 \
                or gameState.getAgentState(self.index).numCarrying > 15:
            problem = find_way_back(gameState, self, self.index)
            if len(self.aStarSearch(problem, self.heuristic())) == 0:
                return 'Stop'
            else:
                return self.aStarSearch(problem, gameState, self.heuristic)[0]

        problem = search_food(gameState, self, self.index)
        return self.aStarSearch(problem, gameState, self.heuristic)[0]


class Defensive(BaseAgent):
   #agent that tries to defend our side
    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}

#####################
# Helping Functions #
#####################

class find_paths:

    def __init__(self, game_state, agent, agentIndex=0, costFn=lambda x: 1):
        self.walls = game_state.getWalls()
        self.costFn = costFn
        self.startState = game_state.getAgentState(agentIndex).getPosition()
        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):

        util.raiseNotDefined()

    def get_successors(self, state):
        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = state
            dx, dy = Actions.direction_to_vector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append((nextState, action, cost))

        # Bookkeeping for display purposes
        self._expanded += 1  # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def cost_actions(self, actions):
        if actions == None: return 999999
        x, y = self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.direction_to_vector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x, y))
        return cost


class search_food(find_paths):
    def __init__(self, game_state, agent, agentIndex=0):
        self.food = agent.getFood(game_state)
        # Store info for the PositionSearchProblem (no need to change this)
        self.startState = game_state.getAgentState(agentIndex).getPosition()
        self.walls = game_state.getWalls()
        self.foodLeft = len(self.food.asList())

    def isGoalState(self, state):
        # the goal state is the position of food or capsule
        # return state in self.food.asList() or state in self.capsule
        x, y = state
        return state in self.food.asList()


class find_way_back(find_paths):

    def __init__(self, gameState, agent, agentIndex=0):
        self.startState = gameState.get_agent_state(agentIndex).get_position()
        self.walls = gameState.get_walls()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0
        self.homeBoundary = agent.boundaryPosition(gameState)

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        return state in self.homeBoundary
