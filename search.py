# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter 
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html

"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util, queue

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def expand(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (child,
        action, stepCost), where 'child' is a child to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that child.
        """
        util.raiseNotDefined()

    def getActions(self, state):
        """
          state: Search state

        For a given state, this should return a list of possible actions.
        """
        util.raiseNotDefined()

    def getActionCost(self, state, action, next_state):
        """
          state: Search state
          action: action taken at state.
          next_state: next Search state after taking action.

        For a given state, this should return the cost of the (s, a, s') transition.
        """
        util.raiseNotDefined()

    def getNextState(self, state, action):
        """
          state: Search state
          action: action taken at state

        For a given state, this should return the next state after taking action from state.
        """
        util.raiseNotDefined()

    def getCostOfActionSequence(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"

    visited = {}
    parents = {}
    solution = []
    stack = util.Stack()

    start = problem.getStartState()
    stack.push((start, 'Undefined', 0))
    visited[start] = 'Undefined'
    print(visited[start])
    if problem.isGoalState(start):
        return solution
    
    goal = False

    while not stack.isEmpty() and not goal:
        node = stack.pop()
        print(node)
        visited[node[0]] = node[1]
        print(node[0])
        print(node[1])
        print(visited[node[0]])
        if problem.isGoalState(node[0]):
            node_sol = node[0]
            goal = True
            break
        for child in problem.expand(node[0]):
            if child[0] not in visited:
                parents[child[0]] = node[0]
                print(child)
                stack.push(child)
    
    while node_sol in parents.keys():
        node_sol_prev = parents[node_sol]
        print(parents[node_sol])
        solution.insert(0, visited[node_sol])
        node_sol = node_sol_prev
        print(parents)

    return solution

    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    "*** YOUR CODE HERE ***"
    visited = {}
    parents = {}
    solution = []
    queue = util.Queue()

    start = problem.getStartState()
    queue.push((start, 'Undefined', 0))
    visited[start] = 'Undefined'
    if problem.isGoalState(start):
        return solution
    
    goal = False

    while not queue.isEmpty() and not goal:
        node = queue.pop()
        visited[node[0]] = node[1]
        if problem.isGoalState(node[0]):
            node_sol = node[0]
            goal = True
            break
        for neighbor in problem.expand(node[0]):
            if neighbor[0] not in visited:
                visited[neighbor[0]] = 'Undefined'
                parents[neighbor[0]] = node[0]
                queue.push(neighbor)
    
    while node_sol in parents:
        node_sol_prev = parents[node_sol]
        solution.insert(0, visited[node_sol])
        node_sol = node_sol_prev

    return solution

    util.raiseNotDefined()

def uniformCostSearch(problem):
    "Search the node of least total cost first. "
    "*** YOUR CODE HERE ***"

    visited = {}
    parents = {}
    solution = []
    queue = util.PriorityQueue()
    cost = {}

    start = problem.getStartState()
    queue.push((start, 'Undefined', 0), 0)
    visited[start] = 'Undefined'
    cost[start] = 0
    if problem.isGoalState(start):
        return solution
    
    goal = False

    while (queue.isEmpty() != True and goal != True):
        node = queue.pop()
        visited[node[0]] = node[1]
        if problem.isGoalState(node[0]):
            node_sol = node[0]
            goal = True
            break
        for child in problem.expand(node[0]):
            if child[0] not in visited.keys():
                priority = node[2] + child[2]
                if child[0] in cost.keys():
                    if cost[child[0]] <= priority:
                        continue #if the cost which is lesser than the current priority is already in the dictionary, it skips it and search for new ones
                queue.push((child[0], child[1], priority), priority)
                cost[child[0]] = priority
                parents[child[0]] = node[0]
    
    while node_sol in parents.keys():
        node_sol_prev = parents[node_sol]
        solution.insert(0, visited[node_sol])
        node_sol = node_sol_prev

    return solution

    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    "*** YOUR CODE HERE ***"

    visited = {}
    parents = {}
    solution = []
    queue = util.PriorityQueue()
    cost = {}

    start = problem.getStartState()
    queue.push((start, 'Undefined', 0), 0)
    visited[start] = 'Undefined'
    cost[start] = 0
    if problem.isGoalState(start):
        return solution
    
    goal = False

    while (queue.isEmpty() != True and goal != True):
        node = queue.pop()
        #print(node)
        visited[node[0]] = node[1]
        # print("node[0]: ", node[0])
        # print("node[1]: ", node[1])
        if problem.isGoalState(node[0]):
            node_sol = node[0]
            goal = True
            break
        for child in problem.expand(node[0]):
            if child[0] not in visited.keys():
                priority = node[2] + child[2] + heuristic(child[0], problem)
                #print("node[2]: ", node[2])
                #print("child[2]: ", child[2])
                #print("heuristic(child[0], problem): ", heuristic(child[0], problem))
                #print("Cost: ", cost)
                if child[0] in cost.keys():
                    if cost[child[0]] <= priority:
                        #print("cost[child[0]]: ", cost[child[0]])
                        #print("priority: ", priority)
                        continue
                queue.push((child[0], child[1], node[2] + child[2]), priority)
                cost[child[0]] = priority
                parents[child[0]] = node[0]
    
    while node_sol in parents.keys():
        node_sol_prev = parents[node_sol]
        solution.insert(0, visited[node_sol])
        node_sol = node_sol_prev

    return solution

    util.raiseNotDefined()

def Dijkstra(problem):
    visited = {}
    queue = util.PriorityQueue()
    parents = {}
    solution = []
    cost = {}

    start = problem.getStartState()
    queue.push((start, 'Undefined', 0),0)
    visited[start] = 'Undefined'
    cost[start] = 0
    if problem.isGoalState(start):
        return solution
    
    goal = False
    
    while (queue.isEmpty() != True and goal != True):
        node = queue.pop()
        visited[node[0]] = node[1]
        if problem.isGoalState(node[0]):
            node_sol = node[0]
            goal = True
            break

        for child in problem.expand(node[0]):
            priority = node[2] + child[2]
            if child[0] not in cost or priority < cost[child[0]]:
                cost[child[0]] = priority
                parents[child[0]] = node[0]
                queue.push((child[0], child[1], child[2]), priority)
    
    while node_sol in parents:
        solution.insert(0, visited[node_sol])
        node_sol = parents[node_sol]

    return solution

def greedy(problem):
    visited = {}
    queue = util.PriorityQueue()
    parents = {}
    solution = []
    cost = {}

    start = problem.getStartState()
    queue.push((start, 'Undefined', 0),0)
    visited[start] = 'Undefined'
    cost[start] = 0
    if problem.isGoalState(start):
        return solution
    
    goal = False
    
    while (queue.isEmpty() != True and goal != True):
        node = queue.pop()
        visited[node[0]] = node[1]
        if problem.isGoalState(node[0]):
            node_sol = node[0]
            goal = True
            break

        for child in problem.expand(node[0]):
            priority = child[2]
            if child[0] not in cost or priority < cost[child[0]]:
                cost[child[0]] = priority
                parents[child[0]] = node[0]
                queue.push(child, priority)
    
    while node_sol in parents:
        solution.insert(0, visited[node_sol])
        node_sol = parents[node_sol]

    return solution

def depthLimitedSearch(problem, limit = 130
                       ):
    visited = {}
    parents = {}
    solution = []
    stack = util.Stack()

    start = problem.getStartState()
    stack.push((start, 'Undefined', 0))
    visited[start] = 'Undefined'
    if problem.isGoalState(start):
        return solution
    
    goal = False
    node_sol = None

    while not stack.isEmpty() and not goal:
        node = stack.pop()
        visited[node[0]] = node[1]
        if problem.isGoalState(node[0]):
            node_sol = node[0]
            goal = True
            break
        if node[2] < limit:
            for child in problem.expand(node[0]):
                if child[0] not in visited:
                    parents[child[0]] = node[0]
                    stack.push((child[0], child[1], node[2] + 1))
    
    while node_sol in parents:
        node_sol_prev = parents[node_sol]
        solution.insert(0, visited[node_sol])
        node_sol = node_sol_prev

    return solution

def iterativeDeepeningSearch(problem):
    depth = 0
    while True:
        result = depthLimitedSearch(problem, depth)
        if result:
            return result
        depth += 1


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
dijk = Dijkstra
grd = greedy
dls = depthLimitedSearch
ids = iterativeDeepeningSearch
