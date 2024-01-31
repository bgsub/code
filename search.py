# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

from custom_types import Direction
from game import Directions
from pacman import GameState
from typing import Any, Literal, Tuple,List
import util

"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self)->Any:
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state:Any)->bool:
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state:Any)->List[Tuple[Any,Direction,int]]:
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions:List[Direction])->int:
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()



def tinyMazeSearch(problem:SearchProblem)->List[Direction]:
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem:SearchProblem)->List[Direction]:
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    
    
    starting_state = problem.getStartState()
    visited = []   # we keep track of the visited states because we are using graph search method
    path = [] #  # we keep track of the path to the node we are visiting
    state_stack = util.Stack()  # a LIFO policy container used to store a state and the path to get to that state
    state_stack.push((starting_state, path))
    while not state_stack.isEmpty():
        state, path = state_stack.pop()
        
        if problem.isGoalState(state):
            return path
        else:
            for (next_state, next_action, cost) in problem.getSuccessors(state):
                if(next_state not in visited):
                    state_stack.push((next_state, path + [next_action])) # we save the unvisited new state and the path to get to it
            visited.append(state)
    return []

def breadthFirstSearch(problem:SearchProblem)->List[Direction]:
    """Search the shallowest nodes in the search tree first."""

    starting_state = problem.getStartState()
    visited = []   # we keep track of the visited states because we are using graph search method
    path = []   # we keep track of the path to the node we are visiting
    state_queue = util.Queue()  # a FIFO policy container used to store a state and the path to get to that state
    state_queue.push((starting_state, path))
    visited.append(starting_state) # unlike in BFS, we first need to visit the start state  
    while not state_queue.isEmpty():
        state, path = state_queue.pop()
        
        if problem.isGoalState(state):
            return path
        else:
            for (next_state, next_action, cost) in problem.getSuccessors(state):
                if(next_state not in visited):
                    visited.append(next_state) # unlike in BFS, we visit a state when we discover it   
                    state_queue.push((next_state, path + [next_action])) # we save the next state and the path to get to it
            
    return []

def uniformCostSearch(problem:SearchProblem)->List[Direction]:
    """Search the node of least total cost first."""
    
    starting_state = problem.getStartState()
    visited = []   # we keep track of the visited states because we are using graph search method
    path = []   # we keep track of the path to the node we are visiting
    cost_to_state = 0
    state_queue = util.PriorityQueue()  # a FIFO policy container used to store a state and the path to get to that state
    state_queue.push((starting_state, path, cost_to_state), cost_to_state)
    visited.append(starting_state) 
    while not state_queue.isEmpty():
        state, path, cost_to_state = state_queue.pop()
        
        if problem.isGoalState(state):
            visited.append(state)
            return path
        else:
            for (next_state, next_action, cost) in problem.getSuccessors(state):
                if(next_state not in visited):
                    state_queue.push((next_state, path + [next_action], cost_to_state + cost), cost_to_state + cost) # we save the next state and the path to get to it
                    if not problem.isGoalState(next_state): 
                        visited.append(next_state)  # We make sure to not visit the goalState if we are not at the goal state.
    return []

def nullHeuristic(state:GameState, problem:SearchProblem=None)->List[Direction]:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem:SearchProblem, heuristic=nullHeuristic)->List[Direction]:
    """Search the node that has the lowest combined cost and heuristic first."""
    starting_state = problem.getStartState()
    visited = []   # We keep track of the visited states because we are using graph search method.
    path = []   # We keep track of the path to the node we are visiting.
    cost_to_state = 0
    state_queue = util.PriorityQueue()  # A priority queue policy container used to store states and the path and priority-cost for each states.
    priority_function = heuristic(starting_state, problem)
    state_queue.push((starting_state, path, cost_to_state), cost_to_state + priority_function)
    visited.append(starting_state) 

    while not state_queue.isEmpty():
        state, path, cost_to_state = state_queue.pop()
        
        if problem.isGoalState(state):
            return path
        else:
            for (next_state, next_action, cost) in problem.getSuccessors(state):
                if(next_state not in visited):
                    priority_function = heuristic(next_state, problem)
                    state_queue.push((next_state, path + [next_action], cost_to_state + cost), cost_to_state + cost + priority_function) # we save the next state and the path to get to it
                    if not problem.isGoalState(next_state):
                        visited.append(next_state) # We make sure to not visit the goalState if we are not at the goal state  
            if(problem.isGoalState(next_state)): visited.append(state)
    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
