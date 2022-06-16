
from __future__ import division
from __future__ import print_function

import sys
import math
import time
import queue as Q
import psutil
from collections import deque

#### SKELETON CODE ####
## The Class that Represents the Puzzle
class PuzzleState(object):
    """
        The PuzzleState stores a board configuration and implements
        movement instructions to generate valid children.
    """
    def __init__(self, config, n, parent=None, action="Initial", cost=0):
        """
        :param config->List : Represents the n*n board, for e.g. [0,1,2,3,4,5,6,7,8] represents the goal state.
        :param n->int : Size of the board
        :param parent->PuzzleState
        :param action->string
        :param cost->int
        """
        if n*n != len(config) or n < 2:
            raise Exception("The length of config is not correct!")
        if set(config) != set(range(n*n)):
            raise Exception("Config contains invalid/duplicate entries : ", config)

        self.n        = n
        self.cost     = cost
        self.parent   = parent
        self.action   = action
        self.config   = config
        self.children = []

        # Get the index and (row, col) of empty block
        self.blank_index = self.config.index(0)

    def display(self):
        """ Display this Puzzle state as a n*n board """
        for i in range(self.n):
            print(self.config[3*i : 3*(i+1)])

    def move_up(self):
        """ 
        Moves the blank tile one row up.
        :return a PuzzleState with the new configuration
        """
        temp_state = self.config[:]
        if self.blank_index not in range(0, self.n):
            temp = temp_state[self.blank_index - self.n]
            temp_state[self.blank_index - self.n] = temp_state[self.blank_index]
            temp_state[self.blank_index] = temp
            res = PuzzleState(temp_state, self.n, parent = self, action = "Up", cost = self.cost+1)
            return res
        else:
            return None
      
    def move_down(self):
        """
        Moves the blank tile one row down.
        :return a PuzzleState with the new configuration
        """
        temp_state = self.config[:]
        board_len = self.n*self.n
        if self.blank_index not in range(board_len-self.n, board_len):
            temp = temp_state[self.blank_index + self.n]
            temp_state[self.blank_index + self.n] = temp_state[self.blank_index]
            temp_state[self.blank_index] = temp
            res = PuzzleState(temp_state, self.n, parent = self, action = "Down", cost = self.cost+1)
            return res
        else:
            return None
      
    def move_left(self):
        """
        Moves the blank tile one column to the left.
        :return a PuzzleState with the new configuration
        """
        temp_state = self.config[:]
        board_len = self.n*self.n
        if self.blank_index not in (0, self.n, board_len-self.n):
            temp = temp_state[self.blank_index - 1]
            temp_state[self.blank_index - 1] = temp_state[self.blank_index]
            temp_state[self.blank_index] = temp
            res = PuzzleState(temp_state, self.n, parent = self, action = "Left", cost = self.cost+1)
            return res
        else:
            return None

    def move_right(self):
        """
        Moves the blank tile one column to the right.
        :return a PuzzleState with the new configuration
        """
        temp_state = self.config[:]
        board_len = self.n*self.n
        if self.blank_index not in (self.n-1, board_len-self.n-1, board_len-1):
            temp = temp_state[self.blank_index + 1]
            temp_state[self.blank_index + 1] = temp_state[self.blank_index]
            temp_state[self.blank_index] = temp
            res = PuzzleState(temp_state, self.n, parent = self, action = "Right", cost = self.cost+1)
            return res
        else:
            return None
      
    def expand(self):
        """ Generate the child nodes of this node """
        
        # Node has already been expanded
        if len(self.children) != 0:
            return self.children
        
        # Add child nodes in order of UDLR
        children = [
            self.move_up(),
            self.move_down(),
            self.move_left(),
            self.move_right()]

        # Compose self.children of all non-None children states
        self.children = [state for state in children if state is not None]
        return self.children

class PriorityQueue(object):
    def __init__(self):
        self.queue = [] 

    def put(self, x):
        self.queue.append(x)

    def get(self):
        tot_cost = 999999999999999
        for i in range(len(self.queue)):
            item = self.queue[i]
            if  item[0] < tot_cost:
                tot_cost = item[0]
                priority = i
        item = self.queue[priority] 
        del self.queue[priority] 
        return item[1]

    
moves = list()
max_search_depth = 0
num_nodes_expanded = 0
def backtrack(start_state, final_state):
    cur = final_state
    while start_state.config != cur.config:
        moves.insert(0, cur.action)
        cur = cur.parent
    return moves        
    
# Function that Writes to output.txt

### Students need to change the method to have the corresponding parameters
def writeOutput(start_state, final, start, end):
    ### Student Code Goes here
    global moves
    moves = backtrack(start_state, final)
    print("HELLO")
    file = open('./output.txt', 'w')
    file.write("path_to_goal: " + str(moves))
    file.write("\ncost_of_path: " + str(final.cost))
    file.write("\nnodes_expanded: " + str(num_nodes_expanded))
    file.write("\nsearch_depth: " + str(final.cost))
    file.write("\nmax_search_depth: " + str(max_search_depth))
    file.write("\nrunning_time: " + format(end-start, '.8f'))
    file.write("\nmax_ram_usage: " + format(psutil.virtual_memory().percent/1000.0, '.8f'))    
    file.close()

def bfs_search(initial_state):
    """BFS search"""
    ### STUDENT CODE GOES HERE ###
    global max_search_depth, num_nodes_expanded
    visited = set()
    frontier = deque([initial_state])
    while frontier:
        state = frontier.popleft()
        state_string = ''.join(str(s) for s in state.config)
        visited.add(state_string)
        if test_goal(state.config):
            return state
        successors = state.expand()
        num_nodes_expanded+=1
        for s in successors:
            state_string = ''.join(str(i) for i in s.config)
            if state_string not in visited:
                frontier.append(s)
                visited.add(state_string)
                if s.cost > max_search_depth:
                    max_search_depth += 1

def dfs_search(initial_state):
    """DFS search"""
    ### STUDENT CODE GOES HERE ###
    global max_search_depth, num_nodes_expanded
    visited = set()
    frontier = list([initial_state])
    while frontier:
        state = frontier.pop()
        state_string = ''.join(str(s) for s in state.config)
        if test_goal(state.config):
            return state
        successors = reversed(state.expand())
        num_nodes_expanded+=1
        for s in successors:
            state_string = ''.join(str(i) for i in s.config)
            if state_string not in visited:
                frontier.append(s)
                visited.add(state_string)
                if s.cost > max_search_depth:
                    max_search_depth += 1

def A_star_search(initial_state):
    """A * search"""
    ### STUDENT CODE GOES HERE ###
    global max_search_depth, num_nodes_expanded
    visited = set()
    total_cost = calculate_total_cost(initial_state)
    frontier = PriorityQueue()
    #Enqueuing on the basis of total cost 
    frontier.put((total_cost, initial_state))
    while frontier:
        state = frontier.get()
        state_string = ''.join(str(s) for s in state.config)
        visited.add(state_string)       
        if test_goal(state.config):
            return state
        successors = state.expand()
        num_nodes_expanded+=1
        for s in successors:
            state_string = ''.join(str(i) for i in s.config)
            if state_string not in visited:
                total_cost = calculate_total_cost(s)
                frontier.put((total_cost, s))
                if s.cost > max_search_depth:
                    max_search_depth += 1
        
def calculate_total_cost(state):
    """calculate the total estimated cost of a state"""
    ### STUDENT CODE GOES HERE ###
    heuristic = 0
    for i in range(1 , state.n*state.n):
        heuristic+= calculate_manhattan_dist(state.config.index(i), i, state.n)    
    return(state.cost + heuristic)

def calculate_manhattan_dist(idx, value, n):
    """calculate the manhattan distance of a tile"""
    ### STUDENT CODE GOES HERE ###
    goal_state = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    distance = abs(idx - goal_state.index(value))
    return (distance/n + distance%n)

def test_goal(puzzle_state):
    """test the state is the goal state or not"""
    ### STUDENT CODE GOES HERE ###
    goal_state = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    if puzzle_state == goal_state:
        return 1
    return 0

# Main Function that reads in Input and Runs corresponding Algorithm
def main():
    search_mode = sys.argv[1].lower()
    begin_state = sys.argv[2].split(",")
    begin_state = list(map(int, begin_state))
    board_size  = int(math.sqrt(len(begin_state)))
    hard_state  = PuzzleState(begin_state, board_size)
    start_time  = time.time()
    if search_mode == "bfs":
        final = bfs_search(hard_state)
        end_time = time.time()
        writeOutput(hard_state, final, start_time, end_time)
    elif search_mode == "dfs":
        final = dfs_search(hard_state)
        end_time = time.time()
        writeOutput(hard_state, final, start_time, end_time)
    elif search_mode == "ast":
        final = A_star_search(hard_state)
        end_time = time.time()
        writeOutput(hard_state, final, start_time, end_time)
    else: 
        print("Enter valid command arguments !")       

    print("Program completed in %.3f second(s)"%(end_time-start_time))

if __name__ == '__main__':
    main()