# N-puzzle-game

## Introduction
The N-puzzle game consists of a board holding N = m2 − 1 distinct movable tiles, plus one empty space. There is
one tile for each number in the set {0, 1,..., m2 − 1}. In this assignment, we will represent the blank space with the
number 0 and focus on the m = 3 case (8-puzzle).

In this combinatorial search problem, the aim is to get from any initial board state to the configuration with all
tiles arranged in ascending order {0, 1,..., m2 − 1} – this is your goal state. The search space is the set of all possible
states reachable from the initial state. Each move consists of swapping the empty space with a component in one of
the four directions {‘Up’, ‘Down’, ‘Left’, ‘Right’}. Give each move a cost of one. Thus, the total cost of a path will
be equal to the number of moves made.

## Running the program
```python3 puzzle.py <method> <board>```
where <method> can be:
  1. bfs (Breadth-First Search)
  2. dfs (Depth-First Search)
  3. ast (A-Star Search)
  
## What the Program Outputs
The program will create and/or write to a file called output.txt, containing the following statistics:
  1. path to goal: the sequence of moves taken to reach the goal
  2. cost of path: the number of moves taken to reach the goal
  3. nodes expanded: the number of nodes that have been expanded
  4. search depth: the depth within the search tree when the goal node is found
  5. max search depth: the maximum depth of the search tree in the lifetime of the algorithm
  6. running time: the total running time of the search instance, reported in seconds
  7. max ram usage: the maximum RAM usage in the lifetime of the process as measured by the ru maxrss attribute
in the resource module, reported in megabytes
