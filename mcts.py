# -*- coding:/ utf-8 -*-
"""
This piece of software is bound by The MIT License (MIT)
Code written by : Prashank Kadam
Email ID : prashankkadam07@gmail.com
Created on - Jan 14 2020
version : 1.0
"""
# Minimal Implementation of Monte Carlo Tree Search in Python3

# Implementing the required libraries:
from abc import ABC, abstractmethod
from _collections import  defaultdict
import math

class MCTS:
    "Monte Carlo tree searcher. First rollout the tree then choose a move"

    def __init__(self, exploration_weight=1):
        self.Q = defaultdict(int)  # total reward for each node
        self.N = defaultdict(int)  # total visit count for each node
        self.children = dict()  # children of each node
        self.exploration_weight = exploration_weight


    def choose(self, node):
        "Choose the best successor of node (Choosing the next move in the game)"
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")

        if node not in self.children:
            return node.find_random_child()

        def score(n):
            if self.N[n] == 0:
                return float("-inf")  # avoid unseen moves
            return self.Q[n] / self.N[n]

        return max(self.children[node], key=score)


    def do_rollout(self, node):
        "Make the tree one layer better. (Train for one iteration)"
        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        reward = self._simulate(leaf)
        self._backpropagate(path, reward)


    def _select(self, node):
        "Find the unexplored descendent of the node"
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                "The node is either unexplored ot terminal"
                return path
            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path
            node = self._uct_select(node)    # descend a layer deeper


    def _expand(self, node):
        "Update the children dict with children of the node"
        if node in self.children:
            return             # Already expanded
        self.children[node] = node.find_children()


    def _simulate(self, node):
        "Return the reward for a random simulation completion of a node"
        invert_reward = True
        while True:
            if node.is_terminal():
                reward = node.reward()
                return 1 - reward if invert_reward else reward
            node = node.find_random_child()
            invert_reward = not invert_reward


    def _backpropagate(self, path, reward):
        "Send the reward back up to the ancestors of the leaf"
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward
            reward = 1 - reward  # 1 for me is 0 for my enemy


    def uct_select(self, node):
        "Select a child of the node balancing exploration and exploitation"

        # All the children of the node should already be expanded
        assert all(n in self.children for n in self.children[node])

        log_N_vertex = math.log(self.N[node])

        def uct(n):
            "Upper confidence bound for the trees"
            return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(
                log_N_vertex / self.N[n]
            )

        return max(self.children[node], key=uct)


class Node(ABC):
    """
    A representation of a single board state.
    MCTS works by constructing a tree of these Nodes.
    Could be e.g. a chess or checkers board state.
    """

    @abstractmethod
    def find_children(self):
        "All the possible successors to the board state"
        return set()

    @abstractmethod
    def find_random_child(self):
        "Random successor of this board state (for efficient simulation)"
        return None

    @abstractmethod
    def is_terminal(self):
        "Returns true if the node has no children"
        return True

    @abstractmethod
    def reward(self):
        "Assumes self is a terminal node. 1-win, 0-loss, 0.5-tie, etc"
        return 0

    @abstractmethod
    def __hash__(self):
        "Nodes must be hashable"
        return 123456789

    @abstractmethod
    def __eq__(node1, node2):
        "Nodes must be comparable"
        return True