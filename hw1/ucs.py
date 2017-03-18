from heapq import heappush, heappop, heapify


class Graph(object):

    def __init__(self):
        self._nodes = set()
        self._edges = {}

    def print__edges(self):
        print self._edges

    def print__nodes(self):
        print self._nodes

    def build_graph(self, input_file):
        with open(input_file, 'r') as input:
            for line in input:
                tuple = line.strip().split(' ')
                assert len(tuple) == 3
                self._nodes.add(tuple[0])
                self._nodes.add(tuple[1])
                # {a: [(distance, b), (distance, c), ...]}
                if tuple[0] not in self._edges.keys():
                    self._edges[tuple[0]] = [(int(tuple[2]), tuple[1])]
                else:
                    self._edges[tuple[0]].append((int(tuple[2]), tuple[1]))
        # Sort edges to break ties alphabetically
        for edge in self._edges.values():
            sorted(edge, key=lambda a: a[1])

    def ucs(self, start, goal):
        # Keep frontier as a heap using heapq, to make sure the top node of frontier has least distance
        frontier = [[0, start, start]]    # [[total_distance, node, solution], ...]
        explored = set()
        if start not in self._nodes or goal not in self._nodes:
            return 'Unreachable'
        while True:
            if not frontier:
                return 'Unreachable'
            total_distance, node, solution = heappop(frontier)    # [total_distance, a, solution]
            if node == goal:
                return solution
            explored.add(node)
            for distance, child in self._edges.get(node, []):
                index, old_distance = self.search_frontier(frontier, child)
                if child not in explored and index == -1:
                    # Push new child into frontier                    
                    heappush(frontier, [distance + total_distance, child, solution + '->' + child])
                elif old_distance > distance + total_distance:
                    # Update distance
                    self.update_frontier(frontier, child, distance + total_distance, solution + '->' + child)

    @staticmethod
    def search_frontier(frontier, item):
        for index, node in enumerate(frontier):
            if item == node[1]:
                # return index, length of edge
                return index, node[0]
        return -1, -1

    @staticmethod
    def update_frontier(frontier, item, distance, solution):
        for index, node in enumerate(frontier):
            if item == node[1]:
                frontier[index][0] = distance
                frontier[index][2] = solution
                heapify(frontier)
                return 0
        return 1

def main():
    graph = Graph()
    graph.build_graph('input.txt')
    solution = graph.ucs('Start', 'Goal')
    with open('output.txt', 'w') as output:
        output.write(solution)
    print solution

if __name__ == '__main__':
    main()
