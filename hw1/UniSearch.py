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

    def uni_search(self, start, goal):
        # frontier is a heap
        frontier = [[0, start]]
        explored = set()
        solution = []
        while True:
            if not frontier:
                return 'Not Found'
            total_distance, node = heappop(frontier)    # (distance, a)           
            solution.append(node)
            if node == goal:
                return '->'.join(solution)
            explored.add(node)
            for distance, child in self._edges[node]:
                if child not in explored and self.search_frontier(frontier, child) == -1:
                    heappush(frontier, [distance + total_distance, child])
                elif self.search_frontier(frontier, child) > distance + total_distance:
                    self.replace_frontier(frontier, child, distance + total_distance)

    @staticmethod
    def search_frontier(frontier, item):
        for node in frontier:
            if item == node[1]:
                return node[0]
        return -1

    @staticmethod
    def replace_frontier(frontier, item, distance):
        for index, node in enumerate(frontier):
            if item == node[1]:
                frontier[index][0] = distance
                heapify(frontier)
                return 0
        return 1

def main():
    graph = Graph()
    graph.build_graph('input.txt')
    solution = graph.uni_search('Start', 'Goal')
    with open('output.txt', 'w') as output:
        output.write(solution)
    print solution

if __name__ == '__main__':
    main()