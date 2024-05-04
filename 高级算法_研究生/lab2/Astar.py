import tkinter as tk
import heapq
from math import sqrt
import time


class Node:
    def __init__(self, x, y, g=float("inf"), h=0, f=None, parent=None):
        self.x = x
        self.y = y
        self.g = g
        self.h = h
        self.f = f if f is not None else g + h
        self.parent = parent

    def __lt__(self, other):
        return self.f < other.f

    def __eq__(self, value: object) -> bool:
        return self.x == value.x and self.y == value.y


class AStar:
    def __init__(self, grid, start, goal):
        self.grid = grid
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.open_set = []
        self.closed_set = set()
        self.nodes = [[None for _ in range(len(grid[0]))] for _ in range(len(grid))]
        self.initialize_nodes()

        # Visualization attributes
        self.window = tk.Tk()
        self.window.title("A* Pathfinding Visualization")
        self.cell_size = 50
        self.grid_visual = None

        self.nodes_reverse = [
            [None for _ in range(len(grid[0]))] for _ in range(len(grid))
        ]
        self.initialize_nodes_reverse()

        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        max_grid_width = len(grid[0]) * self.cell_size
        max_grid_height = len(grid) * self.cell_size

        scale_width = (
            screen_width / max_grid_width if max_grid_width > screen_width else 1
        )
        scale_height = (
            screen_height / max_grid_height if max_grid_height > screen_height else 1
        )
        scale_factor = min(scale_width, scale_height)
        self.cell_size = int(self.cell_size * scale_factor)
        self.canvas = tk.Canvas(
            self.window, scrollregion=(0, 0, max_grid_width, max_grid_height)
        )
        vbar = tk.Scrollbar(self.window, orient=tk.VERTICAL)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)
        vbar.config(command=self.canvas.yview)
        hbar = tk.Scrollbar(self.window, orient=tk.HORIZONTAL)
        hbar.pack(side=tk.BOTTOM, fill=tk.X)
        hbar.config(command=self.canvas.xview)
        self.canvas.config(width=screen_width * 0.8, height=screen_height * 0.8)
        self.canvas.config(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        self.canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

    def initialize_nodes_reverse(self):
        for i, row in enumerate(self.grid):
            for j, cell in enumerate(row):
                if cell != "#":
                    self.nodes_reverse[i][j] = Node(
                        j, i, g=float("inf"), h=self.heuristic(j, i, self.start)
                    )
                else:
                    self.nodes_reverse[i][j] = None

    def initialize_nodes(self):
        for i, row in enumerate(self.grid):
            for j, cell in enumerate(row):
                if cell != "#":
                    self.nodes[i][j] = Node(
                        j, i, g=float("inf"), h=self.heuristic(j, i)
                    )
                else:
                    self.nodes[i][j] = None

    def heuristic(self, x, y, target=None):
        if target is None:
            target = self.goal
        dx = abs(x - target.x)
        dy = abs(y - target.y)
        return min(dx, dy) * sqrt(2) + abs(dx - dy) + int(self.grid[y][x])

    def get_neighbors(self, node):
        neighbors = []
        directions = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (1, 1),
            (-1, 1),
            (1, -1),
        ]
        for dx, dy in directions:
            x2, y2 = node.x + dx, node.y + dy
            if (
                0 <= x2 < len(self.grid[0])
                and 0 <= y2 < len(self.grid)
                and self.grid[y2][x2] != "#"
            ):
                neighbors.append(self.nodes[y2][x2])
        return neighbors

    def search(self):
        start_node = self.nodes[self.start.y][self.start.x]
        start_node.g = 0
        start_node.h = self.heuristic(start_node.x, start_node.y)
        start_node.f = start_node.g + start_node.h
        heapq.heappush(self.open_set, (start_node.f, start_node))
        self.update_grid_visual(start_node, color="light green")

        while self.open_set:
            time.sleep(0.02)
            _, current_node = heapq.heappop(self.open_set)
            self.update_grid_visual(current_node, color="light blue")

            if current_node == self.nodes[self.goal.y][self.goal.x]:
                # Visualize the final path
                self.reconstruct_path_visual(current_node)
                return self.reconstruct_path(current_node), current_node.g

            self.closed_set.add((current_node.x, current_node.y))
            for neighbor in self.get_neighbors(current_node):
                if (neighbor.x, neighbor.y) in self.closed_set:
                    continue
                self.process_neighbor_node(
                    current_node, neighbor, self.open_set, self.closed_set
                )

        return [], float("inf")  # No path found

    def search_bi_directional(self):
        closed_set = set()
        closed_set_reverse = set()

        while self.open_set and self.open_set_reverse:
            # Forward search step
            time.sleep(0.02)
            _, current_node = heapq.heappop(self.open_set)
            closed_set.add((current_node.x, current_node.y))
            # Visualize current node in forward search
            self.update_grid_visual(current_node, color="cyan")
            if (current_node.x, current_node.y) in closed_set_reverse:
                return (
                    self.reconstruct_bi_path(current_node, True),
                    current_node.g
                    + self.nodes_reverse[current_node.y][current_node.x].g,
                )

            for neighbor in self.get_neighbors(current_node):
                if (neighbor.x, neighbor.y) in closed_set:
                    continue
                self.process_neighbor_node(
                    current_node, neighbor, self.open_set, closed_set
                )

            # Reverse search step
            _, current_node_reverse = heapq.heappop(self.open_set_reverse)
            closed_set_reverse.add((current_node_reverse.x, current_node_reverse.y))
            # Visualize current node in reverse search
            self.update_grid_visual(current_node_reverse, color="magenta")
            if (current_node_reverse.x, current_node_reverse.y) in closed_set:
                return (
                    self.reconstruct_bi_path(current_node_reverse, False),
                    current_node_reverse.g
                    + self.nodes[current_node_reverse.y][current_node_reverse.x].g,
                )

            for neighbor in self.get_neighbors_reverse(current_node_reverse):
                if (neighbor.x, neighbor.y) in closed_set_reverse:
                    continue
                self.process_neighbor_node(
                    current_node_reverse,
                    neighbor,
                    self.open_set_reverse,
                    closed_set_reverse,
                )

        return None, float("inf")  # No path found

    def move_cost(self, current, neighbor):

        if current.x != neighbor.x and current.y != neighbor.y:
            return sqrt(2) + int(self.grid[neighbor.y][neighbor.x])
        else:
            return 1 + int(self.grid[neighbor.y][neighbor.x])

    def reconstruct_path_visual(self, current_node):
        if not isinstance(current_node, list):
            while current_node:
                self.update_grid_visual(current_node, color="red")
                current_node = current_node.parent
                self.window.after(50)
                self.window.update()
        else:
            for x, y in current_node:
                print(" -> ", x, y)
                self.update_grid_visual(Node(x, y), color="red")

    def reconstruct_path(self, current_node):
        path = []
        while current_node:
            path.append((current_node.x, current_node.y))
            current_node = current_node.parent
        return path[::-1]

    def create_grid_visual(self):
        self.grid_visual = {}
        for i, row in enumerate(self.grid):
            for j, cell in enumerate(row):
                x1, y1 = j * self.cell_size, i * self.cell_size
                x2, y2 = x1 + self.cell_size, y1 + self.cell_size
                color = "white"  # Default color for normal cells
                if cell == "S":  # Start
                    color = "green"
                elif cell == "T":  # Target
                    color = "red"
                elif cell == "#":  # Obstacle
                    color = "black"
                elif cell == "4":  # Sand
                    color = "yellow"
                elif cell == "2":  # Stream
                    color = "blue"
                cell_id = self.canvas.create_rectangle(
                    x1, y1, x2, y2, fill=color, outline="gray"
                )
                self.grid_visual[(i, j)] = cell_id

        x1, y1 = self.start.x * self.cell_size, self.start.y * self.cell_size
        x2, y2 = x1 + self.cell_size, y1 + self.cell_size
        self.grid_visual[(self.start.y, self.start.x)] = self.canvas.create_rectangle(
            x1, y1, x2, y2, fill="red", outline="gray"
        )
        x1, y1 = self.goal.x * self.cell_size, self.goal.y * self.cell_size
        x2, y2 = x1 + self.cell_size, y1 + self.cell_size
        self.grid_visual[(self.goal.y, self.goal.x)] = self.canvas.create_rectangle(
            x1, y1, x2, y2, fill="red", outline="gray"
        )

    def update_grid_visual(self, node, color="orange"):
        # Update the color of a node in the visual grid to show exploration
        if (node.y, node.x) in self.grid_visual:
            self.canvas.itemconfig(self.grid_visual[(node.y, node.x)], fill=color)
            self.canvas.update_idletasks()
            self.canvas.update()

    def start_bi_directional_search(self):
        self.open_set = []
        self.open_set_reverse = []

        self.start.g = 0
        self.start.h = self.heuristic(self.start.x, self.start.y, self.goal)
        self.start.f = self.start.g + self.start.h
        heapq.heappush(self.open_set, (self.start.f, self.start))

        self.goal.g = 0
        self.goal.h = self.heuristic(self.goal.x, self.goal.y, self.start)
        self.goal.f = self.goal.g + self.goal.h
        heapq.heappush(self.open_set_reverse, (self.goal.f, self.goal))

        # Visualization
        self.create_grid_visual()
        self.update_grid_visual(self.start, color="green")
        self.update_grid_visual(self.goal, color="red")

        path, cost = self.search_bi_directional()
        # Add the cost of the start and goal nodes
        cost += int(self.grid[self.start.y][self.start.x])
        cost += int(self.grid[self.goal.y][self.goal.x])
        if path:
            self.reconstruct_path_visual(path)
            print("Bi-directional path found with cost:", cost)
        else:
            print("No path found.")

        self.window.mainloop()

    def get_neighbors_reverse(self, node):
        neighbors = []
        directions = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (1, 1),
            (-1, 1),
            (1, -1),
        ]
        for dx, dy in directions:
            x2, y2 = node.x + dx, node.y + dy
            if (
                0 <= x2 < len(self.grid[0])
                and 0 <= y2 < len(self.grid)
                and self.grid[y2][x2] != "#"
            ):
                neighbors.append(self.nodes_reverse[y2][x2])
        return neighbors

    def process_neighbor_node(self, current_node, neighbor, open_set, closed_set):
        if neighbor is None or (neighbor.x, neighbor.y) in closed_set:
            return
        tentative_g_score = current_node.g + self.move_cost(current_node, neighbor)
        is_better_path = tentative_g_score < neighbor.g
        is_in_open_set = any(neighbor == n[1] for n in open_set)

        if is_better_path:
            old_f = neighbor.f
            neighbor.parent = current_node
            neighbor.g = tentative_g_score
            neighbor.h = self.heuristic(
                neighbor.x,
                neighbor.y,
                self.goal if open_set is self.open_set else self.start,
            )
            neighbor.f = neighbor.g + neighbor.h

        if not is_in_open_set:
            heapq.heappush(open_set, (neighbor.f, neighbor))
            self.update_grid_visual(
                neighbor,
                color="light green" if open_set is self.open_set else "light pink",
            )
        elif is_in_open_set and is_better_path:

            open_set.remove((old_f, neighbor))
            heapq.heappush(open_set, (neighbor.f, neighbor))
            heapq.heapify(open_set)  # Re-sort the heap

    def reconstruct_bi_path(self, meeting_node, is_forward):
        # Reconstruct the path from the meeting point for bi-directional search
        path_forward = self.reconstruct_path(self.nodes[meeting_node.y][meeting_node.x])
        path_backward = self.reconstruct_path(
            self.nodes_reverse[meeting_node.y][meeting_node.x]
        )
        print(path_forward)
        print(path_backward)
        return path_forward + path_backward

    def run_search(self):
        self.create_grid_visual()
        path, cost = self.search()
        # Add the cost of the start node
        cost += int(self.grid[self.start.y][self.start.x])
        if path:
            for node in path:
                # Update path visualization
                self.update_grid_visual(self.nodes[node[1]][node[0]], color="purple")
            print("Path found with cost:", cost)
        else:
            print("No path found.")
        self.window.mainloop()


def main():

    with open("./map2.txt", "r") as f:
        grid = f.readlines()
        grid = [list(line.strip().split(" ")) for line in grid]

    agent = AStar(grid, (4, 10), (36, 0))
    # agent.run_search()
    agent.start_bi_directional_search()


if __name__ == "__main__":
    main()
