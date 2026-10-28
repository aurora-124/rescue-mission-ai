import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random


grid = np.array([
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 1, 1, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 1, 1, 1, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 1, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])

robot_pos = (0, 0)  
victims = [(0, 8), (2, 4), (4, 9), (8, 0), (9, 9)]  

locations = [robot_pos] + victims
num_locations = len(locations)


def bfs_path(start, goal, grid):
    if start == goal:
        return [start]

    rows, cols = grid.shape
    queue = deque([start])
    visited = {start: None}

    directions = [(-1,0),(1,0),(0,-1),(0,1)]

    while queue:
        current = queue.popleft()

        if current == goal:
            break

        for dr, dc in directions:
            nr, nc = current[0] + dr, current[1] + dc

            
            if (0 <= nr < rows and
                0 <= nc < cols and
                grid[nr, nc] == 0 and
                (nr, nc) not in visited):

                queue.append((nr, nc))
                visited[(nr, nc)] = current

    
    path = []
    node = goal

    while node is not None:
        path.append(node)
        node = visited.get(node)

    path.reverse()
    return path


def bfs_distance(start, goal):
    path = bfs_path(start, goal, grid)
    if not path or path[-1] != goal:
        return 9999
    return len(path) - 1

dist_matrix = np.zeros((num_locations, num_locations), dtype=int)

for i in range(num_locations):
    for j in range(num_locations):
        if i != j:
            dist_matrix[i][j] = bfs_distance(locations[i], locations[j])

print("Distance Matrix:")
print(dist_matrix)


POP_SIZE = 100
GEN_COUNT = 200
MUTATION_RATE = 0.2

def calculate_fitness(chromosome):
    total = dist_matrix[0][chromosome[0]]

    for i in range(len(chromosome)-1):
        total += dist_matrix[chromosome[i]][chromosome[i+1]]

    return total

def get_initial_population():
    pop = []
    victims_idx = list(range(1, num_locations))

    for _ in range(POP_SIZE):
        pop.append(random.sample(victims_idx, len(victims_idx)))

    return pop

def selection(pop, scores):
    i, j = random.sample(range(len(pop)), 2)
    return pop[i] if scores[i] < scores[j] else pop[j]

def crossover(p1, p2):
    size = len(p1)
    a, b = sorted(random.sample(range(size), 2))

    child = [None]*size
    child[a:b] = p1[a:b]

    fill = [x for x in p2 if x not in child]

    for i in range(size):
        if child[i] is None:
            child[i] = fill.pop(0)

    return child

def mutate(chrom):
    if random.random() < MUTATION_RATE:
        i, j = random.sample(range(len(chrom)), 2)
        chrom[i], chrom[j] = chrom[j], chrom[i]
    return chrom


population = get_initial_population()
best = None
best_score = float('inf')

for _ in range(GEN_COUNT):
    scores = [calculate_fitness(c) for c in population]

    for i in range(len(population)):
        if scores[i] < best_score:
            best_score = scores[i]
            best = population[i]

    new_pop = [best.copy()]  

    while len(new_pop) < POP_SIZE:
        p1 = selection(population, scores)
        p2 = selection(population, scores)

        child = crossover(p1, p2)
        child = mutate(child)

        new_pop.append(child)

    population = new_pop

print("\nBest order:", best)
print("Total distance (Steps):", best_score)


def plot_final_path_actual(grid, locations, best_order, total_steps):
    fig, ax = plt.figure(figsize=(10, 10)), plt.gca()
    
    ax.imshow(grid, cmap='Greys', origin='upper')

    ax.set_xticks(np.arange(-0.5, 10, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 10, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    title_obj = plt.title(f"Rescue Mission Route (Actual Path)\nBest Order: {best_order} | Total Steps: {total_steps:.1f}", fontsize=14)
    plt.setp(title_obj, color='black') 

    def draw_label(pos, text, color):
        y, x = pos
        ax.text(x, y, text, fontsize=12, fontweight='bold', color=color,
                ha='center', va='center', zorder=5,
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))

    draw_label(locations[0], 'R', 'blue')
    for i in range(1, num_locations):
        draw_label(locations[i], f'V{i}', 'red')

    
    full_path_indices = [0] + best_order

    for i in range(len(full_path_indices) - 1):
        start_idx = full_path_indices[i]
        end_idx = full_path_indices[i+1]
        
        start_pos = locations[start_idx]
        end_pos = locations[end_idx]

       
        actual_path = bfs_path(start_pos, end_pos, grid)
        
        
        for j in range(len(actual_path) - 1):
            curr_step = actual_path[j]
            next_step = actual_path[j+1]
            
            x_curr, y_curr = curr_step[1], curr_step[0]
            x_next, y_next = next_step[1], next_step[0]
            
           
            ax.plot([x_curr, x_next], [y_curr, y_next], color="#141277", linewidth=3, zorder=2)
            
          
            if j == len(actual_path) - 2:
                ax.annotate("", xy=(x_next, y_next), xytext=(x_curr, y_curr),
                            arrowprops=dict(arrowstyle="->", color="#14007A", linewidth=3, mutation_scale=20), zorder=3)

    plt.show()

plot_final_path_actual(grid, locations, best, best_score)