import networkx as nx
import matplotlib.pyplot as plt
import math
import pandas as pd
from tabulate import tabulate

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

def haversine(coord1, coord2):
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

print("----- 1. Построение графа G* -----")
all_countries = [
    "Albania", "Andorra", "Austria", "Belarus", "Belgium", "Bosnia and Herzegovina",
    "Bulgaria", "Croatia", "Cyprus", "Czechia", "Denmark", "Estonia", "Finland",
    "France", "Germany", "Greece", "Hungary", "Iceland", "Ireland", "Italy", "Latvia",
    "Liechtenstein", "Lithuania", "Luxembourg", "Malta", "Moldova", "Monaco", "Montenegro",
    "Netherlands", "North Macedonia", "Norway", "Poland", "Portugal",
    "Romania", "Russia", "San Marino", "Serbia", "Slovakia", "Slovenia", "Spain",
    "Sweden", "Switzerland", "Turkey", "Ukraine", "United Kingdom", "Vatican City"
]

edge_list = [
    ("Austria", "Czechia"),
    ("Austria", "Germany"),
    ("Austria", "Hungary"),
    ("Austria", "Italy"),
    ("Austria", "Liechtenstein"),
    ("Austria", "Slovakia"),
    ("Austria", "Slovenia"),
    ("Austria", "Switzerland"),
    ("Belgium", "France"),
    ("Belgium", "Germany"),
    ("Belgium", "Luxembourg"),
    ("Belgium", "Netherlands"),
    ("Bulgaria", "Greece"),
    ("Bulgaria", "North Macedonia"),
    ("Bulgaria", "Romania"),
    ("Bulgaria", "Serbia"),
    ("Bulgaria", "Turkey"),
    ("Czechia", "Germany"),
    ("Czechia", "Poland"),
    ("Czechia", "Slovakia"),
    ("Denmark", "Germany"),
    ("Estonia", "Latvia"),
    ("Estonia", "Russia"),
    ("Finland", "Norway"),
    ("Finland", "Russia"),
    ("Finland", "Sweden"),
    ("France", "Andorra"),
    ("France", "Italy"),
    ("France", "Monaco"),
    ("France", "Spain"),
    ("France", "Luxembourg"),
    ("France", "Germany"),
    ("France", "Switzerland"),
    ("Germany", "Luxembourg"),
    ("Germany", "Netherlands"),
    ("Germany", "Poland"),
    ("Germany", "Switzerland"),
    ("Greece", "Albania"),
    ("Greece", "North Macedonia"),
    ("Greece", "Turkey"),
    ("Hungary", "Croatia"),
    ("Hungary", "Romania"),
    ("Hungary", "Serbia"),
    ("Hungary", "Slovenia"),
    ("Hungary", "Slovakia"),
    ("Italy", "San Marino"),
    ("Italy", "Slovenia"),
    ("Italy", "Switzerland"),
    ("Italy", "Vatican City"),
    ("Latvia", "Lithuania"),
    ("Latvia", "Russia"),
    ("Latvia", "Belarus"),
    ("Lithuania", "Belarus"),
    ("Lithuania", "Poland"),
    ("Lithuania", "Russia"),
    ("Moldova", "Romania"),
    ("Moldova", "Ukraine"),
    ("North Macedonia", "Albania"),
    ("North Macedonia", "Greece"),
    ("North Macedonia", "Serbia"),
    ("Norway", "Russia"),
    ("Norway", "Sweden"),
    ("Poland", "Slovakia"),
    ("Poland", "Ukraine"),
    ("Poland", "Russia"),
    ("Belarus", "Poland"),
    ("Portugal", "Spain"),
    ("Romania", "Ukraine"),
    ("Romania", "Serbia"),
    ("Russia", "Belarus"),
    ("Russia", "Ukraine"),
    ("Serbia", "Albania"),
    ("Serbia", "Croatia"),
    ("Serbia", "North Macedonia"),
    ("Slovakia", "Ukraine"),
    ("Slovenia", "Croatia"),
    ("Spain", "Andorra"),
    ("Switzerland", "Liechtenstein"),
    ("Turkey", "Bulgaria"),
    ("Ukraine", "Belarus"),
    ("Ukraine", "Hungary"),
    ("Bosnia and Herzegovina", "Croatia"),
    ("Bosnia and Herzegovina", "Serbia"),
    ("Montenegro", "Albania"),
    ("Montenegro", "Bosnia and Herzegovina"),
    ("Montenegro", "Serbia"),
    ("Montenegro", "Croatia"),
    ("Ireland", "United Kingdom")
]
edges = list({tuple(sorted([u, v])) for u, v in edge_list})

G_star = nx.Graph()
G_star.add_nodes_from(all_countries)
G_star.add_edges_from(edges)

is_planar, embedding = nx.check_planarity(G_star, False)
if is_planar:
    pos = nx.planar_layout(G_star, scale=5)
    print("Граф является планарным, используется planar_layout(scale=5).")
else:
    pos = nx.spring_layout(G_star, seed=42, k=3.0, scale=5, iterations=200)
    print("Граф не является планарным, используется spring_layout(k=3.0, scale=5).")

# Координаты столиц
capitals = {
    "Andorra": (42.5063, 1.5218),
    "Albania": (41.3275, 19.8187),
    "Austria": (48.2082, 16.3738),
    "Belarus": (53.9, 27.5667),
    "Belgium": (50.8503, 4.3517),
    "Bosnia and Herzegovina": (43.8563, 18.4131),
    "Bulgaria": (42.6977, 23.3219),
    "Croatia": (45.8150, 15.9819),
    "Cyprus": (35.1856, 33.3823),
    "Czechia": (50.0755, 14.4378),
    "Denmark": (55.6761, 12.5683),
    "Estonia": (59.4370, 24.7536),
    "Finland": (60.1699, 24.9384),
    "France": (48.8566, 2.3522),
    "Germany": (52.5200, 13.4050),
    "Greece": (37.9838, 23.7275),
    "Hungary": (47.4979, 19.0402),
    "Iceland": (64.1466, -21.9426),
    "Ireland": (53.3498, -6.2603),
    "Italy": (41.9028, 12.4964),
    "Latvia": (56.9496, 24.1052),
    "Liechtenstein": (47.1410, 9.5215),
    "Lithuania": (54.6872, 25.2797),
    "Luxembourg": (49.6116, 6.1319),
    "Malta": (35.8989, 14.5146),
    "Moldova": (47.0105, 28.8638),
    "Monaco": (43.7384, 7.4246),
    "Montenegro": (42.4304, 19.2594),
    "Netherlands": (52.3676, 4.9041),
    "North Macedonia": (41.9962, 21.4314),
    "Norway": (59.9139, 10.7522),
    "Poland": (52.2297, 21.0122),
    "Portugal": (38.7223, -9.1393),
    "Romania": (44.4268, 26.1025),
    "Russia": (55.7558, 37.6173),
    "San Marino": (43.9424, 12.4578),
    "Serbia": (44.7866, 20.4489),
    "Slovakia": (48.1486, 17.1077),
    "Slovenia": (46.0569, 14.5058),
    "Spain": (40.4168, -3.7038),
    "Sweden": (59.3293, 18.0686),
    "Switzerland": (46.9480, 7.4474),
    "Turkey": (41.0082, 28.9784),
    "Ukraine": (50.4501, 30.5234),
    "United Kingdom": (51.5074, -0.1278),
    "Vatican City": (41.9029, 12.4534)
}

for u, v in G_star.edges():
    if u in capitals and v in capitals:
        G_star[u][v]['weight'] = haversine(capitals[u], capitals[v])
    else:
        G_star[u][v]['weight'] = 1

plt.figure(figsize=(16, 12))
nx.draw(G_star, pos, with_labels=True, node_color='lightblue', edge_color='gray',
        node_size=150, font_size=6, alpha=0.9)
edge_labels = nx.get_edge_attributes(G_star, 'weight')
edge_labels = {(u, v): round(w, 2) for (u, v), w in edge_labels.items()}
nx.draw_networkx_edge_labels(G_star, pos, edge_labels=edge_labels, font_size=6)
plt.title("Граф $G^*$ с расстояниями (весами) на рёбрах")
plt.show()

print("\n----- 2. Параметры графа G -----")
components = list(nx.connected_components(G_star))
G = G_star.subgraph(max(components, key=len)).copy()
print("Количество вершин |V| =", G_star.number_of_nodes())
print("Количество ребер |E| =", G_star.number_of_edges())
degrees = dict(G.degree())
print("Минимальная степень δ(G) =", min(degrees.values()))
print("Максимальная степень Δ(G) =", max(degrees.values()))
print("Радиус rad(G) =", nx.radius(G))
print("Диаметр diam(G) =", nx.diameter(G))
print("Центр графа center(G) =", nx.center(G))
cyclomatic_number = G.number_of_edges() - G.number_of_nodes() + 1
print("Цикломатическое число =", cyclomatic_number)

print("\n----- 3. Хроматическое число χ(G) -----")
coloring = nx.coloring.greedy_color(G, strategy="largest_first")
chromatic_number = max(coloring.values())
print("Хроматическое число χ(G) =", chromatic_number)

print("\n----- 4. Максимальная клика Q ⊆ V -----")
cliques = list(nx.find_cliques(G))
max_clique = max(cliques, key=len)
print("Максимальная клика:", max_clique)

print("\n----- 5. Максимальный эйлеров подграф -----") #искал руками
input_countries = [
    "Finland", "Norway", "Russia", "Latvia", "Belarus", "Poland", "Ukraine", "Moldova",
    "Romania", "Hungary", "Serbia", "Bosnia and Herzegovina", "Croatia", "Albania",
    "North Macedonia", "Slovenia", "Czechia", "Italy", "Austria", "Switzerland",
    "Liechtenstein", "France", "Belgium", "Luxembourg", "Andorra", "Spain"
]
H_input = G_star.subgraph(input_countries).copy()

def greedy_max_eulerian_induced_subgraph(G):
    H = G.copy()
    while any(H.degree(v) % 2 != 0 for v in H.nodes()):
        odd_nodes = [v for v in H.nodes() if H.degree(v) % 2 != 0]
        if not odd_nodes:
            break
        v_remove = min(odd_nodes, key=lambda v: H.degree(v))
        H.remove_node(v_remove)
    return H

H_eulerian = greedy_max_eulerian_induced_subgraph(H_input)

if H_eulerian.number_of_nodes() > 0 and all(H_eulerian.degree(v) % 2 == 0 for v in H_eulerian.nodes()):
    print("Максимальный индуцированный эйлеров подграф имеет", H_eulerian.number_of_nodes(),
          "вершин и", H_eulerian.number_of_edges(), "ребер.")
    print("Вершины эйлерова подграфа:", list(H_eulerian.nodes()))
    print("Рёбра эйлерова подграфа:", list(H_eulerian.edges()))
else:
    print("Эйлеров подграф не найден или слишком мал.")

plt.figure(figsize=(10, 8))
nx.draw(H_eulerian, pos, with_labels=True, node_color='orange', edge_color='red', node_size=500, font_size=10)
plt.title("Максимальный индуцированный эйлеров подграф (из введённых стран)")
plt.show()

print("\n----- 6. Максимальный гамильтонов подграф -----") #искал руками
hamiltonian_vertices = [
    "France", "Luxembourg", "Belgium", "Netherlands", "Germany", "Czechia",
    "Slovakia", "Poland", "Belarus", "Lithuania", "Latvia", "Estonia",
    "Russia", "Ukraine", "Moldova", "Romania", "Serbia", "North Macedonia",
    "Bulgaria", "Turkey", "Greece", "Albania", "Montenegro", "Bosnia and Herzegovina",
    "Croatia", "Slovenia", "Hungary", "Austria", "Liechtenstein", "Switzerland", "Italy"
]

H_path = nx.Graph()
H_path.add_nodes_from(hamiltonian_vertices)

n = len(hamiltonian_vertices)
for i in range(n - 1):
    H_path.add_edge(hamiltonian_vertices[i], hamiltonian_vertices[i + 1])

print("Вершины графа-пути:")
print(list(H_path.nodes()))
print("\nРёбра графа-пути:")
print(sorted(H_path.edges()))
print("\nКоличество вершин:", H_path.number_of_nodes())
print("Количество рёбер:", H_path.number_of_edges())

plt.figure(figsize=(10, 8))
nx.draw(H_path, pos, with_labels=True, node_color='lightgreen', edge_color='black',
        node_size=500, font_size=10)
plt.title("Гамильтонов граф (расположение вершин по основному графу)")
plt.show()

print("\n----- 7. Компоненты вершинной двусвязности и граф блоков и точек сочленения -----")
articulation_points = list(nx.articulation_points(G_star))
biconnected_components = list(nx.biconnected_components(G_star))
print("Точки сочленения графа G*:", articulation_points)
for comp in biconnected_components:
    print(comp)
BCT = nx.Graph()
for comp in biconnected_components:
    comp_name = "Block_" + "_".join(sorted(comp))
    BCT.add_node(comp_name, type="block", members=comp)
    for v in comp:
        if v in articulation_points:
            BCT.add_node(v, type="articulation")
            BCT.add_edge(comp_name, v)
print("Граф блоков и точек сочленения построен.")

print("\n----- 8.Компоненты реберной двусвязности и граф компонент реберной двусвязности -----")
edge_biconnected_components = list(nx.biconnected_component_edges(G_star))
# Добавляем изолированные вершины как отдельные компоненты реберной двусвязности
isolated_nodes = [node for node in G_star.nodes() if G_star.degree(node) == 0]
for node in isolated_nodes:
    edge_biconnected_components.append({node})

print("Компоненты реберной двусвязности графа G*:")
for comp in edge_biconnected_components:
    print(comp)
EBCG = nx.Graph()
for i, comp in enumerate(edge_biconnected_components):
    node_name = f"EBC_{i}"
    EBCG.add_node(node_name, members=comp)
print("Граф компонент реберной двусвязности построен.")

print("\n----- 9. Минимальное остовное дерево T -----")
G_weighted = G.copy()
for u, v in G_weighted.edges():
    if u in capitals and v in capitals:
        G_weighted[u][v]['weight'] = haversine(capitals[u], capitals[v])
    else:
        G_weighted[u][v]['weight'] = 1

T = nx.minimum_spanning_tree(G_weighted, weight='weight')
print("Минимальное остовное дерево T найдено.")
print("T имеет", T.number_of_nodes(), "вершин и", T.number_of_edges(), "ребер.")

print("\n----- 10. Код Прюфера и бинарное представление MST T -----")
def prufer_code(T):
    T_copy = T.copy()
    prufer = []
    while T_copy.number_of_nodes() > 2:
        leaves = [node for node in T_copy.nodes() if T_copy.degree(node) == 1]
        leaf = min(leaves)
        neighbor = list(T_copy.neighbors(leaf))[0]
        prufer.append(neighbor)
        T_copy.remove_node(leaf)
    return prufer

prufer = prufer_code(T)
print("Код Прюфера для T:", prufer)

def binary_code_tree(T):
    nodes = sorted(T.nodes())
    n = len(nodes)
    adj_matrix = [[0] * n for _ in range(n)]
    idx = {node: i for i, node in enumerate(nodes)}
    for u, v in T.edges():
        i, j = idx[u], idx[v]
        adj_matrix[i][j] = 1
        adj_matrix[j][i] = 1
    df = pd.DataFrame(adj_matrix, index=nodes, columns=nodes)
    return df

df_matrix = binary_code_tree(T)
print("Бинарное представление T (матрица смежности):")
print(tabulate(df_matrix, headers='keys', tablefmt='grid'))

print("\nБинарный код MST T (строковое представление):")
binary_code_lines = ["".join(str(x) for x in row) for row in df_matrix.values]
for line in binary_code_lines:
    print(line)

print("\n----- Таблица расстояний между столицами -----")
capitals_list = list(capitals.keys())
edge_distance_table = pd.DataFrame(index=capitals_list, columns=capitals_list)
for cap1 in capitals_list:
    for cap2 in capitals_list:
        if cap1 == cap2:
            edge_distance_table.loc[cap1, cap2] = 0
        elif G_star.has_edge(cap1, cap2):
            edge_distance_table.loc[cap1, cap2] = round(G_star[cap1][cap2]['weight'], 2)
        else:
            edge_distance_table.loc[cap1, cap2] = ''
print("Матрица расстояний (только для существующих рёбер):")
print(tabulate(edge_distance_table, headers='keys', tablefmt='grid'))
