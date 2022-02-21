"""CSC111 Winter 2020 Project Phase 2

Instructions
===============================

This Python module contains the program of CSC111 Final Project.
You need to install pandas, networkx, plotly.graph_objs to run this file.

Copyright and Usage Information
===============================

This file is provided solely for the final project for CSC111 at the University of
Toronto St. George campus. All forms of distribution of this code, whether as given
or with any changes, are expressly prohibited. For more information on copyright for
this file, please contact us.

This file is Copyright (c) 2021 Elaine Dai, Nuo Xu, Tommy Gong and Jasmine Zhuang.
"""

from __future__ import annotations
import csv
from typing import Any, Union, List
# from plotly.graph_objs import Scatter, Figure
import pandas as pd
import networkx as nx

COLOUR_SCHEME = [
    '#2E91E5', '#E15F99', '#1CA71C', '#FB0D0D', '#DA16FF', '#222A2A', '#B68100',
    '#750D86', '#EB663B', '#511CFB', '#00A08B', '#FB00D1', '#FC0080', '#B2828D',
    '#6C7C32', '#778AAE', '#862A16', '#A777F1', '#620042', '#1616A7', '#DA60CA',
    '#6C4516', '#0D2A63', '#AF0038'
]

LINE_COLOUR = 'rgb(210,210,210)'
VERTEX_BORDER_COLOUR = 'rgb(50, 50, 50)'
BOOK_COLOUR = 'rgb(89, 205, 105)'
USER_COLOUR = 'rgb(105, 89, 205)'

FILE_ROUTES = ['clusters_500.csv', 'clusters_200.csv',
               'clusters_150.csv', 'clusters_100.csv',
               'clusters_50.csv', 'clusters_20.csv']


################################################################################
# Vertex
################################################################################
class _WeightedVertex:
    """A vertex in a weighted book review graph, used to represent a user or a book.

    Each weighted vertex item is either a user id or book title. Both are represented
    as strings, even though we've kept the type annotation as Any. Neighbours is a
    dictionary mapping a neighbour vertex to the weight of the edge to from self to
    that neighbour.

    Instance Attributes:
        - item: The data stored in this vertex, representing a user or book.
        - kind: The type of this vertex: 'user' or 'book'.
        - neighbours: The vertices that are adjacent to this vertex, and their corresponding
            edge weights.
        - average_rating: For book vertex, it's the average rating of the book;
            for user vertex, it's the average rating given by the user.

    Representation Invariants:
        - self not in self.neighbours
        - all(self in u.neighbours for u in self.neighbours)
        - self.kind in {'user', 'book'}
        - 0.0 <= average_rating <= 5.0
    """
    item: Any
    kind: str
    neighbours: dict[_WeightedVertex, Union[int, float]]
    average_rating: float

    def __init__(self, item: Any, kind: str, average_rating: float = 0.0) -> None:
        """Initialize a new vertex with the given item, kind, and average_rating.

        This vertex is initialized with no neighbours.

        Preconditions:
            - kind in {'user', 'book'}
            - 0.0 <= average_rating <= 5.0
        """
        self.item = item
        self.kind = kind
        self.neighbours = {}
        self.average_rating = average_rating

    def degree(self) -> int:
        """Return the degree of this vertex."""
        return len(self.neighbours)

    def similarity_score_strict(self, other: _WeightedVertex) -> float:
        """Return the strict weighted similarity score between this vertex and other.

        The weighted similarity is calculated by:
        ｜{u ｜ u in V and u is adjacent to u1 AND v2 and u.neighbours[v1] == u.neighbours[v2]}｜
        divided by
        ｜{u ｜ u in V and u is adjacent to u1 OR v2}｜

        """
        if self.degree() == 0 or other.degree() == 0:
            return 0
        else:
            num = len({u for u in self.neighbours if u in other.neighbours
                       and u.neighbours[self] == u.neighbours[other]})
            den = len(set(self.neighbours).union(set(other.neighbours)))
            return num / den


################################################################################
# Graph
################################################################################
class WeightedGraph:
    """A weighted graph used to represent a book review network that keeps track of review scores.

    Note that this is a subclass of the Graph class from Part 1, and so inherits any methods
    from that class that aren't overridden here.
    """
    # Private Instance Attributes:
    #     - _vertices:
    #         A collection of the vertices contained in this graph.
    #         Maps item to _WeightedVertex object.
    _vertices: dict[Any, _WeightedVertex]

    def __init__(self) -> None:
        """Initialize an empty graph (no vertices or edges)."""
        self._vertices = {}

    def get_vertex(self, item: Any) -> _WeightedVertex:
        """Initialize an empty graph (no vertices or edges)."""
        return self._vertices[item]

    def add_vertex(self, item: Any, kind: str, rating: float = 0.0) -> None:
        """Add a vertex with the given item and kind to this graph.

        The new vertex is not adjacent to any other vertices.
        Do nothing if the given item is already in this graph.

        Preconditions:
            - kind in {'user', 'book'}
        """
        if item not in self._vertices:
            self._vertices[item] = _WeightedVertex(item, kind, rating)

    def add_edge(self, item1: Any, item2: Any, weight: Union[int, float] = 1) -> None:
        """Add an edge between the two vertices with the given items in this graph,
        with the given weight.

        Raise a ValueError if item1 or item2 do not appear as vertices in this graph.

        Preconditions:
            - item1 != item2
        """
        if item1 in self._vertices and item2 in self._vertices:
            v1 = self._vertices[item1]
            v2 = self._vertices[item2]

            # Add the new edge
            v1.neighbours[v2] = weight
            v2.neighbours[v1] = weight

        else:
            # We didn't find an existing vertex for both items.
            raise ValueError

    def get_all_vertices(self, kind: str = '') -> set:
        """Return a set of all vertex items in this graph.

        If kind != '', only return the items of the given vertex kind.

        Preconditions:
            - kind in {'', 'user', 'book'}
        """
        if kind != '':
            return {v.item for v in self._vertices.values() if v.kind == kind}
        else:
            return set(self._vertices.keys())

    def get_weight(self, item1: Any, item2: Any) -> Union[int, float]:
        """Return the weight of the edge between the given items.

        Return 0 if item1 and item2 are not adjacent.

        Preconditions:
            - item1 and item2 are vertices in this graph
        """
        v1 = self._vertices[item1]
        v2 = self._vertices[item2]
        return v1.neighbours.get(v2, 0)

    def average_weight(self, item: Any) -> float:
        """Return the average weight of the edges adjacent to the vertex corresponding to item.

        Raise ValueError if item does not corresponding to a vertex in the graph.
        """
        if item in self._vertices:
            v = self._vertices[item]
            return sum(v.neighbours.values()) / len(v.neighbours)
        else:
            raise ValueError

    def update_average_weight(self) -> None:
        """ Update the average_rating of weighted vertices of kind "book".
        """
        for item in self._vertices:
            if self._vertices[item].kind == 'book':
                self._vertices[item].average_rating = self.average_weight(item)

    def get_similarity_score(self, item1: Any, item2: Any) -> float:
        """Return the similarity score between the two given items in this graph.

        Raise a ValueError if item1 or item2 do not appear as vertices in this graph.
        """
        if item1 in self._vertices and item2 in self._vertices:
            return self._vertices[item1].similarity_score_strict(self._vertices[item2])

        else:
            raise ValueError

    def to_networkx(self, max_vertices: int = 5000) -> nx.Graph:
        """Convert this graph into a networkx Graph.

        max_vertices specifies the maximum number of vertices that can appear in the graph.
        (This is necessary to limit the visualization output for large graphs.)
        """
        graph_nx = nx.Graph()
        for v in self._vertices.values():
            graph_nx.add_node(v.item, kind=v.kind)

            for u in v.neighbours:
                if graph_nx.number_of_nodes() < max_vertices:
                    graph_nx.add_node(u.item, kind=u.kind)

                if u.item in graph_nx.nodes:
                    graph_nx.add_edge(v.item, u.item)

            if graph_nx.number_of_nodes() >= max_vertices:
                break

        return graph_nx


################################################################################
# Load Weighted Graph
################################################################################
def load_weighted_review_graph(reviews_file: str) -> WeightedGraph:
    """Return a book review WEIGHTED graph corresponding to the given datasets.

    Preconditions:
        - reviews_file is the path to a CSV file corresponding to the book review data
          format described on the assignment handout
    """
    g = WeightedGraph()

    with open(reviews_file) as csv_file:
        csv_file.readline()
        reader = csv.reader(csv_file)
        for row in reader:
            g.add_vertex('user' + row[1], 'user')
            g.add_vertex('book' + row[2], 'book')
            g.add_edge('user' + row[1], 'book' + row[2], float(row[3]))
        g.update_average_weight()
    return g


################################################################################
# Clusters
################################################################################
def create_book_graph(review_graph: WeightedGraph,
                      threshold: float = 0.05) -> WeightedGraph:
    """Return a book graph based on the given review_graph.

    The returned book graph has the following properties:
        1. Its vertex set is exactly the set of book vertices in review_graph
            (items are book id).
        2. For every two distinct books b1 and b2, let s(b1, b2) be their similarity score.
            - If s(b1, b2) > threshold, there is an edge between b1 and b2 in the book graph
              with weight equal to s(b1, b2).
            - Otherwise, there is no edge between b1 and b2.
    """
    book_names = list(review_graph.get_all_vertices('book'))
    g = WeightedGraph()
    for book in book_names:
        rating = review_graph.get_vertex(book).average_rating
        g.add_vertex(book, 'book', rating)

    for i in range(0, len(book_names)):
        book1 = book_names[i]
        for j in range(i + 1, len(book_names)):
            book2 = book_names[j]
            weight = review_graph.get_similarity_score(book1, book2)
            if weight > threshold:
                g.add_edge(book1, book2, weight)

    return g


def cross_cluster_weight(book_graph: WeightedGraph, cluster1: set, cluster2: set) -> float:
    """Return the cross-cluster weight between cluster1 and cluster2.

    Preconditions:
        - cluster1 != set() and cluster2 != set()
        - cluster1.isdisjoint(cluster2)
        - Every item in cluster1 and cluster2 is a vertex in book_graph
    """
    numerator = sum([book_graph.get_weight(v1, v2) for v1 in cluster1 for v2 in cluster2])
    denominator = len(cluster1) * len(cluster2)
    return numerator / denominator


def find_clusters(graph: WeightedGraph, clusters: list[set], num_clusters: int,
                  filepath: str = '') -> list[set]:
    """Return a list of <num_clusters> vertex clusters for the given graph.
    Input the filepath when users want to store the information of the clusters in a csv file.

    At each iteration, this algorithm chooses the pair of clusters with the highest
    cross-cluster weight to merge.

    Preconditions:
        - num_clusters >= 1
    """
    for _ in range(0, len(clusters) - num_clusters):
        print(f'{len(clusters)} clusters')

        best = -1
        best_c1, best_c2 = None, None

        for i1 in range(0, len(clusters)):
            for i2 in range(i1 + 1, len(clusters)):
                c1, c2 = clusters[i1], clusters[i2]
                score = cross_cluster_weight(graph, c1, c2)
                if score > best:
                    best, best_c1, best_c2 = score, c1, c2

        best_c2.update(best_c1)
        clusters.remove(best_c1)

    # If filepath is not an empty string,
    # then store information of the clusters into a csv file in the given filepath,
    # which we may use later as the function takes time to executes.
    if filepath != '':
        store_into_csv(clusters, filepath)
    return clusters


################################################################################
# Recommender
################################################################################
def find_all_recommended_books(graph: WeightedGraph,
                               item: str, customized_routes: str = '') -> List[str]:
    """Return a full list of books based on the recommended order.
    If the users want to recommend books based on their csv file,
    input the <customized_routes>, which refers to the filepath of their dataset.

    Otherwise,
    the function will find recommended books based on the datasets in FILE_ROUTES.

    Raise ValueError if item is not in the graph.

    Preconditions:
        -item in graph.get_all_vertices('book')
    """
    if item in graph.get_all_vertices('book'):
        result = []
        excluded = set()
        if customized_routes == '':
            input_file_routes = FILE_ROUTES
        else:
            input_file_routes = string_to_list(customized_routes)
        for file_route in input_file_routes:
            books = find_recommended_books(graph, item, file_route, excluded)
            excluded.union(books)
            result.extend(books)

        return result

    else:
        raise ValueError


def find_recommended_books(graph: WeightedGraph,
                           item: str,
                           file_route: str,
                           excluded: set[str]) -> List[str]:
    """Return a list of books arranged in the recommended order based on the data
    stored in <file_route>, which does not contain books in <excluded>.

    Raise ValueError if item is not in the graph.

    Preconditions:
       -item in graph.get_all_vertices('book')
    """
    if item in graph.get_all_vertices('book'):
        clusters = load_csv_clusters(file_route)
        # Find the cluster which contains the given <item>.
        target = set()
        for cluster in clusters:
            if item in cluster:
                target = cluster

        # Find the <num_books> recommended books
        result = []
        while target != set():
            highest_rating_book = ''
            max_score = -1
            for book in target:
                rating = graph.get_vertex(item).average_rating
                if rating > max_score and book not in excluded:
                    max_score = rating
                    highest_rating_book = book

            result.append(highest_rating_book)
            target.remove(highest_rating_book)

        return result

    else:
        raise ValueError


def string_to_list(input_file_routes: str) -> list[str]:
    """Return a list version for the input string.
    """
    remove_string = ['\'', '\"', ' ', '[', ']']
    for replace in remove_string:
        input_file_routes = input_file_routes.replace(f"{replace}", "")
    list_input = input_file_routes.split(',')

    return list_input


def load_csv_clusters(filepath: str) -> List[set]:
    """Load the csv file in the given filepath to a list of set,
    where each element indicates a single cluster.
    """
    result = []
    element = set()

    with open(filepath) as csv_file:
        csv_file.readline()
        for line in csv_file:
            lst = line.replace('\n', '').split(',')
            lst.pop(0)
            for item in lst:
                element.add(item)
            result.append(element)
            element = set()
        for cluster in result:
            if '' in cluster:
                cluster.remove('')
        return result


def store_into_csv(clusters: List[set], filepath: str) -> str:
    """Store the clusters into a csv file in the given filepath.
    """
    df = pd.DataFrame(data=clusters)
    df.to_csv(filepath)

    return 'Stored successfully!'


################################################################################
# Visualization (Used for visualizing the review graph and book graph)
################################################################################
# def visualize_graph(graph: WeightedGraph,
#                     layout: str = 'spring_layout',
#                     max_vertices: int = 5000,
#                     output_file: str = '') -> None:
#     """Use plotly and networkx to visualize the given graph.
#
#     Optional arguments:
#         - layout: which graph layout algorithm to use
#         - max_vertices: the maximum number of vertices that can appear in the graph
#         - output_file: a filename to save the plotly image to (rather than displaying
#             in your web browser)
#     """
#     graph_nx = graph.to_networkx(max_vertices)
#
#     pos = getattr(nx, layout)(graph_nx)
#
#     x_values = [pos[k][0] for k in graph_nx.nodes]
#     y_values = [pos[k][1] for k in graph_nx.nodes]
#     labels = list(graph_nx.nodes)
#     kinds = [graph_nx.nodes[k]['kind'] for k in graph_nx.nodes]
#
#     colours = [BOOK_COLOUR if kind == 'book' else USER_COLOUR for kind in kinds]
#
#     x_edges = []
#     y_edges = []
#     for edge in graph_nx.edges:
#         x_edges += [pos[edge[0]][0], pos[edge[1]][0], None]
#         y_edges += [pos[edge[0]][1], pos[edge[1]][1], None]
#
#     trace3 = Scatter(x=x_edges,
#                      y=y_edges,
#                      mode='lines',
#                      name='edges',
#                      line=dict(color=LINE_COLOUR, width=1),
#                      hoverinfo='none',
#                      )
#     trace4 = Scatter(x=x_values,
#                      y=y_values,
#                      mode='markers',
#                      name='nodes',
#                      marker=dict(symbol='circle-dot',
#                                  size=5,
#                                  color=colours,
#                                  line=dict(color=VERTEX_BORDER_COLOUR, width=0.5)
#                                  ),
#                      text=labels,
#                      hovertemplate='%{text}',
#                      hoverlabel={'namelength': 0}
#                      )
#
#     data1 = [trace3, trace4]
#     fig = Figure(data=data1)
#     fig.update_layout({'showlegend': False})
#     fig.update_xaxes(showgrid=False, zeroline=False, visible=False)
#     fig.update_yaxes(showgrid=False, zeroline=False, visible=False)
#
#     if output_file == '':
#         fig.show()
#     else:
#         fig.write_image(output_file)
#
#
# def visualize_graph_clusters(graph: WeightedGraph, clusters: list[set],
#                              layout: str = 'spring_layout',
#                              max_vertices: int = 5000,
#                              output_file: str = '') -> None:
#     """Visualize the given graph, using different colours to illustrate the different clusters.
#
#     Hides all edges that go from one cluster to another. (This helps the graph layout algorithm
#     positions vertices in the same cluster close together.)
#
#     Same optional arguments as visualize_graph (see that function for details).
#     """
#     graph_nx = graph.to_networkx(max_vertices)
#     all_edges = list(graph_nx.edges)
#     for edge in all_edges:
#         # Check if edge is within the same cluster
#         if any((edge[0] in cluster) != (edge[1] in cluster) for cluster in clusters):
#             graph_nx.remove_edge(edge[0], edge[1])
#
#     pos = getattr(nx, layout)(graph_nx)
#
#     x_values = [pos[k][0] for k in graph_nx.nodes]
#     y_values = [pos[k][1] for k in graph_nx.nodes]
#     labels = list(graph_nx.nodes)
#
#     colors = []
#     for k in graph_nx.nodes:
#         for i, c in enumerate(clusters):
#             if k in c:
#                 colors.append(COLOUR_SCHEME[i % len(COLOUR_SCHEME)])
#                 break
#         else:
#             colors.append(BOOK_COLOUR)
#
#     x_edges = []
#     y_edges = []
#     for edge in graph_nx.edges:
#         x_edges += [pos[edge[0]][0], pos[edge[1]][0], None]
#         y_edges += [pos[edge[0]][1], pos[edge[1]][1], None]
#
#     trace3 = Scatter(x=x_edges,
#                      y=y_edges,
#                      mode='lines',
#                      name='edges',
#                      line=dict(color=LINE_COLOUR, width=1),
#                      hoverinfo='none'
#                      )
#     trace4 = Scatter(x=x_values,
#                      y=y_values,
#                      mode='markers',
#                      name='nodes',
#                      marker=dict(symbol='circle-dot',
#                                  size=5,
#                                  color=colors,
#                                  line=dict(color=VERTEX_BORDER_COLOUR, width=0.5)
#                                  ),
#                      text=labels,
#                      hovertemplate='%{text}',
#                      hoverlabel={'namelength': 0}
#                      )
#
#     data1 = [trace3, trace4]
#     fig = Figure(data=data1)
#     fig.update_layout({'showlegend': False})
#     fig.update_xaxes(showgrid=False, zeroline=False, visible=False)
#     fig.update_yaxes(showgrid=False, zeroline=False, visible=False)
#     fig.show()
#
#     if output_file == '':
#         fig.show()
#     else:
#         fig.write_image(output_file)


if __name__ == '__main__':
    import python_ta.contracts
    python_ta.contracts.check_all_contracts()

    import python_ta
    python_ta.check_all(config={
        'extra-imports': ['csv', 'pandas', 'networkx', 'plotly.graph_objs'],
        'allowed-io': ['load_weighted_review_graph', 'find_clusters', 'load_csv_clusters'],
        'max-line-length': 100,
        'disable': ['E1136'],
        'max-nested-blocks': 4
    })
