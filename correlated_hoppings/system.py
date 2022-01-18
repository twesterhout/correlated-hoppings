import numpy as np
import quspin
from typing import Any, List, Tuple


def are_edges_ordered(edges: List[Tuple[int, int]]) -> bool:
    return all(map(lambda t: t[0] < t[1], edges))


class SquareLattice:
    def __init__(self, edges: List[Tuple[int, int]]):
        assert are_edges_ordered(edges)
        self.edges = edges


def simple_square_lattice_edges(shape: Tuple[int, int]) -> List[Tuple[int, int]]:
    def index(x: int, y: int) -> int:
        return (y % shape[0]) * shape[1] + (x % shape[1])

    edges: List[Tuple[int, int]] = []
    for y in range(shape[0]):
        for x in range(shape[1]):
            i = index(x, y)
            edges.append((i, index(x + 1, y)))
            edges.append((i, index(x, y + 1)))

    def order_edge(edge: Tuple[int, int]) -> Tuple[int, int]:
        (i, j) = edge
        return (i, j) if i < j else (j, i)

    edges = sorted(map(order_edge, edges))
    assert are_edges_ordered(edges)
    return edges


def square_3x3() -> SquareLattice:
    return SquareLattice(simple_square_lattice_edges((3, 3)))


def unconjugated_hamiltonian(
    β1: float, β2: float, γ: float, edges: List[Tuple[int, int]]
) -> List[Any]:
    assert are_edges_ordered(edges)
    edges = edges + [(j, i) for i, j in edges]
    return [
        ["+-|", [[β1, i, j] for i, j in edges]],
        ["+-|n", [[-β1, i, j, i] for i, j in edges]],
        ["+-|n", [[-β1, i, j, j] for i, j in edges]],
        ["+-|nn", [[β1, i, j, i, j] for i, j in edges]],
        ["|+-", [[β1, i, j] for i, j in edges]],
        ["n|+-", [[-β1, i, i, j] for i, j in edges]],
        ["n|+-", [[-β1, j, i, j] for i, j in edges]],
        ["nn|+-", [[β1, i, j, i, j] for i, j in edges]],
        ["+-|nn", [[β2, i, j, i, j] for i, j in edges]],
        ["nn|+-", [[β2, i, j, i, j] for i, j in edges]],
        ["+-|n", [[γ, i, j, i] for i, j in edges]],
        ["+-|nn", [[-γ, i, j, i, j] for i, j in edges]],
        ["n|+-", [[γ, i, i, j] for i, j in edges]],
        ["nn|+-", [[-γ, i, j, i, j] for i, j in edges]],
        ["+-|n", [[γ, i, j, j] for i, j in edges]],
        ["+-|nn", [[-γ, i, j, i, j] for i, j in edges]],
        ["n|+-", [[γ, j, i, j] for i, j in edges]],
        ["nn|+-", [[-γ, i, j, i, j] for i, j in edges]],
    ]


def make_hamiltonian(β1: float, β2: float, γ: float, U: float, edges: List[Tuple[int, int]], basis):
    static_part = unconjugated_hamiltonian(β1, β2, γ, edges)
    # NOTE: we need to divide basis.N by 2 because we have spin ↑ and ↓.
    static_part.append(["n|n", [[U, i, i] for i in range(basis.N // 2)]])
    hamiltonian = quspin.operators.hamiltonian(static_part, [], basis=basis, dtype=np.float64)
    # hamiltonian = hamiltonian + hamiltonian.H
    return hamiltonian


def normal_hubbard(J, U, edges, basis):
    static_part = [
        ["+-|", [[-J, i, j] for i, j in edges]],
        ["-+|", [[+J, i, j] for i, j in edges]],
        ["|+-", [[-J, i, j] for i, j in edges]],
        ["|-+", [[+J, i, j] for i, j in edges]],
        ["n|n", [[U, i, i] for i in range(basis.N // 2)]],
    ]
    return quspin.operators.hamiltonian(static_part, [], basis=basis, dtype=np.float64)


def run_tests():
    # 0 1 2
    # 3 4 5
    # 6 7 8
    print(square_3x3().edges)
    assert square_3x3().edges == sorted(
        [
            (0, 1),
            (1, 2),
            (0, 2),
            (0, 3),
            (1, 4),
            (2, 5),
            (3, 4),
            (4, 5),
            (3, 5),
            (3, 6),
            (4, 7),
            (5, 8),
            (6, 7),
            (7, 8),
            (6, 8),
            (0, 6),
            (1, 7),
            (2, 8),
        ]
    )

    number_sites = 2
    edges = [(0, 1)]
    # number_sites = 3 * 3
    # edges = square_3x3().edges
    J = 1
    U = 5

    basis = quspin.basis.spinful_fermion_basis_general(number_sites)
    h1 = make_hamiltonian(-J, -J, -J, U, edges, basis).tocsr()
    h2 = normal_hubbard(J, U, edges, basis).tocsr()
    assert h1.shape == h2.shape
    assert np.all(h1.data == h2.data)
    assert np.all(h1.indices == h2.indices)
    assert np.all(h1.indptr == h2.indptr)

    # print(basis)
    # hamiltonian = 
    # print(hamiltonian)
    # energy, ground_state = hamiltonian.eigsh(k=1, which="SA")
    # print(energy)
    # return
    # basis = quspin.basis.spinful_fermion_basis_general(3 * 3, Nf=(3, 3))
    # hamiltonian = normal_hubbard(J, U, edges, basis)
    # print(hamiltonian)
    # hamiltonian = make_hamiltonian(-1, -1, -1, 5, square_3x3().edges, basis)
    # energy, ground_state = hamiltonian.eigsh(k=1, which="SA")
    # print(energy)


if __name__ == "__main__":
    run_tests()
