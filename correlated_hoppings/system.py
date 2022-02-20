import numpy as np
import quspin
from quspin.basis import spinful_fermion_basis_general
import scipy.sparse.linalg
from typing import Any, List, Tuple


def are_edges_ordered(edges: List[Tuple[int, int]]) -> bool:
    return all(map(lambda t: t[0] < t[1], edges))


class SquareLattice:
    def __init__(self, number_sites: int, edges: List[Tuple[int, int]]):
        assert are_edges_ordered(edges)
        self.number_sites = number_sites
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


def square_4() -> SquareLattice:
    return SquareLattice(4, [(0, 1), (2, 3), (0, 2), (1, 3)])


def square_5() -> SquareLattice:
    #          0  1
    #       2  3  4
    #
    #             0  1
    #          2  3  4  0  1
    #          0  1  2  3  4
    #       2  3  4  0  1
    #             2  3  4
    #
    # fmt: off
    return SquareLattice(5, [(0, 1), (0, 3), (1, 2), (1, 4), (2, 3),
                             (0, 2), (3, 4), (1, 3), (0, 4), (2, 4)])
    # fmt: on


def square_8() -> SquareLattice:
    #          0  1
    #       2  3  4  5
    #          6  7
    #
    #                    0  1
    #                 2  3  4  5
    #              0  1  6  7  0  1
    #           2  3  4  5  2  3  4  5
    #              6  7  0  1  6  7
    #                 2  3  4  5
    #                    6  7
    #
    # fmt: off
    return SquareLattice(8, [(0, 1), (0, 3), (1, 6), (1, 4),
                             (2, 3), (1, 2), (3, 4), (3, 6),
                             (4, 5), (4, 7), (2, 5), (0, 5),
                             (6, 7), (5, 6), (0, 7), (2, 7)])
    # fmt: on


def square_9() -> SquareLattice:
    return SquareLattice(9, simple_square_lattice_edges((3, 3)))


def square_10() -> SquareLattice:
    #          0  1  2
    #          3  4  5
    #       6  7  8  9
    #
    #                        0  1  2
    #                        3  4  5  0  1  2
    #                     6  7  8  9  3  4  5
    #                     0  1  2  6  7  8  9
    #                     3  4  5  0  1  2
    #                  6  7  8  9  3  4  5
    #                           6  7  8  9
    #
    #
    # fmt: off
    return SquareLattice(10, [(0, 1), (0, 3), (1, 2), (1, 4),
                              (2, 6), (2, 5), (3, 4), (3, 7),
                              (4, 5), (4, 8), (0, 5), (5, 9),
                              (6, 7), (0, 6), (7, 8), (1, 7),
                              (8, 9), (2, 8), (3, 9), (6, 9)])
    # fmt: on


def square_16() -> SquareLattice:
    return SquareLattice(16, simple_square_lattice_edges((4, 4)))


def unconjugated_hamiltonian(
    β1: float, β2: float, γ: float, edges: List[Tuple[int, int]]
) -> List[Any]:
    assert are_edges_ordered(edges)
    edges = edges + [(j, i) for i, j in edges]
    return [
        ["+-|", [[β2, i, j] for i, j in edges]],
        ["+-|n", [[-β2, i, j, i] for i, j in edges]],
        ["+-|n", [[-β2, i, j, j] for i, j in edges]],
        ["+-|nn", [[β2, i, j, i, j] for i, j in edges]],
        ["|+-", [[β2, i, j] for i, j in edges]],
        ["n|+-", [[-β2, i, i, j] for i, j in edges]],
        ["n|+-", [[-β2, j, i, j] for i, j in edges]],
        ["nn|+-", [[β2, i, j, i, j] for i, j in edges]],
        ["+-|nn", [[β1, i, j, i, j] for i, j in edges]],
        ["nn|+-", [[β1, i, j, i, j] for i, j in edges]],
        ["+-|n", [[γ, i, j, i] for i, j in edges]],
        ["+-|nn", [[-2 * γ, i, j, i, j] for i, j in edges]],
        ["n|+-", [[γ, i, i, j] for i, j in edges]],
        ["nn|+-", [[-2 * γ, i, j, i, j] for i, j in edges]],
        ["+-|n", [[γ, i, j, j] for i, j in edges]],
        # ["+-|nn", [[-γ, i, j, i, j] for i, j in edges]],
        ["n|+-", [[γ, j, i, j] for i, j in edges]],
        # ["nn|+-", [[-γ, i, j, i, j] for i, j in edges]],
    ]


def make_hamiltonian(
    β1: float, β2: float, γ: float, U: float, edges: List[Tuple[int, int]], basis, **kwargs
):
    static_part = unconjugated_hamiltonian(β1, β2, γ, edges)
    # NOTE: we need to divide basis.N by 2 because we have spin ↑ and ↓.
    static_part.append(["n|n", [[U, i, i] for i in range(basis.N // 2)]])
    hamiltonian = quspin.operators.hamiltonian(
        static_part, [], basis=basis, dtype=np.float64, check_pcon=False, check_symm=False, **kwargs
    )
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


def measure_total_spin(v, basis):
    indices = [(i, j) for i in range(basis.N // 2) for j in range(basis.N // 2)]

    static_part = [
        # S_x S_x
        # ["--|++", [[-1, i, j, i, j] for i, j in indices]],
        # ["++|--", [[-1, i, j, i, j] for i, j in indices]],
        ["-+|+-", [[1, i, j, i, j] for i, j in indices]],
        ["+-|-+", [[1, i, j, i, j] for i, j in indices]],
        # S_y S_y
        # ["--|++", [[1, i, j, i, j] for i, j in indices]],
        # ["++|--", [[1, i, j, i, j] for i, j in indices]],
        ["-+|+-", [[1, i, j, i, j] for i, j in indices]],
        ["+-|-+", [[1, i, j, i, j] for i, j in indices]],
        # S_z S_z
        ["nn|", [[1, i, j] for i, j in indices]],
        ["n|n", [[-1, i, j] for i, j in indices]],
        ["n|n", [[-1, i, j] for i, j in indices]],
        ["|nn", [[1, i, j] for i, j in indices]],
    ]
    S = quspin.operators.hamiltonian(
        static_part,
        [],
        basis=basis,
        dtype=np.float64,
        check_herm=False,
        check_pcon=False,
        check_symm=False,
    )
    if v.ndim == 1:
        v = v.reshape(-1, 1)
    rs = np.sum(v.conj() * S.dot(v), axis=0)
    print(rs)
    return [0.5 * (-1 + np.sqrt(1 + r)) for r in rs]


def tune_γ_and_U(lattice):
    β1 = 1
    β2 = 1
    number_sites = lattice.number_sites
    number_fermions = number_sites - 1
    number_down = number_fermions // 2
    number_up = number_fermions - number_down
    basis_2_1 = spinful_fermion_basis_general(number_sites, Nf=(number_up, number_down))
    basis_3_0 = spinful_fermion_basis_general(number_sites, Nf=(number_fermions, 0))

    γ_grid = np.linspace(0, 1, num=21)
    U_grid = np.linspace(0, 20, num=21)
    # total_spin_grid = np.zeros((len(γ_grid), len(U_grid)), dtype=np.float64)
    table = []
    filename = "2x2_gamma_U.dat"
    with open(filename, "w") as f:
        f.write("# γ\tU\tE₀\tS\tE₁\tE₂\tE₃\n")
    for i, γ in enumerate(γ_grid):
        with open(filename, "a") as f:
            if i != 0:
                f.write("\n")
        v0 = None
        for j, U in enumerate(U_grid):
            h_2_1 = make_hamiltonian(-β1, -β2, -γ, U, lattice.edges, basis_2_1, check_herm=False)
            if basis_2_1.Ns < 200:
                e, v = np.linalg.eigh(h_2_1.todense())
            else:
                e, v = h_2_1.eigsh(v0=v0, k=3, tol=1e-8, which="SA")
            S = measure_total_spin(v, basis_2_1)
            # print(e, S)
            # total_spin_grid[i, j] = S
            v0 = v[:, 0]
            table.append((γ, U, e, S))
            with open(filename, "a") as f:
                f.write("{}\t{}\t{}\t{}\n".format(γ, U, e[0], S[0], *e[1:]))
        return
    # np.savetxt("table.dat", np.asarray(table))
    # np.savetxt("gamma_grid.dat", γ_grid)
    # np.savetxt("U_grid.dat", U_grid)
    # np.savetxt("total_spin_grid.dat", total_spin_grid)


def tune_β1_and_U(lattice):
    γ = 1
    # β1 = 1
    β2 = 1
    number_sites = lattice.number_sites
    number_fermions = number_sites - 1
    number_down = number_fermions // 2
    number_up = number_fermions - number_down
    basis = spinful_fermion_basis_general(number_sites, Nf=(number_up, number_down))

    β1_grid = np.linspace(0, 1, num=21)
    U_grid = np.linspace(0, 100, num=41)
    # total_spin_grid = np.zeros((len(γ_grid), len(U_grid)), dtype=np.float64)
    table = []
    filename = "2x2_β1_U.dat"
    with open(filename, "w") as f:
        f.write("# β₁\tU\tE\tS\n")
    for i, β1 in enumerate(β1_grid):
        with open(filename, "a") as f:
            if i != 0:
                f.write("\n")
        v0 = None
        for j, U in enumerate(U_grid):
            h = make_hamiltonian(-β1, -β2, -γ, U, lattice.edges, basis, check_herm=False)
            e, v = h.eigsh(v0=v0, k=1, tol=1e-8, which="SA")
            S = measure_total_spin(v, basis)
            # total_spin_grid[i, j] = S
            v0 = v
            table.append((γ, U, e, S))
            with open(filename, "a") as f:
                f.write("{}\t{}\t{}\t{}\n".format(β1, U, e, S))
    # np.savetxt("table.dat", np.asarray(table))
    # np.savetxt("gamma_grid.dat", γ_grid)
    # np.savetxt("U_grid.dat", U_grid)
    # np.savetxt("total_spin_grid.dat", total_spin_grid)


def tune_β2_and_U(lattice):
    γ = 1
    β1 = 1
    # β2 = 1
    number_sites = lattice.number_sites
    number_fermions = number_sites - 1
    number_down = number_fermions // 2
    number_up = number_fermions - number_down
    basis = spinful_fermion_basis_general(number_sites, Nf=(number_up, number_down))

    β2_grid = np.linspace(0, 1, num=21)
    U_grid = np.linspace(0, 100, num=41)
    # total_spin_grid = np.zeros((len(γ_grid), len(U_grid)), dtype=np.float64)
    table = []
    filename = "2x2_β₂_U.dat"
    with open(filename, "w") as f:
        f.write("# β₂\tU\tE\tS\n")
    for i, β2 in enumerate(β2_grid):
        with open(filename, "a") as f:
            if i != 0:
                f.write("\n")
        v0 = None
        for j, U in enumerate(U_grid):
            h = make_hamiltonian(-β1, -β2, -γ, U, lattice.edges, basis, check_herm=False)
            e, v = h.eigsh(v0=v0, k=1, tol=1e-8, which="SA")
            S = measure_total_spin(v, basis)
            # total_spin_grid[i, j] = S
            v0 = v
            table.append((γ, U, e[0], S))
            with open(filename, "a") as f:
                f.write("{}\t{}\t{}\t{}\n".format(β2, U, e[0], S))
    # np.savetxt("table.dat", np.asarray(table))
    # np.savetxt("gamma_grid.dat", γ_grid)
    # np.savetxt("U_grid.dat", U_grid)
    # np.savetxt("total_spin_grid.dat", total_spin_grid)


def nagaoka_ferromagnetism():
    tune_γ_and_U(square_2x2())
    # tune_β1_and_U(square_2x2())
    # tune_β2_and_U(square_2x2())


def metal_insulator(lattice):
    γ = 1
    β1 = 1
    β2 = 0.0
    number_sites = lattice.number_sites
    global_table = []
    for number_fermions in [number_sites - 1, number_sites, number_sites + 1]:
        number_down = number_fermions // 2
        number_up = number_fermions - number_down
        basis = spinful_fermion_basis_general(number_sites, Nf=(number_up, number_down))
        v0 = None
        table = []
        for U in np.linspace(1, 21, num=15):
            h = make_hamiltonian(-β1, -β2, -γ, U, lattice.edges, basis, check_herm=False)
            print("Working on", U, "...")
            e, v = h.eigsh(v0=v0, k=1, tol=1e-8, which="SA")
            # S = measure_total_spin(v, basis)
            # total_spin_grid[i, j] = S
            v0 = v
            table.append((U, e[0]))
            # with open(filename, "a") as f:
            #     f.write("{}\t{}\t{}\t{}\n".format(β2, U, e[0], S))
        global_table.append(table)

    gap = []
    for ((U, emin), (_, e), (_, eplus)) in zip(*global_table):
        gap.append((U, 0.5 * (emin + eplus - 2 * e)))

    filename = "gap_{}_γ={}_β1={}_β2={}_U.dat".format(number_sites, γ, β1, β2)
    with open(filename, "w") as f:
        f.write("# U\tΔ\n")
        for (U, d) in gap:
            f.write("{}\t{}\n".format(U, d))


def superconductivity(lattice):
    γ = 1
    β1 = 0
    β2 = 1
    number_sites = lattice.number_sites
    global_table = []
    for number_fermions in [number_sites - 2, number_sites - 1, number_sites]:
        number_down = number_fermions // 2
        number_up = number_fermions - number_down
        basis = spinful_fermion_basis_general(number_sites, Nf=(number_up, number_down))
        v0 = None
        table = []
        for U in np.linspace(1, 21, num=15):
            h = make_hamiltonian(-β1, -β2, -γ, U, lattice.edges, basis, check_herm=False)
            print("Working on", U, "...")
            e, v = h.eigsh(v0=v0, k=1, tol=1e-8, which="SA")
            # S = measure_total_spin(v, basis)
            # total_spin_grid[i, j] = S
            v0 = v
            table.append((U, e[0]))
            # with open(filename, "a") as f:
            #     f.write("{}\t{}\t{}\t{}\n".format(β2, U, e[0], S))
        global_table.append(table)

    energies = []
    for ((U, e2), (_, e1), (_, e0)) in zip(*global_table):
        energies.append((U, e2, e1, e0))

    filename = "energies_{}_γ={}_β1={}_β2={}_U.dat".format(number_sites, γ, β1, β2)
    with open(filename, "w") as f:
        f.write("# U\tEₙ₋₂\tEₙ₋₁\tEₙ\n")
        for t in energies:
            f.write("{}\t{}\t{}\t{}\n".format(*t))


def dehollain_2020():
    number_sites = 4
    lattice = square_2x2()
    U = 2900

    def sweep(basis):
        table = []
        for t in np.linspace(0, 200, num=11):
            h = make_hamiltonian(-t, -t, 0, U, lattice.edges, basis).tocsr()
            # h = normal_hubbard(t, U, lattice.edges, basis).tocsr()
            e, v = scipy.sparse.linalg.eigsh(h, k=2, tol=1e-14, which="SA")
            if t == 20:
                v[np.abs(v) < 1e-14] = 0
                print(v)
            print(measure_total_spin(v[:, 0], basis), measure_total_spin(v[:, 1], basis))
            table.append((t, e + 2 * t))
        return table

    basis = quspin.basis.spinful_fermion_basis_general(number_sites, Nf=(2, 1))
    print(basis)
    table = sweep(basis)
    print(table)

    # basis = quspin.basis.spinful_fermion_basis_general(number_sites, Nf=(3, 0))
    # table = sweep(basis)
    # print(table)


def energy_scaling():
    β1 = 1
    β2 = 1
    γ = 1
    for lattice in [square_4(), square_5(), square_8(), square_9(), square_10()]:
        number_fermions = lattice.number_sites
        number_down = number_fermions // 2
        number_up = number_fermions - number_down
        basis = spinful_fermion_basis_general(number_sites, Nf=(number_up, number_down))
        v0 = None
        h = make_hamiltonian(-β1, -β2, -γ, U, lattice.edges, basis, check_herm=False)
        e, v = scipy.sparse.linalg.eigsh(h, k=1, tol=1e-7, which="SA")
        print(lattice.number_sites, e[0])

def run_tests():
    # 0 1 2
    # 3 4 5
    # 6 7 8
    print(square_9().edges)
    assert square_9().edges == sorted(
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
    # energy_scaling()
    # superconductivity(square_3x3())
    # metal_insulator(square_3x3())
    # nagaoka_ferromagnetism()
    # dehollain_2020()
    # run_tests()
