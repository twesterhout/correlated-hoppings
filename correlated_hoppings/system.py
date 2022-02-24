import argparse
import h5py
from loguru import logger
import numpy as np
import os
import quspin
from quspin.basis import spinful_fermion_basis_general
import re
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


def square_6() -> SquareLattice:
    return SquareLattice(6, simple_square_lattice_edges((3, 2)))


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
    #          0  1  2
    #          3  4  5
    #          6  7  8
    # fmt: off
    return SquareLattice(9, [(0, 1), (0, 3), (1, 2), (1, 4),
                             (0, 2), (2, 5), (3, 4), (3, 6),
                             (4, 5), (4, 7), (3, 5), (5, 8),
                             (6, 7), (0, 6), (7, 8), (1, 7),
                             (6, 8), (2, 8)])
    # fmt: on
    # return SquareLattice(9, simple_square_lattice_edges((3, 3)))


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


def square_12() -> SquareLattice:
    return SquareLattice(12, simple_square_lattice_edges((4, 3)))


def square_13() -> SquareLattice:
    #
    #             0  1
    #          2  3  4  5
    #       6  7  8  9 10
    #            11 12
    #
    #                           0  1
    #                        2  3  4  5
    #                     6  7  8  9 10  0  1
    #                     0  1 11 12  2  3  4  5
    #                  2  3  4  5  6  7  8  9 10
    #               6  7  8  9 10  0  1 11 12
    #                    11 12  2  3  4  5
    #                        6  7  8  9 10
    #                             11 12
    #
    #
    # fmt: off
    return SquareLattice(13, [(0, 1), (0, 3), (1, 11), (1, 4),
                              (2, 3), (2, 7), (3, 4), (3, 8),
                              (4, 5), (4, 9), (5, 6), (5, 10),
                              (6, 7), (0, 6), (7, 8), (1, 7),
                              (8, 9), (8, 11), (9, 10), (9, 12),
                              (0, 10), (2, 10), (11, 12), (5, 11),
                              (2, 12), (6, 12)])
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


def total_spin_squared_fn(basis):
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

    def measure(v):
        if v.ndim == 1:
            v = v.reshape(-1, 1)
        return np.sum(v.conj() * S.dot(v), axis=0)
        # print(rs)
        # return [0.5 * (-1 + np.sqrt(1 + r)) for r in rs]

    return measure


def list_existing(lattice):
    prefix = "data/{}".format(lattice.number_sites)
    os.makedirs(prefix, exist_ok=True)

    records = []
    for f in os.listdir(prefix):
        m = re.match(r"^U=(.+)_beta1=(.+)_beta2=(.+)_gamma=(.+)$", f)
        if m is not None:
            U = float(m.group(1))
            β1 = float(m.group(2))
            β2 = float(m.group(3))
            γ = float(m.group(4))
            # Let's reconstruct t and F
            F = γ - β2
            t = β2 - F
            assert np.isclose(β1, t + 3 * F)
            assert np.isclose(β2, t + F)
            assert np.isclose(γ, t + 2 * F)
            number_fermions = lattice.number_sites - 1
            path = os.path.join(prefix, f, "Nf={}".format(number_fermions), "eigenvectors.h5")
            records.append((F, U, t, path))
    records = sorted(records)
    return records


def determine_superposition(S, number_fermions):
    valid_spin_values = np.asarray(list(range(number_fermions, -1, -2))) / 2
    valid_spin_values = valid_spin_values[::-1]
    possible = []
    for s1 in valid_spin_values:
        for s2 in valid_spin_values:
            if s1 < s2:
                a = (S - s2 * (s2 + 1)) / (s1 * (s1 + 1) - s2 * (s2 + 1))
                if 0 < a and a < 1:
                    possible.append((a, s1, s2))
    return possible


def analyze_total_spin(lattice):
    number_sites = lattice.number_sites
    number_fermions = number_sites - 1
    number_down = number_fermions // 2
    number_up = number_fermions - number_down
    basis = spinful_fermion_basis_general(number_sites, Nf=(number_up, number_down))
    S2 = total_spin_squared_fn(basis)
    prefix = "data/{}".format(lattice.number_sites)
    os.makedirs(prefix, exist_ok=True)
    output = os.path.join(prefix, "total_spin.dat")
    with open(output, "w") as f:
        f.write("# F\tU\tt\tS\n")
    valid_spin_values = np.asarray(list(range(number_fermions, -1, -2))) / 2
    valid_spin_values = valid_spin_values[::-1]
    logger.info("Valid spin values: {}", valid_spin_values)

    def is_valid_spin(s):
        return np.sum(np.isclose(valid_spin_values, s, rtol=1e-5, atol=1e-6)) > 0

    prev_F = None
    prev_U = None
    for (F, U, t, path) in list_existing(lattice):
        if prev_F is not None and not np.isclose(F, prev_F, rtol=1e-4, atol=1e-5):
            with open(output, "a") as f:
                f.write("\n")
        prev_F = F
        prev_U = U
        with h5py.File(path, "r") as f:
            e = f["/hamiltonian/eigenvalues"][:]
            v = f["/hamiltonian/eigenvectors"][:]
        if not np.isclose(e[0], e[1], atol=1e-5, rtol=1e-4):
            r = [0.5 * (-1 + np.sqrt(1 + r)) for r in S2(v)]
        else:
            print(e)
            original = [0.5 * (-1 + np.sqrt(1 + r)) for r in S2(v)]
            r = []
            for s in original:
                if is_valid_spin(s):
                    r.append(s)
                else:
                    print(s)
                    print(determine_superposition(s, number_fermions))
                    exit(1)
                    r.append(s)
        with open(output, "a") as f:
            f.write("{}\t{}\t{}\t{}\t{}\n".format(F, U, t, *(r[:2])))


def phase_transition_boundaries(lattice, F, U_min=0, U_max=20, t=1, tol=0.1):
    number_sites = lattice.number_sites
    number_fermions = number_sites - 1
    number_down = number_fermions // 2
    number_up = number_fermions - number_down
    basis = spinful_fermion_basis_general(number_sites, Nf=(number_up, number_down))
    S2 = total_spin_squared_fn(basis)

    valid_spin_values = np.asarray(list(range(number_fermions, -1, -2)))
    valid_spin_values = valid_spin_values[::-1]

    def is_valid_spin(s):
        return np.sum(np.isclose(valid_spin_values, s, rtol=1e-5, atol=1e-6)) > 0

    v0 = None
    number_evaluations = 0

    def compute_spin(U):
        logger.debug("Computing t={}, F={}, U={} ...", t, F, U)
        nonlocal v0
        nonlocal number_evaluations
        L = t + F
        β1 = L + 2 * F
        β2 = L
        γ = L + F
        number_evaluations += 1
        e, v = diagonalize_one(β1, β2, γ, U, lattice, lattice.number_sites - 1, v0=v0)
        v0 = v[:, 0]
        r = [(-1 + np.sqrt(1 + r)) for r in S2(v)]
        if is_valid_spin(r[0]):
            logger.info("Total spin is {}", round(r[0]) / 2)
            return int(round(r[0]))
        else:
            logger.error("Invalid spin: t={}, F={}, U={}, r={}", t, F, U, r)
            possible = determine_superposition(r[0], number_fermions)
            print(e, possible)
            (_, s1, s2) = possible[0]
            return int(round(2 * s1))

    def analyze_region(U_left, S_left, U_right, S_right):
        # logger.debug("Analyzing [{}, {}] ...", U_left, U_right)
        if S_left == S_right:
            return [(U_left, U_right, S_left)]
        # assert S_left < S_right
        U_mid = (U_left + U_right) / 2
        S_mid = compute_spin(U_mid)
        # if not (S_left <= S_mid and S_mid <= S_right):
        #     logger.error(":( U_left={}, S_left={}, U_mid={}, S_mid={}, U_right={}, S_right={}",
        #             U_left, S_left, U_mid, S_mid, U_right, S_right)
        if abs(U_left - U_right) < 2 * tol:
            return [(U_left, U_mid, S_left), (U_mid, U_right, S_right)]
        left_region = analyze_region(U_left, S_left, U_mid, S_mid)
        right_region = analyze_region(U_mid, S_mid, U_right, S_right)
        return left_region + right_region

    S_left = compute_spin(U_min)
    S_right = compute_spin(U_max)
    intervals = analyze_region(U_min, S_left, U_max, S_right)
    intervals = [(U_left, U_right, S / 2) for (U_left, U_right, S) in intervals]

    combined_intervals = []
    current = None
    for i in intervals:
        if current is None:
            current = i
            continue
        if current[2] == i[2]:
            current = (current[0], i[1], current[2])
        else:
            combined_intervals.append(current)
            current = i
    combined_intervals.append(current)
    return combined_intervals, number_evaluations


def as_polygon(points, center):
    points = np.asarray(points)
    if center is None:
        center = np.mean(points, axis=0)
    deltas = points - center
    deltas = deltas[:, 0] + 1j * deltas[:, 1]
    angles = np.angle(deltas)
    order = np.argsort(angles)
    points = list(points[order])
    if len(points) > 2:
        points.append(points[0])

def polygon_to_file(points, output):
    with open(output, "w") as f:
        for (x, y) in points:
            f.write("{}\t{}\n".format(x, y))



def analyze_spin_phases(lattice, F_count=40, F_min=-1, F_max=0.5, U_min=0, U_max=20, t=1, tol=0.1, suffix=""):
    prefix = "data/{}".format(lattice.number_sites)
    os.makedirs(prefix, exist_ok=True)
    output = os.path.join(prefix, "phases{}.dat".format(suffix))
    with open(output, "w") as f:
        f.write("# F\tU\tS\n")
    ε = 2e-2
    number_evaluations = 0
    records = []
    for F in np.linspace(F_min, F_max, F_count):
        intervals, k = phase_transition_boundaries(
            lattice, F, U_min=U_min, U_max=U_max, t=t, tol=tol
        )
        number_evaluations += k
        with open(output, "a") as f:
            for (U_left, U_right, S) in intervals:
                f.write("{}\t{}\t{}\n".format(F, U_left + ε, S))
                f.write("{}\t{}\t{}\n".format(F, U_right - ε, S))
        records.append((F, intervals))
    logger.info("{} evaluations performned to obtain the phase diagram", number_evaluations)
    return records


def tune_F_and_U(lattice):
    t = 1
    F_grid = np.linspace(-1, 0.5, num=41)
    U_grid = np.linspace(0, 20, num=11)
    number_sites = lattice.number_sites
    number_fermions = number_sites - 1
    number_down = number_fermions // 2
    number_up = number_fermions - number_down
    basis = spinful_fermion_basis_general(number_sites, Nf=(number_up, number_down))
    S2 = total_spin_squared_fn(basis)
    prefix = "data/{}".format(lattice.number_sites)
    # os.makedirs(prefix, exist_ok=True)
    # output = os.path.join(prefix, "total_spin.dat")
    # with open(output, "w") as f:
    #     f.write("# F\tU\tt\tS\n")
    v0 = None
    # valid_spin_values = np.asarray(list(range(number_fermions, -1, -2))) / 2
    # valid_spin_values = valid_spin_values[::-1]
    # logger.info("Valid spin values: {}", valid_spin_values)

    # def is_valid_spin(s):
    #     return np.sum(np.isclose(valid_spin_values, s, rtol=1e-5, atol=1e-6)) > 0

    for i, F in enumerate(F_grid):
        # with open(output, "a") as f:
        #     if i != 0:
        #         f.write("\n")
        L = t + F
        β1 = L + 2 * F
        β2 = L
        γ = L + F
        for U in U_grid:
            e, v = diagonalize_one(β1, β2, γ, U, lattice, lattice.number_sites - 1, v0=v0)
            v0 = v[:, 0]
            # if not np.isclose(e[0], e[1], atol=1e-5, rtol=1e-4):
            #     r = [0.5 * (-1 + np.sqrt(1 + r)) for r in S2(v)]
            # else:
            #     print(e)
            #     original = [0.5 * (-1 + np.sqrt(1 + r)) for r in S2(v)]
            #     r = []
            #     for s in original:
            #         if is_valid_spin(s):
            #             r.append(s)
            #         else:
            #             print(s)
            #             r.append(s)
            #             # r.append(np.inf)
            # with open(output, "a") as f:
            #     f.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(F, U, t, *r))


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


def diagonalize_one(
    β1: float,
    β2: float,
    γ: float,
    U: float,
    lattice,
    number_fermions: int,
    force: bool = False,
    v0=None,
):
    prefix = "data/{}/U={:09.5f}_beta1={:07.5f}_beta2={:07.5f}_gamma={:07.5f}/Nf={}/".format(
        lattice.number_sites, U, β1, β2, γ, number_fermions
    )
    os.makedirs(prefix, exist_ok=True)
    output = os.path.join(prefix, "eigenvectors.h5")
    if os.path.exists(output) and not force:
        logger.info("'{}' already exists, skipping ...", output)
        with h5py.File(output, "r") as f:
            e = f["/hamiltonian/eigenvalues"][:]
            v = f["/hamiltonian/eigenvectors"][:]
        return e, v
    logger.info("Building the basis ...")
    number_down = number_fermions // 2
    number_up = number_fermions - number_down
    basis = spinful_fermion_basis_general(lattice.number_sites, Nf=(number_up, number_down))
    logger.info("Building the Hamiltonian ...")
    h = make_hamiltonian(β1, β2, γ, U, lattice.edges, basis, check_herm=False)
    logger.info("Diagonalizing ...")
    e, v = h.eigsh(h, v0=v0, k=4, tol=1e-8, which="SA")
    logger.info("Ground state energy: {}", e.tolist())
    with h5py.File(output, "w") as f:
        g = f.create_group("/hamiltonian")
        g["eigenvectors"] = v
        g["eigenvalues"] = e
        g.attrs["β₁"] = β1
        g.attrs["β₂"] = β2
        g.attrs["γ"] = γ
        g.attrs["U"] = U
    return e, v


def diagonalize_command():
    parser = argparse.ArgumentParser(description="Perform exact diagonalization.")
    parser.add_argument("-n", type=int, required=True, help="System size")
    parser.add_argument("-U", type=float, required=True, help="U")
    parser.add_argument("--beta1", type=float, required=True, help="β₁")
    parser.add_argument("--beta2", type=float, required=True, help="β₂")
    parser.add_argument("--gamma", type=float, required=True, help="γ")
    parser.add_argument("--occupation", type=int, required=True, help="Number of electrons")
    args = parser.parse_args()
    lattice = eval("square_{:d}()".format(args.n))
    diagonalize_one(args.beta1, args.beta2, args.gamma, args.U, lattice, args.occupation)


def analyze_command():
    parser = argparse.ArgumentParser(description="Build phase diagram.")
    parser.add_argument("-n", type=int, required=True, help="System size")
    parser.add_argument("--F", type=float, required=True, help="F")
    args = parser.parse_args()
    lattice = eval("square_{:d}()".format(args.n))
    analyze_spin_phases(lattice, F_count=1, F_min=args.F, suffix="_{}".format(args.F))


def energy_scaling():
    β1 = 1
    β2 = 1
    γ = 1
    U = 2
    for lattice in [
        square_16()
    ]:  # [square_4(), square_5(), square_6(), square_8(), square_9(), square_10(), square_12(), square_13()]:
        number_fermions = lattice.number_sites
        number_down = number_fermions // 2
        number_up = number_fermions - number_down
        logger.info("Building the basis ...")
        basis = spinful_fermion_basis_general(lattice.number_sites, Nf=(number_up, number_down))
        v0 = None
        logger.info("Building the Hamiltonian ...")
        h = make_hamiltonian(-β1, -β2, -γ, U, lattice.edges, basis, check_herm=False)
        logger.info("Diagonalizing ...")
        e, v = h.eigsh(h, k=1, tol=1e-7, which="SA")
        logger.info("Done!")
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
    analyze_command()
    # diagonalize_command()
    # energy_scaling()
    # superconductivity(square_3x3())
    # metal_insulator(square_3x3())
    # nagaoka_ferromagnetism()
    # dehollain_2020()
    # run_tests()
