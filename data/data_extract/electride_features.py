"""CIF feature extraction functions."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from itertools import combinations, product
import json
from pathlib import Path
import warnings

import numpy as np
from ase.data import covalent_radii
from pymatgen.core import Element, Structure
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from scipy import ndimage
from scipy.optimize import minimize
from scipy.spatial import ConvexHull, QhullError

VERSION = "0.1.0"
CRITICAL_CAGE_A = {
    "Li": (2.71, 2.25), "Na": (2.88, 2.52), "K": (3.47, 3.15),
    "Rb": (3.63, 3.35), "Cs": (3.94, 3.70), "Be": (1.80, 1.48),
    "Mg": (2.17, 1.90), "Ca": (2.74, 2.50), "Sr": (2.95, 2.73),
    "Ba": (3.27, 3.07),
}
FIXED_ION_STATES = {
    "Li": [1], "Na": [1], "K": [1], "Rb": [1], "Cs": [1],
    "Be": [2], "Mg": [2], "Ca": [2], "Sr": [2], "Ba": [2],
    "F": [-1], "Cl": [-1], "Br": [-1], "I": [-1], "O": [-2],
    "N": [-3], "P": [-3], "As": [-3], "S": [-2], "Se": [-2],
    "Te": [-2], "Y": [3], "Sc": [3], "La": [3], "Al": [3],
    "H": [-1, 1], "B": [-3, 3], "C": [-4, 4],
}


def _load(cif_path):
    path = Path(cif_path)
    if path.suffix.lower() != ".cif":
        raise ValueError("Supply a CIF file; electronic fields and other file types are not inputs.")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        parser = CifParser(path, frac_tolerance=0, check_cif=False)
        structures = parser.parse_structures(primitive=False, on_error="raise")
    if len(structures) != 1:
        raise ValueError("Provide exactly one crystal structure per CIF.")
    structure = structures[0]
    if not structure.is_ordered or any(abs(sum(s.species.values()) - 1) > 1e-8 for s in structure):
        raise ValueError("Disordered or partially occupied CIFs need an explicit ordered configuration.")
    if not np.isfinite(structure.lattice.matrix).all() or not np.isfinite(structure.frac_coords).all():
        raise ValueError("Cell and coordinates must be finite.")
    if structure.volume <= 0 or len(structure) == 0:
        raise ValueError("The structure must have sites and a nonzero cell volume.")
    structure = Structure(
        structure.lattice, [s.specie.symbol for s in structure],
        structure.frac_coords % 1, to_unit_cell=True,
    )
    return structure, list(dict.fromkeys(str(w.message) for w in caught))


def _radii(structure):
    return np.array([covalent_radii[s.specie.Z] for s in structure])


def _stats(values):
    a = np.array(values, dtype=float)
    if not len(a):
        return None
    return {"min": float(a.min()), "mean": float(a.mean()), "max": float(a.max()),
            "std": float(a.std())}


def structure_summary(cif_path: str) -> dict:
    """Describe the supplied configuration without relaxing or standardizing it."""
    s, notes = _load(cif_path)
    elements = [site.specie for site in s]
    x = [e.X for e in elements if e.X is not None and np.isfinite(e.X)]
    try:
        symmetry = SpacegroupAnalyzer(s, symprec=0.001)
        sg = {"symbol": symmetry.get_space_group_symbol(),
              "number": symmetry.get_space_group_number(), "symprec_A": .001}
    except (ValueError, TypeError, RuntimeError) as exc:
        sg = {"unavailable": str(exc)}
    return {
        "formula": s.composition.reduced_formula,
        "composition_per_cell": s.composition.get_el_amt_dict(),
        "site_count": len(s), "cell_matrix_A": s.lattice.matrix.tolist(),
        "volume_A3": float(s.volume), "volume_per_atom_A3": float(s.volume / len(s)),
        "density_g_cm3": float(s.density), "space_group": sg,
        "cell_condition_number": float(np.linalg.cond(s.lattice.matrix)),
        "metal_site_fraction": sum(e.is_metal for e in elements) / len(s),
        "alkali_alkaline_site_fraction": sum(e.is_alkali or e.is_alkaline for e in elements) / len(s),
        "transition_metal_site_fraction": sum(e.is_transition_metal for e in elements) / len(s),
        "f_block_site_fraction": sum(e.is_lanthanoid or e.is_actinoid for e in elements) / len(s),
        "pauling_electronegativity": _stats(x), "parser_warnings": notes,
        "interpretation": "Density and volume describe packing, not pressure or stability. Symmetry is tolerance dependent.",
    }


def electron_count_hypotheses(cif_path: str, oxidation_states: dict | None = None) -> dict:
    """Enumerate formal electron residuals without forcing charge neutrality."""
    s, _ = _load(cif_path)
    reduced, factor = s.composition.get_reduced_composition_and_factor()
    counts = reduced.get_el_amt_dict()
    if len(counts) == 1:
        return {"formula": reduced.reduced_formula, "hypotheses": [],
                "residual_range_e_per_formula": None, "neutral_assignment_exists": None,
                "interpretation": "Elemental structures require a metallic/high-pressure route; ionic counting is not diagnostic."}
    options = {}
    for sym in counts:
        e = Element(sym)
        if oxidation_states is not None and sym in oxidation_states:
            states = oxidation_states[sym]
            if not isinstance(states, list) or not states or not all(isinstance(q, (int, float)) and np.isfinite(q) for q in states):
                raise ValueError("Each oxidation-state override must be a nonempty finite numeric list.")
            options[sym] = sorted(set(states))
        elif sym in FIXED_ION_STATES:
            options[sym] = FIXED_ION_STATES[sym]
        elif e.is_transition_metal or e.is_lanthanoid or e.is_actinoid:
            options[sym] = sorted(q for q in e.oxidation_states if q > 0)
        else:
            options[sym] = list(e.common_oxidation_states)
        if not options[sym]:
            return {"formula": reduced.reduced_formula, "hypotheses": [],
                    "residual_range_e_per_formula": None, "neutral_assignment_exists": None,
                    "interpretation": f"No default oxidation hypotheses for {sym}; supply explicit alternatives."}
    ncombinations = int(np.prod([len(v) for v in options.values()]))
    qmin = sum(counts[e] * min(options[e]) for e in counts)
    qmax = sum(counts[e] * max(options[e]) for e in counts)
    attainable = {0.0}
    for sym, n in counts.items():
        attainable = {round(a + n * q, 8) for a in attainable for q in options[sym]}
    hypotheses = []
    for states in product(*options.values()):
        assignment = dict(zip(options, states))
        q = float(sum(counts[e] * assignment[e] for e in counts))
        hypotheses.append({"oxidation_states": assignment,
                           "formal_residual_e_per_formula": q,
                           "formal_residual_e_per_cell": q * factor})
        if len(hypotheses) >= 256:
            break
    return {
        "formula": reduced.reduced_formula, "state_options": options,
        "hypotheses": hypotheses, "total_assignment_count": ncombinations,
        "hypotheses_truncated": ncombinations > len(hypotheses),
        "residual_range_e_per_formula": [float(qmin), float(qmax)],
        "attainable_residuals_e_per_formula": sorted(attainable),
        "neutral_assignment_exists": 0.0 in attainable,
        "interpretation": (
            "A positive sum is formal electron availability under the stated oxidation assignment, "
            "not a measured interstitial occupation. Zero/negative sums do not exclude electrides. "
            "Mixed valence, polyatomic anions, covalent bonding, and metal reduction can change this accounting."
        ),
    }


def _bonds(s, scale=1.2):
    radii = _radii(s)
    i, j, images, distances = s.get_neighbor_list(2 * float(radii.max()) * scale)
    keep = (distances > 1e-7) & (distances <= scale * (radii[i] + radii[j]))
    records = []
    for a, b, image, d in zip(i[keep], j[keep], images[keep], distances[keep]):
        key = (int(a), int(b), *map(int, image))
        reverse = (int(b), int(a), *map(int, -image))
        if key <= reverse:
            records.append((int(a), int(b), image.astype(int), float(d)))
    return records


def bonding_environment(cif_path: str, bond_radius_scale: float = 1.2) -> dict:
    """Describe possible conventional bonding and polyatomic-anion competition."""
    if bond_radius_scale <= 0:
        raise ValueError("bond_radius_scale must be positive.")
    s, _ = _load(cif_path)
    bonds = _bonds(s, bond_radius_scale)
    degrees = np.zeros(len(s), int)
    pair_lengths = defaultdict(list)
    for i, j, image, distance in bonds:
        degrees[i] += 1
        degrees[j] += 1
        pair_lengths["-".join(sorted([s[i].specie.symbol, s[j].specie.symbol]))].append(distance)
    dimers = []
    adjacency = defaultdict(list)
    for i, j, image, d in bonds:
        if s[i].specie.symbol == s[j].specie.symbol and s[i].specie.symbol in {"N", "O"}:
            adjacency[i].append(j)
            adjacency[j].append(i)
    for i, j, image, d in bonds:
        sym = s[i].specie.symbol
        if i != j and sym in {"N", "O"} and s[j].specie.symbol == sym and len(adjacency[i]) == len(adjacency[j]) == 1:
            dimers.append({"element": sym, "site_indices": [i, j], "image": image.tolist(),
                           "length_A": d, "possible_group_charges_e": [0, -1, -2, -3, -4]})
    by_element = {}
    for sym in sorted(s.composition.get_el_amt_dict()):
        by_element[sym] = _stats([degrees[i] for i, site in enumerate(s) if site.specie.symbol == sym])
    return {
        "bond_radius_scale": bond_radius_scale, "unique_bonds_per_cell": len(bonds),
        "geometric_coordination_by_element": by_element,
        "pair_distance_statistics_A": {k: _stats(v) for k, v in sorted(pair_lengths.items())},
        "isolated_N2_O2_motifs": dimers,
        "variable_valence_elements": [e.symbol for e in s.composition.elements if e.is_transition_metal or e.is_lanthanoid or e.is_actinoid],
        "p_block_lone_pair_candidates": [e.symbol for e in s.composition.elements if e.symbol in {"N", "P", "As", "Sb", "Bi", "O", "S", "Se", "Te", "Sn", "Pb"}],
        "interpretation": "Covalent-radius contacts are geometric hypotheses, not bond orders. N2/O2 group charges are alternatives, not length-derived assignments.",
    }


def _grid(s, spacing):
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("grid_spacing_A must be positive and finite.")
    shape = np.maximum(8, np.ceil(np.array(s.lattice.abc) / spacing).astype(int))
    points = np.indices(tuple(shape)).reshape(3, -1).T / shape
    nearest = np.empty(len(points))
    clearance = np.empty(len(points))
    radii = _radii(s)
    for start in range(0, len(points), 2048):
        distances = s.lattice.get_all_distances(points[start:start + 2048], s.frac_coords)
        nearest[start:start + 2048] = distances.min(axis=1)
        clearance[start:start + 2048] = (distances - radii).min(axis=1)
    return shape, points, nearest.reshape(shape), clearance.reshape(shape)


def _enclosure(vectors):
    if len(vectors) < 4:
        return False
    try:
        hull = ConvexHull(vectors)
        return bool(np.all(hull.equations[:, -1] <= 1e-8))
    except QhullError:
        return False


def _site_environment(s, frac, shell_width):
    center = s.lattice.get_cartesian_coords(frac)
    distances = s.lattice.get_all_distances([frac], s.frac_coords)[0]
    nearest = float(distances.min())
    neighbors = s.get_sites_in_sphere(center, nearest + shell_width + 1e-8, include_index=True, include_image=True)
    vectors = np.array([n.coords - center for n in neighbors])
    lengths = np.linalg.norm(vectors, axis=1)
    directions = vectors / lengths[:, None]
    metal = np.array([n.specie.is_metal for n in neighbors], bool)
    metal_vectors = vectors[metal]
    available_ratios = []
    for n, distance in zip(neighbors, lengths):
        if n.specie.symbol in CRITICAL_CAGE_A:
            b3, b2 = CRITICAL_CAGE_A[n.specie.symbol]
            available_ratios.append({"element": n.specie.symbol, "distance_A": float(distance),
                                     "distance_over_critical_3D": float(distance / b3),
                                     "distance_over_critical_2D": float(distance / b2)})
    eig = np.linalg.eigvalsh(directions.T @ directions / len(directions))
    all_elements = [site.specie for site in s]
    metal_distances = [d for d, e in zip(distances, all_elements) if e.is_metal]
    nonmetal_distances = [d for d, e in zip(distances, all_elements) if not e.is_metal]
    return {
        "fractional_coordinates": [float(x) for x in (frac % 1)],
        "nearest_nucleus_A": nearest,
        "covalent_surface_clearance_A": float((distances - _radii(s)).min()),
        "shell_width_A": shell_width, "shell_atom_count": len(neighbors),
        "shell_composition": dict(Counter(n.specie.symbol for n in neighbors)),
        "shell_atoms": [{"element": n.specie.symbol, "distance_A": float(d),
                         "relative_cartesian_A": v.tolist()}
                        for n, d, v in zip(neighbors, lengths, vectors)],
        "shell_metal_fraction": float(metal.mean()),
        "shell_distance_statistics_A": _stats(lengths),
        "shell_distance_coefficient_of_variation": float(lengths.std() / lengths.mean()),
        "angular_second_moment_eigenvalues": eig.tolist(),
        "directional_imbalance": float(np.linalg.norm(directions.mean(axis=0))),
        "inside_shell_convex_hull": _enclosure(vectors),
        "inside_metal_shell_convex_hull": _enclosure(metal_vectors),
        "nearest_metal_A": min(map(float, metal_distances), default=None),
        "nearest_nonmetal_A": min(map(float, nonmetal_distances), default=None),
        "critical_cage_ratios": available_ratios,
        "critical_cage_model_coverage": len(available_ratios) / len(neighbors),
    }


def _metal_motif_sites(s, max_sites, shell_width):
    """Propose centers of metal triangles/tetrahedra, without electronic fields."""
    candidates = []
    seen = set()
    for site in s:
        if not site.specie.is_metal:
            continue
        radius = max(4.0, s.volume ** (1 / 3))
        neighbors = []
        while len(neighbors) < 12:
            neighbors = [n for n in s.get_sites_in_sphere(site.coords, radius)
                         if n.specie.is_metal and n.nn_distance > 1e-6]
            radius *= 2
        neighbors.sort(key=lambda n: (round(float(n.nn_distance), 8),
                                     *np.round(n.coords - site.coords, 8)))
        edge = neighbors[11].nn_distance
        neighbors = [n for n in neighbors if n.nn_distance <= edge + 1e-6]
        positions = np.array([n.coords for n in neighbors])
        for order in (2, 3):
            for selected in combinations(range(len(neighbors)), order):
                vertices = np.vstack([site.coords, positions[list(selected)]])
                centered = vertices - vertices.mean(axis=0)
                if np.linalg.matrix_rank(centered, tol=1e-7) < order:
                    continue
                center = vertices.mean(axis=0)
                frac = s.lattice.get_fractional_coords(center) % 1
                key = tuple(np.round(frac, 6))
                if key in seen:
                    continue
                seen.add(key)
                candidates.append({
                    "frac": frac, "motif_size": order + 1,
                    "motif_species": [site.specie.symbol] + [neighbors[i].specie.symbol for i in selected],
                    "motif_radius_max_A": float(np.linalg.norm(centered, axis=1).max()),
                })
    if not candidates:
        return [], 0
    for start in range(0, len(candidates), 2048):
        batch = candidates[start:start + 2048]
        d = s.lattice.get_all_distances([c["frac"] for c in batch], s.frac_coords)
        for c, nearest in zip(batch, d.min(axis=1)):
            c["empty_motif_balance"] = float(nearest / c["motif_radius_max_A"])
            c["nearest"] = float(nearest)
    candidates.sort(key=lambda c: (-round(c["empty_motif_balance"], 6),
                                   c["motif_radius_max_A"]))
    selected = []
    for candidate in candidates:
        if candidate["nearest"] < 1e-6:
            continue
        if any(s.lattice.get_all_distances([candidate["frac"]], [p["fractional_coordinates"]])[0, 0] < .35 for p in selected):
            continue
        entry = _site_environment(s, candidate["frac"], shell_width)
        entry.update(
            proposal="metal_triangle_or_tetrahedron_centroid",
            source_motif_size=candidate["motif_size"],
            source_motif_species=candidate["motif_species"],
            source_motif_radius_max_A=candidate["motif_radius_max_A"],
            empty_motif_balance=candidate["empty_motif_balance"],
        )
        selected.append(entry)
        if len(selected) >= max_sites:
            break
    return selected, len(candidates)


def _cavities(s, spacing, max_sites, shell_width):
    if max_sites < 1 or shell_width <= 0:
        raise ValueError("max_sites and shell_width_A must be positive.")
    shape, points, nearest, clearance = _grid(s, spacing)
    proposals = []
    for mode, field in [("covalent_clearance", clearance), ("nucleus_distance", nearest)]:
        local = np.argwhere(field >= ndimage.maximum_filter(field, size=3, mode="wrap") - 1e-10)
        local = sorted(local, key=lambda x: float(field[tuple(x)]), reverse=True)
        for index in local[:max_sites * 4]:
            start = index / shape
            def objective(frac):
                d = s.lattice.get_all_distances([np.asarray(frac) % 1], s.frac_coords)[0]
                return -float((d - _radii(s)).min() if mode == "covalent_clearance" else d.min())
            simplex = np.vstack([start, start + np.diag(.3 / shape)])
            opt = minimize(objective, start, method="Nelder-Mead",
                           options={"initial_simplex": simplex, "xatol": 1e-6, "fatol": 1e-6})
            frac = np.asarray(opt.x) % 1
            if any(s.lattice.get_all_distances([frac], [p["frac"]])[0, 0] < .35 for p in proposals):
                continue
            proposals.append({"frac": frac, "mode": mode, "optimizer_success": bool(opt.success),
                              "clearance": float((s.lattice.get_all_distances([frac], s.frac_coords)[0] - _radii(s)).min())})
    proposals.sort(key=lambda p: p["clearance"], reverse=True)
    sites = []
    for p in proposals[:max_sites]:
        entry = _site_environment(s, p["frac"], shell_width)
        entry.update(proposal=p["mode"], optimizer_success=p["optimizer_success"])
        sites.append(entry)
    motif_sites, motif_count = _metal_motif_sites(s, max_sites, shell_width)
    return {
        "grid_spacing_requested_A": spacing, "grid_shape": shape.tolist(),
        "actual_grid_edge_lengths_A": (np.array(s.lattice.abc) / shape).tolist(),
        "sampled_grid_clearance_statistics_A": _stats(clearance.ravel()),
        "max_clearance_A": max((p["clearance"] for p in proposals), default=None),
        "candidate_count_before_output_limit": len(proposals),
        "max_sites": max_sites, "sites_truncated": len(proposals) > max_sites, "sites": sites,
        "metal_motif_sites": motif_sites,
        "metal_motif_candidate_count_before_output_limit": motif_count,
        "metal_motif_sites_truncated": motif_count > len(motif_sites),
        "site_limit_convention": "max_sites applies separately to empty-space maxima and metal-motif centers.",
        "interpretation": (
            "Sites are geometry-derived candidate empty-space centers, not predicted ELF maxima. "
            "Distances minus covalent radii are a size convention, not electron-accessible volume. "
            "Critical cage ratios are literature heuristics for regular same-element s-orbital cages; "
            "mixed/irregular cages and orbital occupation require separate reasoning. "
            "Metal-motif centers use triangles/tetrahedra among each metal's nearest 12 metal neighbors, "
            "including distance ties; centers are ranked by nearest-nucleus distance divided by "
            "the source motif's largest radius, not by an electride likelihood. "
            "Site detection and connectivity are resolution dependent."
        ),
    }, (shape, clearance)


def cavity_geometry(cif_path: str, grid_spacing_A: float = .4,
                    max_sites: int = 24, shell_width_A: float = .5) -> dict:
    """Find empty-space centers and quantify surrounding chemistry and cage shape."""
    s, _ = _load(cif_path)
    return _cavities(s, grid_spacing_A, max_sites, shell_width_A)[0]


def _periodic_components(mask):
    labels, count = ndimage.label(mask, structure=ndimage.generate_binary_structure(3, 1))
    if count == 0:
        return []
    sizes = np.bincount(labels.ravel())
    adjacency = defaultdict(list)
    for axis in range(3):
        first, last = np.take(labels, 0, axis=axis), np.take(labels, -1, axis=axis)
        valid = (first != 0) & (last != 0)
        pairs = np.unique(np.stack([first[valid], last[valid]], axis=1), axis=0)
        step = np.zeros(3, dtype=int)
        step[axis] = -1
        for a, b in pairs:
            adjacency[int(a)].append((int(b), step))
            adjacency[int(b)].append((int(a), -step))
    visited = set()
    components = []
    for root in range(1, count + 1):
        if root in visited:
            continue
        offsets = {root: np.zeros(3, dtype=int)}
        visited.add(root)
        queue = deque([root])
        cycles, size = [], 0
        while queue:
            a = queue.popleft()
            size += int(sizes[a])
            for b, step in adjacency[a]:
                proposed = offsets[a] + step
                if b not in offsets:
                    offsets[b] = proposed
                    visited.add(b)
                    queue.append(b)
                else:
                    cycle = proposed - offsets[b]
                    if np.any(cycle):
                        cycles.append(cycle)
        basis = []
        for cycle in cycles:
            if np.linalg.matrix_rank(np.array(basis + [cycle])) > len(basis):
                basis.append(cycle)
        components.append({"voxel_count": size, "cell_volume_fraction": size / mask.size,
                           "periodic_rank": len(basis),
                           "independent_winding_vectors": [v.tolist() for v in basis]})
    return sorted(components, key=lambda c: c["voxel_count"], reverse=True)


def _connectivity(shape, clearance, margins):
    profiles = []
    for margin in margins:
        if not np.isfinite(margin):
            raise ValueError("clearance margins must be finite.")
        components = _periodic_components(clearance >= margin)
        profiles.append({
            "clearance_margin_A": margin,
            "void_cell_fraction": float((clearance >= margin).mean()),
            "component_count": len(components),
            "max_periodic_rank": max((c["periodic_rank"] for c in components), default=None),
            "largest_components": components[:10],
        })
    return {
        "grid_shape": shape.tolist(), "profiles": profiles,
        "interpretation": "Periodic rank 0/1/2/3 means isolated pocket/channel/sheet/network in the geometric mask. Boundary crossing alone is not percolation. This is not electronic dimensionality.",
    }


def void_connectivity(cif_path: str, grid_spacing_A: float = .4,
                      clearance_margins_A: list | None = None) -> dict:
    """Measure periodic void topology over several geometric radius conventions."""
    s, _ = _load(cif_path)
    shape, _, _, clearance = _grid(s, grid_spacing_A)
    margins = [0.0, .25, .5, .75] if clearance_margins_A is None else clearance_margins_A
    return _connectivity(shape, clearance, margins)


def extract_all_features(cif_path: str, grid_spacing_A: float = .4,
                         max_sites: int = 24) -> dict:
    """Return one structured evidence report for the supplied CIF."""
    s, _ = _load(cif_path)
    cavities, (shape, clearance) = _cavities(s, grid_spacing_A, max_sites, .5)
    return {
        "tool_version": VERSION, "input_kind": "CIF geometry and species only",
        "structure": structure_summary(cif_path),
        "electron_accounting": electron_count_hypotheses(cif_path),
        "bonding": bonding_environment(cif_path),
        "cavities": cavities,
        "void_topology": _connectivity(shape, clearance, [0.0, .25, .5, .75]),
        "not_determined": [
            "electride identity or probability", "ELF or charge density",
            "interstitial electron occupation", "magnetic state", "pressure",
            "thermodynamic stability", "experimental accessibility", "literature novelty",
        ],
    }


TOOL_FUNCTIONS = {
    fn.__name__: fn for fn in (
        structure_summary, electron_count_hypotheses, bonding_environment,
        cavity_geometry, void_connectivity, extract_all_features,
    )
}


def dispatch_tool(name: str, arguments: dict) -> dict:
    """Dispatch only registered scientific functions; arguments are structured JSON."""
    if name not in TOOL_FUNCTIONS:
        raise ValueError(f"Unknown tool: {name}")
    result = TOOL_FUNCTIONS[name](**arguments)
    json.dumps(result, allow_nan=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cif")
    parser.add_argument("--grid-spacing", type=float, default=.4)
    parser.add_argument("--max-sites", type=int, default=24)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = extract_all_features(args.cif, args.grid_spacing, args.max_sites)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")
