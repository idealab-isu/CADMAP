"""Initialize NURBS control points from MeshMap's versioned, bound field.

Face identity comes from the original STEP topology, not a nearest-face search.
UV is reprojected on that same face after analytic-to-NURBS conversion because
conversion can change the surface parameterization even when bounds agree.
"""
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np


def load_bound_field(cad_path, mapping_path, fit_path):
    from EigenFit.io.correspondence import sha256, mesh_hash, text
    with np.load(fit_path, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    if str(data['schema_version']) != 'meshmap-bound-fit/1':
        raise ValueError('Unsupported MeshMap bound-field schema')
    if str(data['length_unit']) != 'mm':
        raise ValueError('Expected millimetres')
    if str(data['modal_mapping_sha256']) != sha256(mapping_path):
        raise ValueError('Mapping hash differs from the bound fit')
    nodes, tris = data['vertices_original'], data['triangles']
    if not np.array_equal(data['vertex_ids'], np.arange(len(nodes))):
        raise ValueError('Bound-fit vertex order is not canonical')
    digest = mesh_hash(nodes, tris)
    if str(data['modal_mesh_sha256']) != digest:
        raise ValueError('Bound-fit mesh hash mismatch')
    with h5py.File(mapping_path, 'r') as mapping:
        if text(mapping.attrs['schema_version']) != 'meshmap-modal-cad/1':
            raise ValueError('Unsupported modal/CAD schema')
        if text(mapping.attrs['length_unit']) != 'mm' or int(mapping.attrs['index_base']) != 0:
            raise ValueError('Expected millimetres and zero-based indices')
        if text(mapping.attrs['cad_sha256']) != sha256(cad_path):
            raise ValueError('Source STEP changed after MeshMap correspondence')
        if text(mapping.attrs['modal_mesh_sha256']) != digest:
            raise ValueError('Mapping belongs to a different modal mesh/order')
        if not np.array_equal(mapping['mesh/vertices_mm'][:], nodes) or not np.array_equal(mapping['mesh/triangles'][:], tris):
            raise ValueError('Mapping arrays disagree with the bound fit')
        data['node_face_pairs'] = mapping['cad/node_face_pairs'][:]
        data['cad_kernel_version'] = text(mapping.attrs['cad_kernel_version'])
        data['tolerance_mm'] = float(mapping.attrs['tolerance_mm'])
    pairs = data['node_face_pairs']
    if pairs.ndim != 2 or pairs.shape[1] != 2 or not np.issubdtype(pairs.dtype, np.integer):
        raise ValueError('Invalid node/face pairs')
    if not np.array_equal(np.unique(pairs[:, 0]), np.arange(len(nodes))):
        raise ValueError('Every sampled node needs a CAD face membership')
    T, delta = data['T_original_to_scan'], data['displacement_original']
    if T.shape != (4, 4) or delta.shape != nodes.shape or not np.isfinite(T).all() or not np.isfinite(delta).all():
        raise ValueError('Invalid registration/displacement arrays')
    R, t = T[:3, :3], T[:3, 3]
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-8) or not np.isclose(np.linalg.det(R), 1.) or not np.allclose(T[3], [0, 0, 0, 1]):
        raise ValueError('Registration must be a proper rigid transform')
    for actual, expected in [(data['vertices_registered_scan'], nodes @ R.T + t),
                             (data['vertices_fitted_scan'], (nodes + delta) @ R.T + t)]:
        if not np.allclose(actual, expected, atol=1e-7, rtol=0):
            raise ValueError('Field and registration coordinate frames disagree')
    return data


def mapped_footprints(solid, geometry, nodes, pairs, tolerance):
    from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf
    from OCC.Core.gp import gp_Pnt
    from cadmap.deformation.mesh_transfer import NodeFootprint
    face_map = {}
    for source_id, original in enumerate(geometry.shapes['face']):
        matches = [i for i, f in enumerate(solid.descendent) if f.TopoDS.IsSame(original)]
        if len(matches) != 1:
            raise ValueError('CADMap topology cannot uniquely identify original face %d' % source_id)
        face_map[source_id] = matches[0]
    surfaces = [f.geom._to_geomsurface() for f in solid.descendent]
    prints = []
    for node, source_face in pairs:
        if int(source_face) not in face_map:
            raise ValueError('Unknown CAD face ID')
        f = face_map[int(source_face)]
        projection = GeomAPI_ProjectPointOnSurf(gp_Pnt(*map(float, nodes[node])), surfaces[f], 0., 1., 0., 1.)
        if not projection.NbPoints() or projection.LowerDistance() > tolerance:
            raise ValueError('Node %d cannot be projected onto converted face %d' % (node, f))
        u, v = projection.LowerDistanceParameters()
        prints.append(NodeFootprint(int(node), f, float(u), float(v), float(projection.LowerDistance())))
    return prints


def initialize_from_meshmap(cad_path, mapping_path, fit_path, step_path, archive_path,
                           report_path, output_mode='target_coord', continuity_samples=24,
                           smoothing_weight=.001, ridge=.01):
    from EigenFit.io.cad_geometry import StepGeometry
    from EigenFit.io.correspondence import atomic_json, sha256
    from cadmap.geometry.topology import Topology
    from cadmap.conversion.occ import SolidConvertor, add_geom_to_solid
    from cadmap.conversion.export import solid_to_occ_shape, write_step, DEFAULT_SEW_TOL
    from cadmap.deformation.mesh_transfer import deform_solid, interface_gaps, surface_basis_matrix
    from cadmap.deformation.validity import brep_status
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.gp import gp_Trsf

    if output_mode not in ('target_coord', 'original_coord'):
        raise ValueError('output_mode must be target_coord or original_coord')
    if continuity_samples < 2 or min(smoothing_weight, ridge) < 0:
        raise ValueError('Invalid initialization parameters')
    data = load_bound_field(cad_path, mapping_path, fit_path)
    geometry = StepGeometry(cad_path)
    if geometry.version != data['cad_kernel_version']:
        raise ValueError('CAD importer version changed; regenerate MeshMap correspondence')
    solid = Topology(geometry.shape)
    SolidConvertor(solid).convert()
    add_geom_to_solid(solid)
    nodes, delta = data['vertices_original'], data['displacement_original']
    prints = mapped_footprints(solid, geometry, nodes, data['node_face_pairs'], data['tolerance_mm'])
    print('CAD initialization: %d faces, %d node/face memberships' % (len(solid.descendent), len(prints)), flush=True)
    result = deform_solid(solid, nodes, delta, continuity_samples=continuity_samples,
                          smoothing_weight=smoothing_weight, ridge=ridge, footprints=prints)
    gaps = interface_gaps(solid, result.control_point_delta, sample_size=100)
    errors = result.node_error(solid, delta)
    actual = np.empty((len(prints), 3))
    for face in range(len(solid.descendent)):
        indices = [k for k, fp in enumerate(prints) if fp.face == face]
        if not indices:
            continue
        surf = solid.descendent[face].geom
        start, stop = surf.ctp_index
        uv = [[prints[k].u, prints[k].v] for k in indices]
        actual[indices] = surface_basis_matrix(surf, uv) @ result.control_points[start:stop]
    node_ids = np.array([fp.node for fp in prints])
    T = data['T_original_to_scan']
    output_T = T if output_mode == 'target_coord' else np.eye(4)
    target = data['vertices_fitted_scan'][node_ids] if output_mode == 'target_coord' else (nodes + delta)[node_ids]
    actual = actual @ output_T[:3, :3].T + output_T[:3, 3]
    target_errors = np.linalg.norm(actual-target, axis=1)
    source_edge_tolerance = max(float(BRep_Tool.Tolerance(e)) for e in geometry.shapes['edge'])
    sewing_tolerance = max(DEFAULT_SEW_TOL, source_edge_tolerance)
    shape = solid_to_occ_shape(solid, result.control_points, tol=sewing_tolerance)
    if output_mode == 'target_coord':
        transform = gp_Trsf()
        transform.SetValues(*map(float, T[:3].ravel()))
        shape = BRepBuilderAPI_Transform(shape, transform, True).Shape()
    status = brep_status(shape)
    report = dict(schema_version='cadmap-initialization/1', output_mode=output_mode,
                  length_unit='mm', cad_sha256=sha256(cad_path),
                  modal_mapping_sha256=sha256(mapping_path), bound_fit_sha256=sha256(fit_path),
                  solver=result.info, brep=asdict(status),
                  source_max_edge_tolerance_mm=source_edge_tolerance,
                  sewing_tolerance_mm=sewing_tolerance,
                  displacement_residual_rms_mm=float(np.sqrt(np.mean(errors**2))),
                  target_coordinate_residual_rms_mm=float(np.sqrt(np.mean(target_errors**2))),
                  target_coordinate_residual_max_mm=float(target_errors.max()),
                  shared_edge_displacement_gap_max_mm=float(gaps.max()) if len(gaps) else 0.,
                  note='Initialization from a finite NURBS basis, not a direct point-cloud optimization.')
    for path in (step_path, archive_path, report_path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    valid = status.topologically_valid and status.boolean_valid and status.n_shells > 0 and status.n_closed_shells == status.n_shells
    report['status'] = 'complete' if valid else 'invalid_brep'
    atomic_json(report_path, report)
    if not valid:
        raise ValueError('Initialized CAD is not a valid closed B-Rep; inspect %s' % report_path)
    write_step(shape, str(step_path))
    np.savez_compressed(archive_path, schema_version='cadmap-initialization/1',
                        output_mode=output_mode, length_unit='mm',
                        cad_sha256=sha256(cad_path), cad_kernel_version=geometry.version,
                        modal_mesh_sha256=data['modal_mesh_sha256'],
                        control_points_original_frame=result.control_points,
                        control_point_displacement_original=result.control_point_delta,
                        control_points_output=result.control_points @ output_T[:3, :3].T + output_T[:3, 3],
                        T_original_to_output=output_T,
                        node_face_pairs=np.array([[fp.node, fp.face] for fp in prints]),
                        node_face_uv=np.array([[fp.u, fp.v] for fp in prints]),
                        paired_target_error_mm=target_errors,
                        modal_mapping_sha256=sha256(mapping_path), bound_fit_sha256=sha256(fit_path))
    return report
