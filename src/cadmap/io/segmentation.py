"""Read MeshMap's CAD-face segmented point cloud as per-face CADMap targets.

MeshMap face IDs index the original STEP's faces in TopExp order; CADMap faces
are ``solid.descendent``. The two are joined by B-Rep identity (IsSame), the
same way the bound field is in :mod:`cadmap.io.meshmap`, never by proximity.
"""
import numpy as np

SCHEMA_VERSION = 'meshmap-segmented-pointcloud/1'


def load_segmented_pointcloud(path, cad_path=None, mapping_path=None, fit_path=None):
    """Load and check the archive; optional paths must hash to the recorded inputs."""
    from EigenFit.io.correspondence import sha256
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    if str(data['schema_version']) != SCHEMA_VERSION:
        raise ValueError('Unsupported segmented point-cloud schema')
    if str(data['length_unit']) != 'mm':
        raise ValueError('Expected millimetres')
    for given, key in [(cad_path, 'cad_sha256'), (mapping_path, 'modal_mapping_sha256'),
                       (fit_path, 'bound_fit_sha256')]:
        if given is not None and sha256(given) != str(data[key]):
            raise ValueError('%s does not match the segmented point cloud (%s)' % (given, key))
    n = len(data['points_scan'])
    face_ids = data['face_ids']
    if (data['points_original'].shape != (n, 3) or face_ids.shape != (n,)
            or not np.issubdtype(face_ids.dtype, np.integer)
            or face_ids.min() < -1 or face_ids.max() >= int(data['face_count'])):
        raise ValueError('Inconsistent segmented point-cloud arrays')
    T = data['T_original_to_scan']
    if not np.allclose(data['points_original'] @ T[:3, :3].T + T[:3, 3], data['points_scan'],
                       atol=1e-7, rtol=0):
        raise ValueError('Point frames disagree with T_original_to_scan')
    return data


def source_face_map(solid, geometry):
    """MeshMap/StepGeometry face ID -> index into ``solid.descendent``."""
    face_map = {}
    for source_id, original in enumerate(geometry.shapes['face']):
        matches = [i for i, f in enumerate(solid.descendent) if f.TopoDS.IsSame(original)]
        if len(matches) != 1:
            raise ValueError('CADMap topology cannot uniquely identify original face %d' % source_id)
        face_map[source_id] = matches[0]
    return face_map


def face_targets(data, solid, geometry, frame='original', max_distance_mm=None,
                 min_boundary_distance_mm=0.):
    """Per-face target arrays in ``solid.descendent`` order.

    ``frame='original'`` gives points in the CAD frame (the frame CADMap
    deforms in); ``'scan'`` gives the measured coordinates. Points with
    face ID -1, farther than ``max_distance_mm`` from the fitted mesh, or
    closer than ``min_boundary_distance_mm`` to a face boundary are left out.
    Returns ``(targets, point_indices)``; indices refer to the archive rows.
    """
    if frame not in ('original', 'scan'):
        raise ValueError("frame must be 'original' or 'scan'")
    if str(data['cad_kernel_version']) != geometry.version:
        raise ValueError('CAD importer version changed; regenerate the MeshMap correspondence')
    if int(data['face_count']) != len(geometry.shapes['face']):
        raise ValueError('Segmented point cloud and STEP have different face counts')
    face_map = source_face_map(solid, geometry)
    keep = data['face_ids'] >= 0
    if max_distance_mm is not None:
        keep &= data['distance_mm'] <= max_distance_mm
    if min_boundary_distance_mm > 0:
        keep &= data['boundary_distance_mm'] >= min_boundary_distance_mm
    points = data['points_original'] if frame == 'original' else data['points_scan']
    targets = [np.empty((0, 3)) for _ in solid.descendent]
    indices = [np.empty(0, dtype=np.int64) for _ in solid.descendent]
    for source_id, face in face_map.items():
        rows = np.flatnonzero(keep & (data['face_ids'] == source_id))
        targets[face], indices[face] = points[rows], rows
    return targets, indices
