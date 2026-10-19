"""Inherit the parent CAD/mapping and the specimen's bound MeshMap result."""
from limatix.dc_value import hrefvalue
from cadmap.io.meshmap import initialize_from_meshmap


def run(dc_dest_href, dc_measident_str, dc_CADfilename_href,
        dc_modal_cad_mapping_href, dc_cad_bound_fit_href,
        dc_cad_output_mode_str='target_coord',
        dc_cad_continuity_samples_int=24,
        dc_cad_smoothing_weight_float=0.001,
        dc_cad_ridge_float=0.01):
    stem = dc_measident_str + '_cad_initialized'
    step = hrefvalue(stem + '.step', contexthref=dc_dest_href)
    archive = hrefvalue(stem + '.npz', contexthref=dc_dest_href)
    report = hrefvalue(stem + '.json', contexthref=dc_dest_href)
    initialize_from_meshmap(
        dc_CADfilename_href.getpath(), dc_modal_cad_mapping_href.getpath(),
        dc_cad_bound_fit_href.getpath(), step.getpath(), archive.getpath(),
        report.getpath(), output_mode=dc_cad_output_mode_str,
        continuity_samples=dc_cad_continuity_samples_int,
        smoothing_weight=dc_cad_smoothing_weight_float, ridge=dc_cad_ridge_float)
    return {'dc:cad_initialized': step, 'dc:cad_initialization': archive,
            'dc:cad_initialization_report': report}
