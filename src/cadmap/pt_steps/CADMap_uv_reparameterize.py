"""Replace the measurement CAD by a geometry-identical STEP with automatic UV charts.

Run it first after copyinput, so COMSOL, the MeshMap CAD correspondence and
CADMap all read the same file. The input STEP is kept as dc:CADfilename_original
and is always the source, so re-running the step never compounds charts.
A failed check falls back to an unchanged copy of the input (or raises when
dc:uv_reparam_on_failure is "error"); the JSON report records every check.
"""
from limatix.dc_value import hrefvalue
from cadmap.conversion.uv_reparameterize_step import StepUVOptions, reparameterize_step
from cadmap.geometry.trim_uv_diagnostics import TrimUVOptions


def run(dc_dest_href, dc_measident_str, dc_CADfilename_href=None,
        dc_CADfilename_original_href=None,
        dc_uv_reparam_str='auto',
        dc_uv_reparam_on_failure_str='fallback',
        dc_uv_trigger_occupancy_float=0.2,
        dc_uv_buffer_fraction_float=0.1,
        dc_uv_restrict_bspline_str='true',
        dc_uv_geometry_tolerance_mm_float=1e-7,
        dc_uv_edge_tolerance_mm_float=1e-6):
    source = dc_CADfilename_original_href if dc_CADfilename_original_href is not None else dc_CADfilename_href
    if source is None:
        raise ValueError('uvreparam needs dc:CADfilename_original or dc:CADfilename')
    # Write dc:CADfilename_original only when it is new; rewriting an element that
    # came from the xlg makes a later mergeinput conflict.
    keep = {} if dc_CADfilename_original_href is not None else {'dc:CADfilename_original': source}
    if dc_uv_reparam_str not in ('auto', 'off'):
        raise ValueError('dc:uv_reparam must be auto or off')
    if dc_uv_reparam_str == 'off':
        return {'dc:CADfilename': source, **keep}
    stem = dc_measident_str + '_uv_reparameterized'
    step = hrefvalue(stem + '.step', contexthref=dc_dest_href)
    report = hrefvalue(stem + '.json', contexthref=dc_dest_href)
    trigger = float(dc_uv_trigger_occupancy_float)
    options = StepUVOptions(
        trim=TrimUVOptions(trigger_occupancy=trigger, satisfactory_occupancy=max(trigger, .2),
                           transverse_buffer_fraction=float(dc_uv_buffer_fraction_float)),
        restrict_bspline=str(dc_uv_restrict_bspline_str).strip().lower() in ('true', '1', 'yes'),
        geometry_tolerance=float(dc_uv_geometry_tolerance_mm_float),
        edge_tolerance=float(dc_uv_edge_tolerance_mm_float))
    result = reparameterize_step(source.getpath(), step.getpath(), report.getpath(), options,
                                 on_failure=dc_uv_reparam_on_failure_str)
    print('UV reparameterization: %s; %s; reverted faces %s; failures %s' % (
        result['status'], result['actions'], result['reverted_faces'],
        result['failures'] or result.get('readback', {}).get('failures', [])), flush=True)
    return {'dc:CADfilename': step, **keep,
            'dc:uv_reparameterization_report': report,
            'dc:uv_reparameterization_status': result['status']}
