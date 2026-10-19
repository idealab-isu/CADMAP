"""Rebuild a deformed B-Rep with one vertex/edge per original entity.

The original local topology/orientations are copied explicitly. No sewing or
tolerance inflation is used to join independently constructed face boundaries.
OCC SameParameter reparameterizes pcurves against each shared spatial curve;
its measured residuals and resulting tolerances must be checked by the caller.
"""
from collections import defaultdict

import numpy as np


def _children(shape):
    from OCC.Core.TopoDS import TopoDS_Iterator
    iterator = TopoDS_Iterator(shape, False, False)
    while iterator.More():
        yield iterator.Value()
        iterator.Next()


def polish_pcurve(curve, surface, pcurve, tolerance=1e-9, max_points=8192):
    """Fit a synchronized pcurve to closest surface parameters adaptively.

    Newton corrections remove tangential parameter error, not geometric normal
    separation. The latter is measured separately and is never hidden by fitting.
    """
    from scipy.interpolate import make_interp_spline
    from OCC.Core.gp import gp_Pnt,gp_Vec
    from cadmap.geometry.nurbs import Curve2d
    from OCC.Core.Geom import Geom_BSplineCurve
    from OCC.Core.Geom2d import Geom2d_BSplineCurve
    curve=Geom_BSplineCurve.DownCast(curve);pcurve=Geom2d_BSplineCurve.DownCast(pcurve)
    def project(t, seed):
        target=np.array(curve.Value(float(t)).Coord());uv=np.array(seed,float)
        for _ in range(10):
            p,du,dv=gp_Pnt(),gp_Vec(),gp_Vec()
            surface.D1(float(uv[0]),float(uv[1]),p,du,dv)
            delta=np.linalg.lstsq(np.array([du.Coord(),dv.Coord()]).T,target-np.array(p.Coord()),rcond=1e-13)[0]
            uv+=delta
            if np.linalg.norm(delta)<2e-14:break
        value=np.array(surface.Value(float(uv[0]),float(uv[1])).Coord())
        return uv,value,float(np.linalg.norm(value-target))
    grid=list(np.linspace(0,1,33))
    for rep in (curve,pcurve):
        if hasattr(rep,'NbKnots'):grid.extend(rep.Knot(i) for i in range(1,rep.NbKnots()+1))
    # Geometric conversion produces almost coincident knots in the 2D and 3D
    # representations. Treating their roundoff differences as interpolation
    # intervals makes the quintic interpolation system ill-conditioned.
    raw=np.unique(grid)
    interior=raw[(raw>1e-9)&(raw<1-1e-9)]
    grid=[0.]
    for value in interior:
        if value-grid[-1]>1e-9:grid.append(float(value))
    grid=np.array(grid+[1.])
    values=np.array([project(t,pcurve.Value(float(t)).Coord())[0] for t in grid])
    for iteration in range(10):
        bs=make_interp_spline(grid,values,k=5)
        sample=np.unique(np.concatenate([grid[:-1]+q*np.diff(grid) for q in (.25,.5,.75)]))
        guess=bs(sample)
        projected=[project(t,uv) for t,uv in zip(sample,guess)]
        expected=np.array([p[1] for p in projected])
        actual=np.array([surface.Value(float(u),float(v)).Coord() for u,v in guess])
        errors=np.linalg.norm(expected-actual,axis=1)
        bad=errors>tolerance
        if not bad.any():
            result=Curve2d(bs.c,bs.t,'synchronized',False,degree=5)._to_Geom2d_BSplineCurve()
            old=np.array([surface.Value(*pcurve.Value(float(t)).Coord()).Coord() for t in sample])
            return result,{'interpolation_error':float(errors.max()),
                           'normal_distance_max':max(p[2] for p in projected),
                           'trim_correction_max':float(np.linalg.norm(actual-old,axis=1).max()),
                           'samples':len(sample),'poles':len(bs.c),'iterations':iteration+1}
        grid2=np.r_[grid,sample[bad]]
        values2=np.vstack([values,np.array([p[0] for p in projected])[bad]])
        order=np.argsort(grid2);grid=grid2[order];values=values2[order]
        if len(grid)>max_points:break
    raise RuntimeError(f'pcurve interpolation budget exceeded; residual {errors.max()}')


def rebuild_shared(solid, geometry, deformation, same_parameter_tolerance=1e-8, progress=None,
                   polish=True, uv_maps=None, intersection_pairs=None, intersection_local_coordinates=False):
    """Use practical OCC geometry keyed by ``face_N`` and ``edge_N``.

    Source locations must be identity; coordinate transforms are not guessed.
    Returned maps link original CADMap IDs to the actual shared output shapes.
    """
    from OCC.Core.BRep import BRep_Builder, BRep_Tool
    from OCC.Core.BRepLib import breplib
    from OCC.Core.TopoDS import (TopoDS_Vertex, TopoDS_Edge, TopoDS_Wire, TopoDS_Face,
                                TopoDS_Shell, TopoDS_Solid, TopoDS_Compound, topods)
    from OCC.Core.TopAbs import (TopAbs_FORWARD, TopAbs_REVERSED, TopAbs_VERTEX,
                                TopAbs_EDGE, TopAbs_WIRE, TopAbs_FACE, TopAbs_SHELL,
                                TopAbs_SOLID, TopAbs_COMPOUND)
    from OCC.Core.gp import gp_Pnt
    builder = BRep_Builder()
    uv_maps = uv_maps or {}
    reparameterized_pcurves = []
    face_ids, edge_ids, sources, pcurves = {}, {}, {}, defaultdict(list)
    for fi, face in enumerate(solid.descendent):
        face_ids[hash(face.TopoDS)] = f"face_{fi}"
        sources[hash(face.TopoDS)] = face.TopoDS
        for wire in face.descendent:
            for edge in wire.descendent:
                key = "edge_" + edge.geom.curv_id.rsplit("_",1)[-1]
                edge_ids[hash(edge.TopoDS)] = key
                sources[hash(edge.TopoDS)] = edge.TopoDS
                curve = edge.geom2d._to_Geom2d_BSplineCurve()
                # Undo the importer reversal, which was for rebuilding each
                # face's oriented wire. Here the original topology is retained.
                reverse = ((edge.TopoDS.Orientation() == TopAbs_REVERSED)
                           ^ (face.TopoDS.Orientation() == TopAbs_REVERSED))
                if reverse:
                    curve.Reverse()
                if f"face_{fi}" in uv_maps:
                    curve,info=uv_maps[f"face_{fi}"].transform_pcurve(curve)
                    reparameterized_pcurves.append({'edge':key,'face':f'face_{fi}',**info})
                pcurves[(hash(edge.TopoDS), hash(face.TopoDS))].append(curve)
    memo, faces, edges, vertices = {}, {}, {}, {}

    def copy(shape):
        key, kind = hash(shape), shape.ShapeType()
        if not shape.Location().IsIdentity():
            raise ValueError("shared reconstruction currently requires identity source locations")
        if key in memo:
            return memo[key].Oriented(shape.Orientation())
        children = list(_children(shape))
        if kind == TopAbs_VERTEX:
            original = topods.Vertex(shape)
            p = np.array([BRep_Tool.Pnt(original).Coord()])
            target = deformation.evaluate(p)[0]
            new = TopoDS_Vertex()
            builder.MakeVertex(new, gp_Pnt(*map(float,target)), BRep_Tool.Tolerance(original))
            vertices[key] = new
        elif kind == TopAbs_EDGE:
            original = topods.Edge(shape)
            new = TopoDS_Edge()
            builder.MakeEdge(new)
            builder.UpdateEdge(new, geometry[edge_ids[key]], BRep_Tool.Tolerance(original))
            builder.Range(new, 0., 1.)
            builder.SameParameter(new, False)
            builder.SameRange(new, True)
            edges[edge_ids[key]] = new
        elif kind == TopAbs_FACE:
            new = TopoDS_Face()
            builder.MakeFace(new, geometry[face_ids[key]], BRep_Tool.Tolerance(topods.Face(shape)))
            faces[face_ids[key]] = new
        else:
            types = {TopAbs_WIRE:(TopoDS_Wire,builder.MakeWire),
                     TopAbs_SHELL:(TopoDS_Shell,builder.MakeShell),
                     TopAbs_SOLID:(TopoDS_Solid,builder.MakeSolid),
                     TopAbs_COMPOUND:(TopoDS_Compound,builder.MakeCompound)}
            if kind not in types:
                raise ValueError(f"unsupported source topology type {kind}")
            cls, make = types[kind]
            new = cls(); make(new)
        memo[key] = new
        for child in children:
            mapped = copy(child)
            builder.Add(new, mapped)
            if kind == TopAbs_EDGE:
                # A forward vertex is the start of a forward native edge.
                parameter = 0. if child.Orientation() == TopAbs_FORWARD else 1.
                builder.UpdateVertex(topods.Vertex(mapped), parameter, topods.Edge(new),
                                     BRep_Tool.Tolerance(topods.Vertex(mapped)))
        new.Closed(shape.Closed())
        return new.Oriented(shape.Orientation())

    result = copy(solid.TopoDS)
    for (edge_key, face_key), curves in pcurves.items():
        edge, face = topods.Edge(memo[edge_key]), topods.Face(memo[face_key])
        tol = BRep_Tool.Tolerance(topods.Edge(sources[edge_key]))
        if len(curves) == 1:
            builder.UpdateEdge(edge, curves[0], face, tol)
        elif len(curves) == 2:
            builder.UpdateEdge(edge, curves[0], curves[1], face, tol)
        else:
            raise ValueError("more than two pcurves on the same face/edge")
        builder.Range(edge, face, 0., 1.)
    before = {key: BRep_Tool.Tolerance(edge) for key,edge in edges.items()}
    for i,(key, edge) in enumerate(edges.items()):
        breplib.SameParameter(edge, same_parameter_tolerance)
        if progress and i % 40 == 0:
            progress(f"  SameParameter: {i}/{len(edges)}")
    corrections=[]
    if polish:
        for i,((edge_key,face_key),curves) in enumerate(pcurves.items()):
            edge,face=topods.Edge(memo[edge_key]),topods.Face(memo[face_key])
            spatial,_,_=BRep_Tool.Curve(edge)
            surface=BRep_Tool.Surface(face)
            fitted=[]
            for side in range(len(curves)):
                oriented=topods.Edge(edge.Oriented(TopAbs_FORWARD if side==0 else TopAbs_REVERSED))
                pc,_,_=BRep_Tool.CurveOnSurface(oriented,face)
                try:
                    new_pc,info=polish_pcurve(spatial,surface,pc)
                except RuntimeError as exc:
                    raise RuntimeError(f'{edge_ids[edge_key]} on {face_ids[face_key]}: {exc}') from exc
                fitted.append(new_pc)
                corrections.append({'edge':edge_ids[edge_key],'face':face_ids[face_key],**info})
            if len(fitted)==1:builder.UpdateEdge(edge,fitted[0],face,BRep_Tool.Tolerance(edge))
            else:builder.UpdateEdge(edge,fitted[0],fitted[1],face,BRep_Tool.Tolerance(edge))
            builder.Range(edge,face,0.,1.)
            builder.SameParameter(edge,True)
            if progress and i%80==0:progress(f'  pcurve projection/refit: {i}/{len(pcurves)}')
    intersection_rows=[]
    if intersection_pairs:
        from cadmap.conversion.intersection_boundary import fit_intersection_boundary
        for edge_key,face_keys in intersection_pairs.items():
            edge=edges[edge_key];spatial,_,_=BRep_Tool.Curve(edge)
            pair=[faces[k] for k in face_keys]
            surfaces=[BRep_Tool.Surface(f) for f in pair]
            pcs=[BRep_Tool.CurveOnSurface(edge,f)[0] for f in pair]
            origin=spatial.Value(.5).Coord() if intersection_local_coordinates else None
            new_curve,new_pcs,info=fit_intersection_boundary(spatial,surfaces,pcs,coordinate_origin=origin)
            builder.UpdateEdge(edge,new_curve,BRep_Tool.Tolerance(edge))
            builder.Range(edge,0.,1.)
            for face,pc in zip(pair,new_pcs):
                builder.UpdateEdge(edge,pc,face,BRep_Tool.Tolerance(edge));builder.Range(edge,face,0.,1.)
            for vertex in _children(edge):
                v=topods.Vertex(vertex)
                builder.UpdateVertex(v,new_curve.Value(0.),BRep_Tool.Tolerance(v))
            builder.SameParameter(edge,True);builder.SameRange(edge,True)
            intersection_rows.append({'edge':edge_key,'faces':face_keys,**info})
            if progress:progress(f'  joint intersection boundary {edge_key}: motion {info["curve_motion_max"]:.3g} mm')
    report = {"faces": len(faces), "edges": len(edges), "vertices": len(vertices),
              "same_parameter_tolerance": same_parameter_tolerance,"pcurve_corrections":corrections,
              "uv_maps":{key:value.metadata() for key,value in uv_maps.items()},
              "reparameterized_pcurves":reparameterized_pcurves,
              "boundary_geometry_adjusted":bool(intersection_rows),"intersection_boundaries":intersection_rows,
              "edge_tolerances": {key: {"original": before[key], "rebuilt": BRep_Tool.Tolerance(edge),
                                          "same_parameter": bool(BRep_Tool.SameParameter(edge))}
                                  for key,edge in edges.items()}}
    return result, {"faces":faces,"edges":edges,"vertices":vertices}, report


def edge_surface_gaps(maps, solid, samples=101):
    """Check synchronized edge parameters against both incident surface traces."""
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.Geom import Geom_BSplineCurve
    from OCC.Core.Geom2d import Geom2d_BSplineCurve
    result, visited = [], set()
    for fi, original in enumerate(solid.descendent):
        face = maps["faces"][f"face_{fi}"]
        surf = BRep_Tool.Surface(face)
        for wire in original.descendent:
            for e in wire.descendent:
                key = "edge_" + e.geom.curv_id.rsplit("_",1)[-1]
                from OCC.Core.TopoDS import topods
                shared=topods.Edge(maps['edges'][key])
                side=int(e.TopoDS.Orientation()) if BRep_Tool.IsClosed(shared,face) else 0
                if (key,fi,side) in visited:
                    continue
                visited.add((key,fi,side))
                edge = maps["edges"][key].Oriented(e.TopoDS.Orientation())
                from OCC.Core.TopoDS import topods
                edge = topods.Edge(edge)
                curve, first, last = BRep_Tool.Curve(edge)
                pc, pfirst, plast = BRep_Tool.CurveOnSurface(edge,face)
                curve=Geom_BSplineCurve.DownCast(curve);pc=Geom2d_BSplineCurve.DownCast(pc)
                if pc is None:
                    raise RuntimeError(f"missing pcurve {key} face {fi}")
                breaks = [first,last]
                for rep in (curve,pc):
                    if hasattr(rep,"NbKnots"):
                        breaks.extend(rep.Knot(i) for i in range(1,rep.NbKnots()+1))
                breaks = np.unique(breaks)
                breaks = breaks[(breaks>=first)&(breaks<=last)]
                extra = np.concatenate([breaks[:-1]+q*np.diff(breaks) for q in (.25,.5,.75)])
                t = np.unique(np.r_[np.linspace(first,last,samples),breaks,extra])
                xyz = np.array([curve.Value(float(s)).Coord() for s in t])
                uv = np.array([pc.Value(float(s)).Coord() for s in t])
                trace = np.array([surf.Value(float(u),float(v)).Coord() for u,v in uv])
                distance = np.linalg.norm(trace-xyz,axis=1)
                result.append({"edge":key,"face":fi,"side":side,"maximum":float(distance.max()),
                               "rms":float(np.sqrt(np.mean(distance**2))), "samples":len(t),
                               "worst_parameter":float(t[distance.argmax()]),
                               "edge_range":[first,last],"pcurve_range":[pfirst,plast]})
    return result


def shared_trace_gaps(maps,solid,samples=257):
    """Compare both incident traces at the same spatial-edge parameter.

    Seam occurrences on one face are kept separately. Every 2D and 3D knot
    interval is probed in addition to the uniform grid; this remains a sampled
    bound, not an analytic maximum certificate.
    """
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.TopoDS import topods
    from OCC.Core.Geom import Geom_BSplineCurve
    from OCC.Core.Geom2d import Geom2d_BSplineCurve
    links=defaultdict(list);visited=set()
    for fi,original in enumerate(solid.descendent):
        face=maps['faces'][f'face_{fi}']
        for wire in original.descendent:
            for e in wire.descendent:
                key='edge_'+e.geom.curv_id.rsplit('_',1)[-1]
                edge=topods.Edge(maps['edges'][key].Oriented(e.TopoDS.Orientation()))
                side=int(e.TopoDS.Orientation()) if BRep_Tool.IsClosed(edge,face) else 0
                if (key,fi,side) in visited:continue
                visited.add((key,fi,side))
                pc,_,_=BRep_Tool.CurveOnSurface(edge,face)
                links[key].append((fi,side,BRep_Tool.Surface(face),Geom2d_BSplineCurve.DownCast(pc)))
    rows=[]
    for key,occurrences in links.items():
        if len(occurrences)!=2:raise ValueError(f'{key} has {len(occurrences)} traces instead of two')
        curve,first,last=BRep_Tool.Curve(maps['edges'][key]);curve=Geom_BSplineCurve.DownCast(curve)
        breaks=np.unique(np.concatenate([[first,last]]+[[r.Knot(i) for i in range(1,r.NbKnots()+1)]
                         for r in [curve]+[o[3] for o in occurrences]]))
        breaks=breaks[(breaks>=first)&(breaks<=last)]
        grid=np.unique(np.concatenate([np.linspace(first,last,samples),breaks]+
                              [breaks[:-1]+q*np.diff(breaks) for q in (.25,.5,.75)]))
        traces=[]
        for fi,side,surface,pc in occurrences:
            uv=[pc.Value(float(t)).Coord() for t in grid]
            traces.append(np.array([surface.Value(*p).Coord() for p in uv]))
        distance=np.linalg.norm(traces[0]-traces[1],axis=1)
        rows.append({'edge':key,'faces':[o[0] for o in occurrences],
                     'maximum':float(distance.max()),'rms':float(np.sqrt(np.mean(distance**2))),
                     'samples':len(grid),'worst_parameter':float(grid[distance.argmax()])})
    return rows
