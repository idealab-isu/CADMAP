"""Joint spatial curve and pcurve fitting at a two-surface boundary."""
import numpy as np
from scipy.interpolate import make_interp_spline


def fit_intersection_boundary(curve,surfaces,pcurves,tolerance=1e-10,max_points=8192,coordinate_origin=None):
    """Refit one closed shared boundary against both fixed surfaces together.

    A minimum-norm Newton correction follows the intersection near the existing
    trims. Geometry and parameter corrections are measured, never hidden in
    increased B-Rep tolerances. The surfaces themselves are left unchanged.
    """
    from OCC.Core.gp import gp_Pnt,gp_Vec
    from cadmap.geometry.nurbs import Curve2d
    from OCC.Core.Geom import Geom_BSplineCurve
    from OCC.Core.TColgp import TColgp_Array1OfPnt
    from OCC.Core.TColStd import TColStd_Array1OfReal,TColStd_Array1OfInteger

    if coordinate_origin is not None:
        # Common translation improves cancellation in small cavity intersections
        # far from the coordinate origin. Copies preserve the caller's surfaces.
        from OCC.Core.gp import gp_Trsf
        from OCC.Core.Geom import Geom_Curve, Geom_Surface
        origin=np.asarray(coordinate_origin,float)
        if origin.shape!=(3,) or not np.isfinite(origin).all():
            raise ValueError('finite 3-vector intersection coordinate origin required')
        shift=gp_Trsf();shift.SetTranslation(gp_Vec(*map(float,-origin)))
        local_curve=Geom_Curve.DownCast(curve.Copy());local_curve.Transform(shift)
        local_surfaces=[Geom_Surface.DownCast(s.Copy()) for s in surfaces]
        for surface in local_surfaces:surface.Transform(shift)
        spatial,pcs,info=fit_intersection_boundary(local_curve,local_surfaces,pcurves,tolerance,max_points)
        shift.SetTranslation(gp_Vec(*map(float,origin)));spatial.Transform(shift)
        knots=np.array([spatial.Knot(i) for i in range(1,spatial.NbKnots()+1)])
        sample=np.unique(np.r_[knots,np.concatenate([knots[:-1]+q*np.diff(knots) for q in (.25,.5,.75)])])
        xyz=np.array([spatial.Value(float(t)).Coord() for t in sample])
        residual=max(float(np.linalg.norm(np.array([s.Value(*pc.Value(float(t)).Coord()).Coord()
                     for t in sample])-xyz,axis=1).max()) for s,pc in zip(surfaces,pcs))
        if residual>tolerance:
            raise RuntimeError(f'world-frame intersection residual exceeds tolerance: {residual}')
        return spatial,pcs,{**info,'coordinate_origin':origin.tolist(),
                            'world_frame_trace_max':residual,'world_frame_samples':len(sample)}

    def project(t,seed):
        uv=np.array(seed,float).reshape(2,2)
        reference,tangent=gp_Pnt(),gp_Vec();curve.D1(float(t),reference,tangent)
        reference=np.array(reference.Coord());tangent=np.array(tangent.Coord())
        tangent/=np.linalg.norm(tangent)
        for _ in range(20):
            points=[];jac=[]
            for s,param in zip(surfaces,uv):
                p,du,dv=gp_Pnt(),gp_Vec(),gp_Vec()
                s.D1(float(param[0]),float(param[1]),p,du,dv)
                points.append(np.array(p.Coord()));jac.append(np.array([du.Coord(),dv.Coord()]).T)
            residual=points[1]-points[0]
            along=-np.dot(.5*(points[0]+points[1])-reference,tangent)
            if max(np.linalg.norm(residual),abs(along))<2e-13:break
            # Retain correspondence to the existing edge parameter. Without
            # this fourth condition the null direction can drift along the
            # intersection while the surface residual remains tiny.
            system=np.vstack([np.column_stack([jac[0],-jac[1]]),
                              np.r_[.5*tangent@jac[0],.5*tangent@jac[1]]])
            correction=np.linalg.lstsq(system,np.r_[residual,along],rcond=1e-13)[0]
            uv+=correction.reshape(2,2)
        points=np.array([s.Value(float(p[0]),float(p[1])).Coord() for s,p in zip(surfaces,uv)])
        if np.linalg.norm(points[0]-points[1])>tolerance:
            raise RuntimeError('joint boundary projection did not converge')
        if np.linalg.norm(points.mean(0)-reference)>1e-5:
            raise RuntimeError('joint boundary correction exceeded 1e-5 mm motion budget')
        return np.r_[points.mean(0),uv.ravel()]

    grid=np.linspace(0,1,129)
    values=np.array([project(t,[p.Value(float(t)).Coord() for p in pcurves]) for t in grid])
    if np.linalg.norm(values[-1]-values[0])>1e-7:
        raise ValueError('intersection boundary currently requires a closed trim on both surfaces')
    values[-1]=values[0]
    for iteration in range(12):
        spline=make_interp_spline(grid,values,k=5)
        sample=np.unique(np.concatenate([grid[:-1]+q*np.diff(grid) for q in (.25,.5,.75)]))
        predicted=spline(sample)
        targets=np.array([project(t,row[3:]) for t,row in zip(sample,predicted)])
        traces=[np.array([s.Value(float(row[3+2*i]),float(row[4+2*i])).Coord()
                         for row in predicted]) for i,s in enumerate(surfaces)]
        errors=np.maximum.reduce([np.linalg.norm(predicted[:,:3]-targets[:,:3],axis=1)]+
                                  [np.linalg.norm(trace-predicted[:,:3],axis=1) for trace in traces])
        bad=errors>tolerance
        if not bad.any():
            # Preserve the interpolant's C4 knot multiplicities. Expanding it
            # to artificial C0 Bezier joins makes kernel checks unnecessarily
            # expensive even though the physical curve is unchanged.
            poles=TColgp_Array1OfPnt(1,len(spline.c))
            for i,point in enumerate(spline.c[:,:3],1):poles.SetValue(i,gp_Pnt(*map(float,point)))
            unique,multiplicity=np.unique(spline.t,return_counts=True)
            knots=TColStd_Array1OfReal(1,len(unique));mults=TColStd_Array1OfInteger(1,len(unique))
            for i,(k,m) in enumerate(zip(unique,multiplicity),1):knots.SetValue(i,float(k));mults.SetValue(i,int(m))
            spatial=Geom_BSplineCurve(poles,knots,mults,5,False)
            pcs=[Curve2d(spline.c[:,3+2*i:5+2*i],spline.t,'joint_intersection',False,degree=5)
                 ._to_Geom2d_BSplineCurve() for i in range(2)]
            old=np.array([curve.Value(float(t)).Coord() for t in sample])
            old_traces=[np.array([s.Value(*p.Value(float(t)).Coord()).Coord() for t in sample])
                        for s,p in zip(surfaces,pcurves)]
            return spatial,pcs,{'samples':len(sample),'interpolation_error':float(errors.max()),
                         'curve_motion_max':float(np.linalg.norm(predicted[:,:3]-old,axis=1).max()),
                         'trim_motion_max':max(float(np.linalg.norm(a-b,axis=1).max()) for a,b in zip(traces,old_traces)),
                         'vertex_motion':float(np.linalg.norm(values[0,:3]-np.array(curve.Value(0.).Coord()))),
                         'iterations':iteration+1,'tolerance':tolerance,'poles':len(spline.c)}
        newgrid=np.r_[grid,sample[bad]];newvalues=np.vstack([values,targets[bad]])
        order=np.argsort(newgrid);grid=newgrid[order];values=newvalues[order]
        if len(grid)>max_points:break
    raise RuntimeError(f'joint boundary interpolation exceeded its budget: {errors.max()}')
