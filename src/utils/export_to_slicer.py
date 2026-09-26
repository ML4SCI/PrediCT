"""
export_to_slicer.py
===================
Single-command export tool for the Phase 2 PINN pipeline.

Generates from the trained checkpoint:
  - 9 NIfTI volumes (.nii.gz) preserving CT spacing/origin/direction
  - 6 publication-quality figures (300 dpi PNG)
  - results_summary.json with all metrics

Usage
-----
    cd phase2_v2
    python export_to_slicer.py [--checkpoint PATH] [--out-dir PATH] [--log PATH]
"""

import argparse, json, logging, math, random, re
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch

from config import PINNConfig
from geometry import CTASegmentationLoader, FlowDomainGenerator, flow_domain_to_coronary_geometry
from network import HemodynamicsPINN
from sampling import AdaptiveCoronarySampler
import ess as ESSCalculator
from ess import evaluate_normal_quality
import networkx as nx

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-8s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("export_to_slicer")

BLUE="#3B82F6"; ORANGE="#F97316"; GREEN="#22C55E"
RED="#EF4444"; PURPLE="#8B5CF6"; GRAY="#94A3B8"
STYLE={"figure.dpi":300,"font.size":11,"axes.titlesize":13,
       "axes.labelsize":11,"axes.spines.top":False,"axes.spines.right":False,
       "axes.grid":True,"grid.alpha":0.3}

# VoxelMetadata helpers (meta has .spacing=(dx,dy,dz), .origin=(ox,oy,oz))
def vox2phys(vox_ijk: np.ndarray, meta) -> np.ndarray:
    dir_mat = np.array(meta.direction, dtype=float)
    scaled = vox_ijk * np.array(meta.spacing, dtype=float)
    return np.dot(scaled, dir_mat.T) + np.array(meta.origin, dtype=float)

def phys2vox(pts_mm: np.ndarray, meta) -> np.ndarray:
    diff = pts_mm - np.array(meta.origin, dtype=float)
    inv_dir = np.linalg.inv(np.array(meta.direction, dtype=float))
    indices = np.dot(diff, inv_dir.T) / np.array(meta.spacing, dtype=float)
    return np.round(indices).astype(int)

# ─────────────── helpers ────────────────────────────────────────────────────

def set_seeds(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def load_model(cfg, ckpt_path, device):
    m = HemodynamicsPINN.from_config(cfg.architecture).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    m.load_state_dict(ckpt.get("model_state_dict", ckpt))
    m.eval(); logger.info(f"Loaded: {ckpt_path}"); return m

def eval_pts(model, pts_nd, device, chunk=4096):
    u,v,w,p=[],[],[],[]
    for i in range(0,len(pts_nd),chunk):
        x=torch.tensor(pts_nd[i:i+chunk],dtype=torch.float32,device=device)
        with torch.no_grad(): uo,vo,wo,po=model(x)
        u.append(uo.cpu().numpy()); v.append(vo.cpu().numpy())
        w.append(wo.cpu().numpy()); p.append(po.cpu().numpy())
    return (np.concatenate(u).flatten(), np.concatenate(v).flatten(),
            np.concatenate(w).flatten(), np.concatenate(p).flatten())

def save_fig(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig); logger.info(f"  Figure: {path.name}")

# ─────────────── Task 3: NIfTI exports ──────────────────────────────────────

def scatter_to_nifti(values, vox_coords, shape, affine, path, fill=0.0):
    vol = np.full(shape, fill, dtype=np.float32)
    iz,iy,ix = vox_coords[:,0].astype(int), vox_coords[:,1].astype(int), vox_coords[:,2].astype(int)
    mask = (iz>=0)&(iz<shape[0])&(iy>=0)&(iy<shape[1])&(ix>=0)&(ix<shape[2])
    np.add.at(vol, (iz[mask],iy[mask],ix[mask]), values[mask])
    cnt = np.zeros(shape, dtype=np.int32)
    np.add.at(cnt, (iz[mask],iy[mask],ix[mask]), 1)
    ok = cnt>0; vol[ok] /= cnt[ok]
    img = nib.Nifti1Image(vol, affine); img.header.set_xyzt_units("mm")
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(path))
    logger.info(f"  {path.name}  shape={shape}  non-zero={ok.sum()}")

def export_nifti(model, cfg, device, mask, meta, out_dir):
    logger.info("="*60 + "\nTASK 3: NIfTI exports …")
    L0_mm = cfg.scales.char_length*1000.0
    U0 = cfg.scales.char_velocity
    P0 = cfg.blood.density * U0**2
    tau_scale = cfg.blood.dynamic_viscosity * U0 / cfg.scales.char_length

    interior = np.argwhere(mask>0)           # (N,3) as (iz, iy, ix) — matches argwhere output
    grid = mask.shape
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = np.array(meta.direction, dtype=float) * np.array(meta.spacing, dtype=float)
    affine[:3, 3] = np.array(meta.origin, dtype=float)

    pts_mm = vox2phys(interior, meta)         # physical mm for each interior voxel
    pts_nd = pts_mm / L0_mm
    logger.info(f"  Evaluating on {len(pts_nd):,} interior voxels …")
    u_nd,v_nd,w_nd,p_nd = eval_pts(model, pts_nd, device)

    ndir = out_dir/"nifti"; ndir.mkdir(parents=True, exist_ok=True)
    for name, vals in [
        ("velocity_magnitude", np.sqrt(u_nd**2+v_nd**2+w_nd**2)*U0),
        ("velocity_x", u_nd*U0), ("velocity_y", v_nd*U0), ("velocity_z", w_nd*U0),
        ("pressure", p_nd*P0),
    ]:
        scatter_to_nifti(vals, interior, grid, affine, ndir/f"{name}.nii.gz")

    # Masks
    from scipy.ndimage import binary_erosion
    lumen = (mask>0).astype(np.uint8)
    wall_m = (lumen & ~binary_erosion(lumen, iterations=1).astype(bool)).astype(np.uint8)
    for name, arr in [("lumen_mask",lumen),("flow_domain",lumen),("wall_mask",wall_m)]:
        img = nib.Nifti1Image(arr, affine); img.header.set_xyzt_units("mm")
        nib.save(img, str(ndir/f"{name}.nii.gz")); logger.info(f"  {name}.nii.gz")

    # ESS volume
    fd = FlowDomainGenerator(meta).generate_flow_domain(mask, num_interior_points=500, num_wall_points=2000)
    cg = flow_domain_to_coronary_geometry(fd)
    smp = AdaptiveCoronarySampler(cg, device=device).build_pinn_dataset(
        500, 2000, 100, 100, char_length_mm=L0_mm)
    xw_g = smp.x_wall.detach().requires_grad_(True)
    uw,vw,ww,_ = model(xw_g)
    jac = ESSCalculator.compute_velocity_jacobian(uw,vw,ww,xw_g)
    _,ess_m = ESSCalculator.compute_endothelial_shear_stress(jac, smp.n_wall, dynamic_viscosity=tau_scale)
    ess_np = ess_m.detach().cpu().numpy().flatten()
    wall_mm = smp.x_wall.detach().cpu().numpy() * L0_mm
    wall_vox = phys2vox(wall_mm, meta)
    scatter_to_nifti(ess_np, wall_vox, grid, affine, ndir/"ess.nii.gz")
    logger.info("TASK 3 complete.")

# ─────────────── Task 5: JSON ───────────────────────────────────────────────

def parse_log(log_path):
    m = {}
    if log_path is None or not Path(log_path).exists(): return m
    txt = Path(log_path).read_text()
    for pat,key in [(r"Training time\s*:\s*([\d.]+)s","training_time_s"),
                    (r"Epochs completed\s*:\s*(\d+)","training_epochs"),
                    (r"Final loss\s*:\s*([\d.e+\-]+)","final_total_loss"),
                    (r"Best loss\s*:\s*([\d.e+\-]+)","best_total_loss")]:
        mo = re.search(pat, txt)
        if mo: m[key] = float(mo.group(1))
    last = [l for l in txt.splitlines() if "Epoch [" in l and "Total Loss:" in l]
    if last:
        for key,tag in [("final_mass_loss","mass"),("final_momentum_loss","momentum"),
                        ("final_wall_loss","wall"),("final_inlet_loss","inlet"),
                        ("final_outlet_loss","outlet"),("final_integral_mass_loss","integral_mass")]:
            mo = re.search(rf"{tag}:\s*([\d.e+\-]+)", last[-1])
            if mo: m[key] = float(mo.group(1))
    return m

def compute_live(model, cfg, device, mask, meta):
    from physics import SteadyNavierStokesPhysics
    L0_mm = cfg.scales.char_length*1000.0
    U0 = cfg.scales.char_velocity
    P0 = cfg.blood.density * U0**2
    tau_scale = cfg.blood.dynamic_viscosity * U0 / cfg.scales.char_length

    fd = FlowDomainGenerator(meta).generate_flow_domain(mask, 2000, 2000)
    cg = flow_domain_to_coronary_geometry(fd)
    smp = AdaptiveCoronarySampler(cg, device=device).build_pinn_dataset(
        2000, 2000, 300, 300, char_length_mm=L0_mm)

    u,v,w,p = eval_pts(model, smp.x_interior.detach().cpu().numpy(), device)
    vel = np.sqrt(u**2+v**2+w**2)*U0; p_dim = p*P0

    x_wall_grad = smp.x_wall.detach().requires_grad_(True)
    uw,vw,ww,_ = model(x_wall_grad)
    jac = ESSCalculator.compute_velocity_jacobian(uw,vw,ww,x_wall_grad)
    _,ess_m = ESSCalculator.compute_endothelial_shear_stress(jac, smp.n_wall, dynamic_viscosity=tau_scale)
    ess_np = ess_m.detach().cpu().numpy().flatten()
    q3 = float(np.percentile(ess_np,75)); iqr = q3-float(np.percentile(ess_np,25))
    ec = ess_np[ess_np <= q3+3*iqr]

    phy = SteadyNavierStokesPhysics(rho=cfg.blood.density, mu=cfg.blood.dynamic_viscosity,
                                    L=cfg.scales.char_length, U=U0)
    xi = smp.x_interior.clone().requires_grad_(True)
    ut,vt,wt,pt = model(xi)
    rc,_,_,_ = phy.compute_residuals(xi,ut,vt,wt,pt)
    div = rc.detach().cpu().numpy().flatten()

    # Inlet RMSE
    with torch.no_grad():
        u_in,v_in,w_in,_ = model(smp.x_inlet)
    ic = smp.x_inlet.mean(0); inn = smp.n_inlet.mean(0)
    R_nd = 0.0015/cfg.scales.char_length
    d = smp.x_inlet - ic; pr = torch.sum(d*inn,1,keepdim=True)
    r2 = torch.sum((d-pr*inn)**2,1,keepdim=True)
    umx = cfg.boundary.inlet_peak_velocity_scale
    uex = umx*torch.clamp(1.0-r2/R_nd**2,min=0.0)*(-inn[0])
    vex = umx*torch.clamp(1.0-r2/R_nd**2,min=0.0)*(-inn[1])
    wex = umx*torch.clamp(1.0-r2/R_nd**2,min=0.0)*(-inn[2])
    rmse = float(torch.sqrt(torch.mean((u_in-uex)**2+(v_in-vex)**2+(w_in-wex)**2)).item())

    def pct(a,q): return float(np.percentile(a,q))
    return {
        "inlet_velocity_rmse": rmse,
        "mass_conservation": {"continuity_residual_mean": float(np.mean(np.abs(div))),
                               "continuity_residual_std": float(np.std(div)),
                               "continuity_residual_l2": float(np.sqrt(np.mean(div**2)))},
        "ess_pa": {"mean":float(ec.mean()),"median":float(np.median(ec)),"std":float(ec.std()),
                   "p5":pct(ec,5),"p25":pct(ec,25),"p75":pct(ec,75),"p95":pct(ec,95),
                   "min":float(ec.min()),"max":float(ec.max()),"n_points":len(ec),
                   "pct_atherogenic_lt1Pa":float(100.0*(ec<1.0).mean())},
        "velocity_ms": {"mean":float(vel.mean()),"median":float(np.median(vel)),
                        "std":float(vel.std()),"max":float(vel.max()),"p95":pct(vel,95)},
        "pressure_pa": {"mean":float(p_dim.mean()),"std":float(p_dim.std()),
                        "min":float(p_dim.min()),"max":float(p_dim.max())},
        "divergence": {"mean_abs":float(np.mean(np.abs(div))),"max_abs":float(np.max(np.abs(div))),
                       "l2_norm":float(np.sqrt(np.mean(div**2)))},
    }

# ─────────────── Task 6: Figures ────────────────────────────────────────────

def fig_ess_histogram(ess, fdir):
    with plt.rc_context(STYLE):
        fig,ax = plt.subplots(figsize=(8,5))
        ax.hist(ess, bins=50, color=BLUE, alpha=0.85, edgecolor="white", lw=0.4)
        ax.axvline(1.0, color=RED, ls="--", lw=1.5, label="Low ESS threshold (1 Pa)")
        ax.axvline(float(np.median(ess)), color=GREEN, lw=1.5, label=f"Median={np.median(ess):.3f} Pa")
        ax.axvspan(0,1.0,alpha=0.06,color=RED); ax.set_xlim(left=0)
        ax.set_xlabel("ESS (Pa)"); ax.set_ylabel("Wall points")
        ax.set_title("ESS Distribution — Phase 2 PINN"); ax.legend()
        save_fig(fig, fdir/"fig_ess_histogram.png")

def fig_ess_spatial(x_wall_mm, ess_full, fdir):
    """3D scatter ESS spatial map (3 projections). ess_full must match x_wall_mm length."""
    ess = ess_full.copy()
    with plt.rc_context(STYLE):
        fig,axes = plt.subplots(1,3,figsize=(15,4.5))
        vm = float(np.percentile(ess,98))
        for ax,(xi,yi,xl,yl) in zip(axes,[(0,1,"X","Y"),(0,2,"X","Z"),(1,2,"Y","Z")]):
            sc=ax.scatter(x_wall_mm[:,xi],x_wall_mm[:,yi],c=ess,cmap="plasma",
                          vmin=0,vmax=vm,s=4,alpha=0.7)
            ax.set_xlabel(xl+" (mm)"); ax.set_ylabel(yl+" (mm)")
            ax.set_title(f"{xl}{yl}-projection"); ax.set_aspect("equal")
        fig.colorbar(sc, ax=axes, shrink=0.7).set_label("ESS (Pa)")
        fig.suptitle("ESS Spatial Distribution", fontsize=14, fontweight="bold")
        save_fig(fig, fdir/"fig_ess_spatial_map.png")

def fig_centreline(model, cfg, device, meta, mask, fdir):
    L0_mm = cfg.scales.char_length*1000.0; U0=cfg.scales.char_velocity
    fd = FlowDomainGenerator(meta).generate_flow_domain(mask, 100, 100)
    g = fd.centerline_graph
    terms=[n for n,d in g.degree() if d==1]
    src=max(terms, key=lambda n: g.nodes[n]["pos"][2])
    valid_paths = []
    for t in terms:
        if t != src:
            try:
                valid_paths.append(nx.shortest_path(g, src, t))
            except nx.NetworkXNoPath:
                pass
    path = max(valid_paths, key=len) if valid_paths else [src]
    pos=np.array([g.nodes[n]["pos"] for n in path])
    rad=np.array([g.nodes[n]["radius"] for n in path])
    arc=np.concatenate([[0],np.cumsum(np.linalg.norm(np.diff(pos,axis=0),axis=1))])
    nd=arc/arc[-1]
    xt=torch.tensor(pos/L0_mm,dtype=torch.float32,device=device)
    with torch.no_grad(): uc,vc,wc,_ = model(xt)
    vel=torch.sqrt(uc**2+vc**2+wc**2).cpu().numpy().flatten()*U0
    R0_nd=g.nodes[src]["radius"]/L0_mm
    theory=cfg.boundary.inlet_peak_velocity_scale*(R0_nd/(rad/L0_mm))**2*U0
    with plt.rc_context(STYLE):
        fig,(a1,a2)=plt.subplots(1,2,figsize=(12,4.5))
        a1.plot(nd,vel,color=BLUE,lw=2,label="PINN |u|")
        a1.plot(nd,theory,color=ORANGE,ls="--",lw=1.5,label="Poiseuille U_max")
        a1.set_xlabel("Normalised distance"); a1.set_ylabel("Velocity (m/s)")
        a1.set_title("Centreline Velocity"); a1.legend()
        a2.plot(nd,rad,color=PURPLE,lw=2)
        a2.set_xlabel("Normalised distance"); a2.set_ylabel("Radius (mm)")
        a2.set_title("Vessel Radius")
        fig.suptitle("Centreline Analysis",fontsize=14,fontweight="bold")
        save_fig(fig, fdir/"fig_centreline_velocity.png")

def fig_flux(model, cfg, device, flow_domain, meta, fdir):
    L0_mm=cfg.scales.char_length*1000.0; U0=cfg.scales.char_velocity
    g=flow_domain.centerline_graph
    terms=[n for n,d in g.degree() if d==1]
    src=max(terms, key=lambda n: g.nodes[n]["pos"][2])
    valid_paths = []
    for t in terms:
        if t != src:
            try:
                valid_paths.append(nx.shortest_path(g, src, t))
            except nx.NetworkXNoPath:
                pass
    path = max(valid_paths, key=len) if valid_paths else [src]
    pos=np.array([g.nodes[n]["pos"] for n in path])
    rad=np.array([g.nodes[n]["radius"] for n in path])
    arc=np.concatenate([[0],np.cumsum(np.linalg.norm(np.diff(pos,axis=0),axis=1))])
    idx=np.linspace(0,len(path)-1,15,dtype=int)
    qs,ds=[],[]
    for i in idx:
        ctr=pos[i]; R=rad[i]
        tang=(pos[i+1]-ctr) if i<len(path)-1 else (ctr-pos[i-1])
        th=tang/(np.linalg.norm(tang)+1e-12)
        arb=np.array([1.,0.,0.]) if abs(th[0])<0.9 else np.array([0.,1.,0.])
        uv=np.cross(th,arb); uv/=np.linalg.norm(uv); vv=np.cross(th,uv)
        r_s=np.sqrt(np.random.rand(500,1))*R; theta=np.random.rand(500,1)*2*math.pi
        disk=(ctr+r_s*np.cos(theta)*uv+r_s*np.sin(theta)*vv)/L0_mm
        ud,vd,wd,_=eval_pts(model,disk,device)
        Q=abs(np.mean(ud*th[0]+vd*th[1]+wd*th[2]))*U0*math.pi*R**2
        qs.append(Q); ds.append(arc[i])
    with plt.rc_context(STYLE):
        fig,ax=plt.subplots(figsize=(9,4.5))
        ax.plot(ds,qs,color=GREEN,lw=2,marker="o",ms=5,label="|Q|")
        ax.axhline(qs[0],color=GRAY,ls="--",lw=1,label=f"Inlet Q={qs[0]:.3f}")
        ax.set_xlabel("Distance from inlet (mm)"); ax.set_ylabel("|Q| (ml/s approx.)")
        ax.set_title("Volumetric Flux Along Vessel"); ax.legend()
        save_fig(fig, fdir/"fig_flux_conservation.png")

def fig_vel_hist(model, cfg, device, mask, meta, fdir):
    L0_mm=cfg.scales.char_length*1000.0; U0=cfg.scales.char_velocity
    iv=np.argwhere(mask>0)
    idx=np.random.choice(len(iv),min(20000,len(iv)),replace=False)
    pts_mm=vox2phys(iv[idx], meta)
    u,v,w,_=eval_pts(model,pts_mm/L0_mm,device)
    vel=np.sqrt(u**2+v**2+w**2)*U0
    with plt.rc_context(STYLE):
        fig,ax=plt.subplots(figsize=(8,5))
        ax.hist(vel,bins=60,color=PURPLE,alpha=0.85,edgecolor="white",lw=0.4)
        ax.axvline(vel.mean(),color=RED,ls="--",lw=1.5,label=f"Mean={vel.mean():.4f} m/s")
        ax.set_xlabel("Velocity magnitude (m/s)"); ax.set_ylabel("Count")
        ax.set_title("Global Interior Velocity Distribution"); ax.legend()
        save_fig(fig, fdir/"fig_velocity_histogram.png")

def fig_pres_hist(model, cfg, device, mask, meta, fdir):
    L0_mm=cfg.scales.char_length*1000.0; U0=cfg.scales.char_velocity
    P0=cfg.blood.density*U0**2
    iv=np.argwhere(mask>0)
    idx=np.random.choice(len(iv),min(20000,len(iv)),replace=False)
    pts_mm=vox2phys(iv[idx], meta)
    _,_,_,p=eval_pts(model,pts_mm/L0_mm,device)
    p_dim=p*P0
    with plt.rc_context(STYLE):
        fig,ax=plt.subplots(figsize=(8,5))
        ax.hist(p_dim,bins=60,color=ORANGE,alpha=0.85,edgecolor="white",lw=0.4)
        ax.axvline(p_dim.mean(),color=BLUE,ls="--",lw=1.5,label=f"Mean={p_dim.mean():.2f} Pa")
        ax.set_xlabel("Pressure (Pa)"); ax.set_ylabel("Count")
        ax.set_title("Global Interior Pressure Distribution"); ax.legend()
        save_fig(fig, fdir/"fig_pressure_histogram.png")

# ─────────────── main ───────────────────────────────────────────────────────

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoint",default="/Users/karan/Desktop/PrediCT/pinn_checkpoints/best_model.pt")
    parser.add_argument("--out-dir",default="/Users/karan/Desktop/PrediCT/output_v2/exports")
    parser.add_argument("--log",default="/Users/karan/Desktop/PrediCT/pinn_checkpoints/training.log")
    parser.add_argument("--skip-nifti",action="store_true")
    args=parser.parse_args()

    cfg=PINNConfig(); set_seeds(cfg.runtime.seed)
    device=cfg.runtime.resolve_device()
    out_dir=Path(args.out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    fdir=out_dir/"figures"

    logger.info(f"Device: {device} | Output: {out_dir}")

    if cfg.patient_mask_path is not None:
        mask, meta = CTASegmentationLoader.load_nifti_coronary_volume(cfg.patient_mask_path)
    else:
        mask, meta = CTASegmentationLoader.create_synthetic_coronary_volume()
    fd=FlowDomainGenerator(meta).generate_flow_domain(mask,4000,2000)
    cg=flow_domain_to_coronary_geometry(fd)
    model=load_model(cfg,Path(args.checkpoint),device)

    # Task 5
    logger.info("="*60+"\nTASK 5: results_summary.json …")
    results={"pipeline":"Phase 2 Coronary Hemodynamics PINN",
             "checkpoint":args.checkpoint,
             "config":{
                 "reynolds_number":round(cfg.blood.density*cfg.scales.char_velocity*cfg.scales.char_length/cfg.blood.dynamic_viscosity,2),
                 "L0_m":cfg.scales.char_length,"U0_ms":cfg.scales.char_velocity,
                 "mu_Pas":cfg.blood.dynamic_viscosity,
                 "tau_ref_Pa":round(cfg.blood.dynamic_viscosity*cfg.scales.char_velocity/cfg.scales.char_length,6),
                 "loss_weights":{k:getattr(cfg.loss_weights,k) for k in
                     ["lambda_continuity","lambda_momentum","lambda_wall_noslip",
                      "lambda_inlet","lambda_outlet","lambda_integral_mass"]}},
             "training":parse_log(args.log),
             "metrics":compute_live(model,cfg,device,mask,meta)}
    jpath=out_dir/"results_summary.json"
    jpath.write_text(json.dumps(results,indent=2))
    logger.info(f"  Saved: results_summary.json")

    # Task 3
    if not args.skip_nifti:
        export_nifti(model,cfg,device,mask,meta,out_dir)
    else:
        logger.info("Skipping NIfTI (--skip-nifti).")

    # Task 6
    logger.info("="*60+"\nTASK 6: Publication figures …")
    L0_mm=cfg.scales.char_length*1000.0
    tau_scale=cfg.blood.dynamic_viscosity*cfg.scales.char_velocity/cfg.scales.char_length
    smp=AdaptiveCoronarySampler(cg,device=device).build_pinn_dataset(
        2000,2000,300,300,char_length_mm=L0_mm)
    xw_grad=smp.x_wall.detach().requires_grad_(True)
    uw,vw,ww,_=model(xw_grad)
    jac=ESSCalculator.compute_velocity_jacobian(uw,vw,ww,xw_grad)
    _,ess_m=ESSCalculator.compute_endothelial_shear_stress(jac,smp.n_wall,dynamic_viscosity=tau_scale)
    ess_np=ess_m.detach().cpu().numpy().flatten()
    q3=float(np.percentile(ess_np,75)); iqr=q3-float(np.percentile(ess_np,25))
    ess_c=ess_np[ess_np<=q3+3*iqr]
    xw_mm=smp.x_wall.detach().cpu().numpy()*L0_mm

    fig_ess_histogram(ess_c, fdir)
    fig_ess_spatial(xw_mm, ess_np, fdir)  # use full array matching x_wall length
    fig_centreline(model,cfg,device,meta,mask,fdir)
    fig_flux(model,cfg,device,fd,meta,fdir)
    fig_vel_hist(model,cfg,device,mask,meta,fdir)
    fig_pres_hist(model,cfg,device,mask,meta,fdir)

    logger.info("="*60)
    logger.info("All exports complete.")
    logger.info(f"  NIfTI:   {out_dir/'nifti'}")
    logger.info(f"  Figures: {fdir}")
    logger.info(f"  JSON:    {jpath}")

if __name__=="__main__":
    main()

