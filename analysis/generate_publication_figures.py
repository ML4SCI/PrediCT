import sys, json, datetime, re
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

from config import PINNConfig
from geometry import CTASegmentationLoader, FlowDomainGenerator, flow_domain_to_coronary_geometry
from network import HemodynamicsPINN
from sampling import AdaptiveCoronarySampler
import ess as ESSCalculator

CKPT = Path('/Users/karan/Desktop/PrediCT/pinn_checkpoints/best_model.pt')
LOG  = Path('/Users/karan/Desktop/PrediCT/pinn_checkpoints/training.log')
OUT  = Path('/Users/karan/Desktop/PrediCT/output_v2/exports')
FDIR = OUT / 'figures'
FDIR.mkdir(parents=True, exist_ok=True)

cfg = PINNConfig()
device = cfg.runtime.resolve_device()
L0_mm  = cfg.scales.char_length * 1000.0
U0     = cfg.scales.char_velocity
rho    = cfg.blood.density
P0     = rho * U0**2
tau    = cfg.blood.dynamic_viscosity * U0 / cfg.scales.char_length
mu     = cfg.blood.dynamic_viscosity
L0     = cfg.scales.char_length

model = HemodynamicsPINN.from_config(cfg.architecture).to(device)
raw   = torch.load(CKPT, map_location=device, weights_only=True)
model.load_state_dict(raw.get('model_state_dict', raw)); model.eval()
ckpt_epoch = raw.get('epoch', 0)
ckpt_best_loss = raw.get('best_loss', 0.0)

if cfg.patient_mask_path is not None:
    mask, meta = CTASegmentationLoader.load_nifti_coronary_volume(cfg.patient_mask_path)
else:
    mask, meta = CTASegmentationLoader.create_synthetic_coronary_volume()
domain_gen  = FlowDomainGenerator(meta)
flow_domain = domain_gen.generate_flow_domain(mask, num_interior_points=2000, num_wall_points=2000)
cg          = flow_domain_to_coronary_geometry(flow_domain)
smp = AdaptiveCoronarySampler(cg, device=device).build_pinn_dataset(2000, 2000, 300, 300, char_length_mm=L0_mm)

def eval_chunked(pts_nd, chunk=2048):
    u,v,w,p = [],[],[],[]
    for i in range(0,len(pts_nd),chunk):
        x = torch.tensor(pts_nd[i:i+chunk], dtype=torch.float32, device=device)
        with torch.no_grad(): uo,vo,wo,po = model(x)
        u.append(uo.cpu().numpy()); v.append(vo.cpu().numpy())
        w.append(wo.cpu().numpy()); p.append(po.cpu().numpy())
    return np.concatenate(u).flatten(), np.concatenate(v).flatten(), np.concatenate(w).flatten(), np.concatenate(p).flatten()

u_nd,v_nd,w_nd,p_nd = eval_chunked(smp.x_interior.detach().cpu().numpy())
vel  = np.sqrt(u_nd**2+v_nd**2+w_nd**2) * U0
pres = p_nd * P0

xwg = smp.x_wall.detach().requires_grad_(True)
uw,vw,ww,_ = model(xwg)
jac = ESSCalculator.compute_velocity_jacobian(uw,vw,ww,xwg)
_,ess_m = ESSCalculator.compute_endothelial_shear_stress(jac, smp.n_wall, dynamic_viscosity=tau)
ess_np = ess_m.detach().cpu().numpy().flatten()
q3 = float(np.percentile(ess_np,75)); iqr = q3-float(np.percentile(ess_np,25))
ess_c = ess_np[ess_np <= q3+3*iqr]
xw_mm = smp.x_wall.detach().cpu().numpy() * L0_mm

mask_arr = (mask > 0).astype(np.float32)
mask_slice = mask_arr[mask_arr.shape[0]//2, :, :]

from physics import SteadyNavierStokesPhysics
phy = SteadyNavierStokesPhysics(rho=rho, mu=mu, L=L0, U=U0)
xi = smp.x_interior.clone().requires_grad_(True)
ut,vt,wt,pt = model(xi)
rc,_,_,_ = phy.compute_residuals(xi,ut,vt,wt,pt)
div = rc.detach().cpu().numpy().flatten()

def pct(a,q): return float(np.percentile(a,q))

def parse_log(log_path):
    txt = log_path.read_text()
    blocks = txt.split("Epochs completed")
    best_m = {}
    for blk in blocks[1:]:
        m = {}
        mo = re.search(r"Final loss\s*:\s*([\d.e+\-]+)", blk)
        if mo: m['final_loss'] = float(mo.group(1))
        mo = re.search(r"Best loss\s*:\s*([\d.e+\-]+)", blk)
        if mo: m['best_loss'] = float(mo.group(1))
        mo = re.search(r"Training time\s*:\s*([\d.]+)s", blk)
        if mo: m['training_time_s'] = float(mo.group(1))
        mo = re.search(r"Epochs completed\s*:\s*(\d+)", "Epochs completed"+blk)
        if mo: m['epochs'] = int(mo.group(1))
        if m.get('best_loss', 999) < best_m.get('best_loss', 999): best_m = m
    return best_m

log_m = parse_log(LOG)
metrics = {
    "checkpoint": {
        "filename": CKPT.name, "path": str(CKPT), "epoch_saved": ckpt_epoch,
        "best_loss_at_save": ckpt_best_loss, "seed": cfg.runtime.seed,
        "timestamp_generated": datetime.datetime.now().isoformat(),
    },
    "training": {
        "total_epochs": log_m.get("epochs", ckpt_epoch),
        "training_time_s": log_m.get("training_time_s"),
        "best_total_loss": log_m.get("best_loss"),
        "final_total_loss": log_m.get("final_loss"),
    },
    "ess_pa": {
        "mean": round(float(ess_c.mean()), 6),
        "median": round(float(np.median(ess_c)), 6),
    },
    "velocity_ms": {
        "mean": round(float(vel.mean()), 6),
        "max": round(float(vel.max()), 6),
    }
}
OUT.joinpath('results_summary.json').write_text(json.dumps(metrics, indent=2))

print("Generating phase2_summary.png...")
FONT = {'fontfamily': 'DejaVu Sans'}
FG, BG = '#F8FAFC', '#1E293B'
PANEL_BG = '#F1F5F9'
ACCENT = ['#3B82F6', '#F97316', '#22C55E', '#EF4444']

fig = plt.figure(figsize=(16, 10), facecolor=BG)
fig.suptitle('Phase 2 — Coronary Hemodynamics PINN  ·  ESS Prediction', fontsize=16, fontweight='bold', color=FG, y=0.97, **FONT)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.28, left=0.07, right=0.96, top=0.91, bottom=0.07)

ax0 = fig.add_subplot(gs[0,0])
ax0.set_facecolor(PANEL_BG)
ax0.hist(vel*100, bins=55, color=ACCENT[0], alpha=0.9, edgecolor='white', linewidth=0.3)
ax0.set_title('A  ·  Interior Velocity Distribution', fontweight='bold', color='#1E293B', fontsize=11, loc='left', **FONT)

ax1 = fig.add_subplot(gs[0,1])
ax1.set_facecolor(PANEL_BG)
ax1.hist(ess_c, bins=50, color=ACCENT[2], alpha=0.9, edgecolor='white', linewidth=0.3)
ax1.set_title('B  ·  Endothelial Shear Stress Distribution', fontweight='bold', color='#1E293B', fontsize=11, loc='left', **FONT)

ax2 = fig.add_subplot(gs[1,0])
ax2.set_facecolor(PANEL_BG)
sc = ax2.scatter(xw_mm[:,0], xw_mm[:,2], c=ess_np, cmap='plasma', vmin=0, vmax=float(np.percentile(ess_np, 97)), s=5, alpha=0.75)
ax2.set_title('C  ·  ESS Spatial Map (XZ projection)', fontweight='bold', color='#1E293B', fontsize=11, loc='left', **FONT)
fig.colorbar(sc, ax=ax2, shrink=0.9, pad=0.02)

ax3 = fig.add_subplot(gs[1,1])
ax3.set_facecolor('#0F172A')
ax3.imshow(mask_slice.T, origin='lower', cmap='Blues', aspect='auto', alpha=0.9)
ax3.set_title('D  ·  Lumen Mask (mid-axial slice)', fontweight='bold', color=FG, fontsize=11, loc='left', **FONT)

fig.savefig(FDIR / 'phase2_summary.png', dpi=300, bbox_inches='tight', facecolor=BG)
plt.close(fig)
print("Done.")
