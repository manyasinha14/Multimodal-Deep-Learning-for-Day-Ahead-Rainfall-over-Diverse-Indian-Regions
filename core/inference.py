"""Loads the deployed 5-model ensemble and runs it (torch imported lazily)."""
import numpy as np

from core.paths import BUNDLE


def load_models(cfg):
    """{'full': [...], 'no_gpm': [...]} - the main ensemble and the fallback used when IMERG is unreachable."""
    import torch
    from core.model import FusionNet

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    ck = torch.load(BUNDLE / "model.pt", map_location="cpu", weights_only=True)
    out = {}
    for key, states, arch in [("full", "states", cfg["full"]), ("no_gpm", "states_no_gpm", cfg.get("full_no_gpm"))]:
        if arch is None or states not in ck:
            continue
        out[key] = []
        for sd in ck[states]:
            m = FusionNet(modalities=arch["modalities"], dims=cfg["dims"], n_regions=len(cfg["region_names"]),
                          ctx_dim=2, fusion=arch["fusion"], use_region=arch["use_region"])
            m.load_state_dict(sd)
            m.eval()
            out[key].append(m)
    return out


def predict(models, inputs, cfg, n_gates=None):
    """inputs: dict from features.model_inputs. Returns mean mm, mean P(rain), mean gates, per-model mm."""
    import torch

    batch = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in inputs.items()}
    regs, probs, gates = [], [], []
    with torch.no_grad():
        for m in models:
            reg, logit, g = m(batch)
            regs.append(reg.numpy())
            probs.append(torch.sigmoid(logit).numpy())
            gates.append(g.numpy() if g is not None else np.full((len(reg), n_gates or len(cfg["full"]["modalities"])), np.nan))
    members = (np.array(regs) * cfg["y_std"]).clip(min=0)
    return members.mean(0), np.mean(probs, 0), np.mean(gates, 0), members
