"""Loads the deployed 5-model ensemble and runs it."""
import numpy as np

from core.paths import BUNDLE


def load_models(cfg):
    import torch
    from core.model import FusionNet

    device = torch.device("cpu")
    torch.set_num_threads(2)

    checkpoint = torch.load(
        BUNDLE / "model.pt",
        map_location=device,
        weights_only=True
    )

    models = {}

    ensembles = [
        ("full", "states", cfg["full"]),
        ("no_gpm", "states_no_gpm", cfg.get("full_no_gpm"))
    ]

    for key, state_key, arch in ensembles:
        if arch is None or state_key not in checkpoint:
            continue

        models[key] = []

        for state_dict in checkpoint[state_key]:
            model = FusionNet(
                modalities=arch["modalities"],
                dims=cfg["dims"],
                n_regions=len(cfg["region_names"]),
                ctx_dim=2,
                fusion=arch["fusion"],
                use_region=arch["use_region"]
            )

            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()

            models[key].append(model)

    return models


def predict(models, inputs, cfg, n_gates=None):
    import torch

    batch = {
        key: torch.from_numpy(
            np.ascontiguousarray(value)
        ).to("cpu")
        for key, value in inputs.items()
    }

    rainfall_predictions = []
    probabilities = []
    gates = []

    with torch.inference_mode():
        for model in models:
            reg, logit, gate = model(batch)

            rainfall_predictions.append(reg.cpu().numpy())
            probabilities.append(
                torch.sigmoid(logit).cpu().numpy()
            )

            if gate is not None:
                gates.append(gate.cpu().numpy())
            else:
                gates.append(
                    np.full(
                        (
                            len(reg),
                            n_gates or len(cfg["full"]["modalities"])
                        ),
                        np.nan
                    )
                )

    members = (
        np.array(rainfall_predictions) * cfg["y_std"]
    ).clip(min=0)

    return (
        members.mean(0),
        np.mean(probabilities, 0),
        np.mean(gates, 0),
        members
    )