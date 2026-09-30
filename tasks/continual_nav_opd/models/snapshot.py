"""Complete inference snapshots; stage-level optimizer/RNG checkpoints belong to 06."""
from pathlib import Path
import torch
from ..config import parse_config, resolved_dict, config_hash
from .policy import StandalonePolicy, DualPolicy, frozen_copy


def save_snapshot(policy: StandalonePolicy | DualPolicy, path: str | Path) -> None:
    if type(policy) not in (StandalonePolicy, DualPolicy):
        raise TypeError("only complete v3 policies can be exported")
    config = policy.config
    artifact = {"artifact_type": "v3_complete_inference", "method": config.method,
                "schema_version": 3, "sequence_protocol": config.sequence_protocol,
                "config": resolved_dict(config), "config_hash": config_hash(config),
                "policy_type": "dual" if isinstance(policy, DualPolicy) else "standalone",
                # Clone each tensor: the artifact includes both encoders and the P-time old KB.
                "state_dict": {name: value.detach().cpu().clone() for name, value in policy.state_dict().items()}}
    with Path(path).open("xb") as stream:
        torch.save(artifact, stream)


def load_snapshot(path: str | Path, device: str | torch.device = "cpu") -> StandalonePolicy | DualPolicy:
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if (artifact.get("artifact_type"), artifact.get("method"), artifact.get("schema_version"), artifact.get("sequence_protocol")) != (
            "v3_complete_inference", "ctm_pnc_opd", 3, "rollout_state_v1"):
        raise ValueError("incompatible inference artifact identity")
    config = parse_config(artifact["config"])
    if config_hash(config) != artifact["config_hash"]:
        raise ValueError("inference configuration hash mismatch")
    # Rebuild independent parameter storage; never reattach a newer live KB at load time.
    with torch.random.fork_rng(devices=[]):
        kb = StandalonePolicy(config)
        if artifact["policy_type"] == "standalone":
            policy = kb
        elif artifact["policy_type"] == "dual":
            policy = DualPolicy(config, kb)
        else:
            raise ValueError("unknown inference policy type")
        policy.load_state_dict(artifact["state_dict"], strict=True)
    return frozen_copy(policy.to(device))
