"""Minimal CTM forward-pass smoke test for deployment verification."""

import json

import torch

from models.ctm import ContinuousThoughtMachine


def main() -> None:
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    sequence_length = 16
    iterations = 3

    model = ContinuousThoughtMachine(
        iterations=iterations,
        d_model=32,
        d_input=8,
        heads=2,
        n_synch_out=4,
        n_synch_action=4,
        synapse_depth=1,
        memory_length=4,
        deep_nlms=True,
        memory_hidden_dims=4,
        do_layernorm_nlm=False,
        backbone_type="parity_backbone",
        positional_embedding_type="custom-rotational-1d",
        out_dims=sequence_length * 2,
        prediction_reshaper=[sequence_length, 2],
        dropout=0.0,
        neuron_select_type="random-pairing",
        n_random_pairing_self=0,
    ).to(device)
    model.eval()

    inputs = (
        torch.randint(0, 2, (batch_size, sequence_length), device=device).float()
        * 2
        - 1
    )
    with torch.inference_mode():
        predictions, certainties, synchronisation = model(inputs)

    assert predictions.shape == (batch_size, sequence_length * 2, iterations)
    assert certainties.shape == (batch_size, 2, iterations)
    assert synchronisation.shape == (batch_size, 4)
    assert torch.isfinite(predictions).all()
    assert torch.isfinite(certainties).all()
    assert torch.isfinite(synchronisation).all()

    print(
        json.dumps(
            {
                "status": "passed",
                "device": str(device),
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "gpu_name": (
                    torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else None
                ),
                "input_shape": list(inputs.shape),
                "prediction_shape": list(predictions.shape),
                "certainty_shape": list(certainties.shape),
                "synchronisation_shape": list(synchronisation.shape),
                "finite_outputs": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
