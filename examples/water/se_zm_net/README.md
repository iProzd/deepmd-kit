# Input for SeZM-Net: Smooth equivariant ZBL Message-passing Network (PyTorch)

This directory stores a minimal configuration for training SeZM-Net
(`model.type: SeZM-Net`, `descriptor.type: SeZM`) on the water example dataset.

Run:

```bash
cd examples/water/se_zm_net
dp --pt train input_torch.json
```
