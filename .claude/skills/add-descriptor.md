# Adding a New Descriptor to deepmd-kit (PT + dpmodel)

Simplified guide for PT and dpmodel backends only.

## Step 1: Implement in dpmodel

**Create** `deepmd/dpmodel/descriptor/<name>.py`

Inherit from `NativeOP` and `BaseDescriptor`. Register with decorators:

```python
from deepmd.dpmodel import NativeOP
from .base_descriptor import BaseDescriptor

@BaseDescriptor.register("your_name")
class DescrptYourName(NativeOP, BaseDescriptor): ...
```

Key requirements:
- `__init__`: initialize cutoff, sel, networks, davg/dstd statistics
- `call(coord_ext, atype_ext, nlist, mapping=None)`: forward pass returning `(descriptor, rot_mat, g2, h2, sw)`
- `serialize() -> dict`: save with `@class`, `type`, `@version`, `@variables` keys
- `deserialize(cls, data)`: reconstruct from dict
- Property/getter methods: `get_rcut`, `get_sel`, `get_dim_out`, `mixed_types`, etc.
- `__getitem__`/`__setitem__` for `davg`/`dstd` access

All dpmodel code **must** use `array_api_compat` for cross-backend compatibility.

**Reference**: `deepmd/dpmodel/descriptor/dpa1.py`, `se_t.py`

## Step 2: Register

**Edit** `deepmd/dpmodel/descriptor/__init__.py` — add import and `__all__` entry.

**Edit** `deepmd/utils/argcheck.py` — register descriptor arguments:

```python
@descrpt_args_plugin.register("your_name", doc="Description")
def descrpt_your_name_args() -> list[Argument]:
    return [
        Argument("rcut", float, optional=True, default=6.0, doc=doc_rcut),
        # ... add all constructor parameters
    ]
```

## Step 3: Implement PT backend

**Create** `deepmd/pt/model/descriptor/<name>.py`

PT descriptors are fully reimplemented in PyTorch (not wrapping dpmodel). Inherit from `BaseDescriptor` and `torch.nn.Module`. Must implement `forward()`, `serialize()`, `deserialize()`.

**Edit** `deepmd/pt/model/descriptor/__init__.py` — add import.

**Reference**: `deepmd/pt/model/descriptor/dpa1.py`

## Step 4: Write tests

| Test | File | Purpose |
|------|------|---------|
| dpmodel | `source/tests/common/dpmodel/test_descriptor_<name>.py` | Serialize/deserialize round-trip |
| PT | `source/tests/pt/model/test_descriptor_<name>.py` | PT hard-coded tests |

## Verification

```bash
python -m pytest source/tests/common/dpmodel/test_descriptor_<name>.py -v
python -m pytest source/tests/pt/model/test_descriptor_<name>.py -v
```

## Files summary

| Step | Action | File |
|------|--------|------|
| 1 | Create | `deepmd/dpmodel/descriptor/<name>.py` |
| 2 | Edit | `deepmd/dpmodel/descriptor/__init__.py` |
| 2 | Edit | `deepmd/utils/argcheck.py` |
| 3 | Create | `deepmd/pt/model/descriptor/<name>.py` |
| 3 | Edit | `deepmd/pt/model/descriptor/__init__.py` |
| 4 | Create | `source/tests/common/dpmodel/test_descriptor_<name>.py` |
| 4 | Create | `source/tests/pt/model/test_descriptor_<name>.py` |
