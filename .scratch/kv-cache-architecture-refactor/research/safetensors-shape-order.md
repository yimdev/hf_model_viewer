# Safetensors shape order and Linear weight layout

## Conclusion

Safetensors **does not define a Linear-weight semantic convention** such as
`[out_features, in_features]`. It stores a tensor's existing ordered shape and
its packed data. The format does explicitly choose little-endian element bytes
and C/row-major order, but those are serialization rules, not meanings for the
axes.

The `[out_features, in_features]` shapes seen in most Hugging Face PyTorch
checkpoints come from the PyTorch `nn.Linear`/`F.linear` contract. The
safetensors PyTorch writer preserves that shape; it does not transpose it.

## Evidence

### 1. What the safetensors format specifies

The official format description says each tensor header contains a `dtype`, an
ordered `shape` array, and `data_offsets`; the remaining file is the byte
buffer. It also specifies little-endian elements and C/row-major order, and
does not serialize strides. There is no field that labels dimensions as
`input`, `output`, or any other model-level role.

Sources: [official safetensors format specification](https://github.com/huggingface/safetensors/blob/main/README.md#format),
[format notes on order and stride](https://github.com/huggingface/safetensors/blob/main/README.md#notes).

Consequently, for a two-dimensional shape `[A, B]`, row-major order means the
last dimension `B` is contiguous in the serialized buffer. It does **not** mean
that `A` must be `out_features` or that `B` must be `in_features`.

### 2. What the safetensors PyTorch writer actually does

The official PyTorch binding requires input tensors to be contiguous and
dense. In its flattening path it constructs a tensor specification with
`shape=v.shape`; it does not swap dimensions. `save_model` may call
`.contiguous()` to pack tensors, but making a tensor contiguous preserves its
logical shape.

Sources: [safetensors PyTorch writer: contiguity check and `shape=v.shape`](https://github.com/huggingface/safetensors/blob/main/bindings/python/py_src/safetensors/torch.py#L2353-L2386),
[safetensors `save_model`: optional `.contiguous()` packing](https://github.com/huggingface/safetensors/blob/main/bindings/python/py_src/safetensors/torch.py#L1679-L1727).

Thus, if the caller saves a contiguous tensor with shape `[6144, 12288]`, that
is the shape safetensors records. If the caller saves `[12288, 6144]`, it
records that instead.

### 3. Where `[out_features, in_features]` comes from

PyTorch officially defines `nn.Linear.weight` as having shape
`(out_features, in_features)` and describes the affine transform as
`y = x A^T + b`. The implementation allocates the parameter with
`torch.empty((out_features, in_features))` and passes it unchanged to
`F.linear`.

Sources: [PyTorch `nn.Linear` documentation](https://docs.pytorch.org/docs/main/generated/torch.nn.Linear.html),
[PyTorch `Linear` implementation](https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/linear.py#L1053-L1123),
[PyTorch `F.linear` shape contract](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.linear.html).

Hugging Face's standard PyTorch serialization path saves a model's PyTorch
state dictionary as safetensors. Therefore, PyTorch module parameter shapes
normally flow into the safetensors header unchanged. This explains the broad
consistency across vendor checkpoints published for the Hugging Face PyTorch
ecosystem; it is an ecosystem/framework convention, not a safetensors Linear
rule.

Source: [Hugging Face Hub `save_torch_state_dict` documentation](https://huggingface.co/docs/huggingface_hub/en/package_reference/serialization#huggingface_hub.save_torch_state_dict).

## Reading the example

For the displayed MLP tensors:

- `gate_proj.weight: [12288, 6144]` means a PyTorch Linear projection
  `6144 -> 12288`.
- `up_proj.weight: [12288, 6144]` likewise means `6144 -> 12288`.
- `down_proj.weight: [6144, 12288]` means `12288 -> 6144`.

Those are the tensors' original PyTorch shapes. Calling them "transposed
shapes" is only relative to the alternative convention of storing a matrix as
`[in_features, out_features]`; no serialization-time transpose has occurred.

## Limits of the observation

It is not safe to infer that every `.safetensors` file or every parameter named
`weight` must follow `[out, in]`. Safetensors can store arbitrary tensors, and
custom, fused, tensor-parallel, or quantized checkpoints may use different
logical layouts. The semantics must come from the producing model/framework or
its loading code; safetensors itself supplies dtype, shape, offsets, and packed
data only.
