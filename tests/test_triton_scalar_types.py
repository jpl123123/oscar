"""Guard Ascend-incompatible float literals without claiming NPU compilation."""

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

KERNELS = Path(__file__).resolve().parents[1] / "oscar_ascend" / "kernels"


def jit_functions(path):
    return [
        node
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(d, ast.Attribute) and d.attr == "jit"
            for d in node.decorator_list
        )
    ]


def test_jit_float_literals_do_not_require_fp64():
    # Triton's scalar inference uses the fp32 NORMAL range. In particular,
    # 1e-38 < 2**-126 is inferred as fp64, despite being an fp32 subnormal.
    for path in KERNELS.glob("*.py"):
        for fn in jit_functions(path):
            for node in ast.walk(fn):
                if isinstance(node, ast.Constant) and isinstance(node.value, float):
                    value = abs(node.value)
                    assert value == 0 or 2**-126 <= value <= (2 - 2**-23) * 2**127, (
                        f"{path.name}:{node.lineno}: literal {node.value} may infer fp64"
                    )


@pytest.mark.parametrize(
    "denominator, maximum", [(0.0, -math.inf), (1.0, -100.0), (3.25, 17.0)]
)
def test_actual_reducer_tail_preserves_empty_and_nonempty_semantics(
    denominator, maximum
):
    fn = next(
        f
        for f in jit_functions(KERNELS / "decode_kernel.py")
        if f.name == "_oscar_decode_stage2"
    )
    # Execute the actual normalization expression, replacing only tl's tensor
    # primitives. This tests arithmetic and dtype, not the Ascend compiler.
    assign = next(
        n
        for n in fn.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "safe_sum" for t in n.targets)
    )
    stores = [
        n.value
        for n in fn.body
        if isinstance(n, ast.Expr)
        and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Attribute)
        and n.value.func.attr == "store"
    ]
    term = torch.zeros(8) if denominator == 0 else torch.arange(8, dtype=torch.float32)
    namespace = {
        "tl": SimpleNamespace(where=torch.where, log=torch.log),
        "e_sum": torch.tensor(denominator, dtype=torch.float32),
        "term": term,
        "m": torch.tensor(maximum, dtype=torch.float32),
    }
    # Only execute the trusted, checked-in normalization assignment above.
    exec(  # noqa: S102
        compile(ast.Module(body=[assign], type_ignores=[]), "reducer-tail", "exec"),
        namespace,
    )
    output, lse = [
        eval(compile(ast.Expression(call.args[1]), "reducer-tail", "eval"), namespace)
        for call in stores
    ]
    assert output.dtype == lse.dtype == torch.float32
    assert torch.isfinite(output).all()
    if denominator == 0:
        assert torch.equal(output, torch.zeros_like(output)) and torch.isneginf(lse)
    else:
        torch.testing.assert_close(output, term / denominator)
        torch.testing.assert_close(lse, torch.tensor(maximum + math.log(denominator)))
