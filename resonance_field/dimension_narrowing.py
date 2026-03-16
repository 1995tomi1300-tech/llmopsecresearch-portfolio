from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class NarrowingSpec:
    logical_nodes: int
    effective_nodes: int
    compression_ratio: float


def narrow_nodes(
    logical_nodes: int,
    *,
    exponent: float = 0.5,
    scale: float = 32.0,
    min_effective_nodes: int = 3000,
    max_effective_nodes: int = 17000,
) -> NarrowingSpec:
    if logical_nodes <= 0:
        raise ValueError("logical_nodes must be positive")
    if exponent <= 0.0:
        raise ValueError("exponent must be positive")
    if scale <= 0.0:
        raise ValueError("scale must be positive")

    projected = int(math.ceil(scale * (float(logical_nodes) ** exponent)))
    effective = max(min_effective_nodes, min(max_effective_nodes, projected))
    ratio = float(logical_nodes) / float(effective)
    return NarrowingSpec(
        logical_nodes=int(logical_nodes),
        effective_nodes=int(effective),
        compression_ratio=ratio,
    )


def build_narrowed_node_list(
    logical_nodes_list: list[int],
    *,
    exponent: float = 0.5,
    scale: float = 32.0,
    min_effective_nodes: int = 3000,
    max_effective_nodes: int = 17000,
) -> tuple[list[int], list[NarrowingSpec]]:
    specs: list[NarrowingSpec] = []
    for nodes in logical_nodes_list:
        specs.append(
            narrow_nodes(
                nodes,
                exponent=exponent,
                scale=scale,
                min_effective_nodes=min_effective_nodes,
                max_effective_nodes=max_effective_nodes,
            )
        )

    unique_effective_nodes = sorted({item.effective_nodes for item in specs})
    return unique_effective_nodes, specs
