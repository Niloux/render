#!/usr/bin/env python3
"""Copy Gaussian actor components from one checkpoint into another."""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import GSModel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy selected obj* components from a source .pth to a target .pth."
    )
    parser.add_argument("--source", type=Path, required=True, help="Actor source .pth")
    parser.add_argument("--target", type=Path, required=True, help="Target scene .pth")
    parser.add_argument("--output", type=Path, required=True, help="Merged output .pth")
    parser.add_argument(
        "--actors",
        nargs="+",
        required=True,
        help="Actor component names or zero-based indices from the source actor list.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Only list source actor components; do not write output.",
    )
    return parser.parse_args()


def resolve_actor_names(source_model: GSModel, selectors: list[str]) -> list[str]:
    actors = source_model.get_components_by_type("obj")
    actor_names = [actor.name for actor in actors]
    resolved = []

    for selector in selectors:
        if selector.isdigit():
            index = int(selector)
            if index >= len(actor_names):
                raise IndexError(
                    f"actor index {index} 超出范围，source只有{len(actor_names)}个actor"
                )
            resolved.append(actor_names[index])
            continue
        if selector not in actor_names:
            raise KeyError(f"source checkpoint中不存在actor组件: {selector}")
        resolved.append(selector)

    return resolved


def main() -> int:
    args = parse_args()
    source = GSModel.load_from_pth(args.source)
    actor_names = [actor.name for actor in source.get_components_by_type("obj")]

    print("Source actors:")
    for index, name in enumerate(actor_names):
        print(f"- [{index}] {name}")

    if args.list:
        return 0

    target = GSModel.load_from_pth(args.target)
    selected_names = resolve_actor_names(source, args.actors)
    for name in selected_names:
        component = source.get_component(name)
        if component is None:
            raise RuntimeError(f"内部错误: 已解析actor不存在: {name}")
        target.add_component(component)

    target.save_to_pth(args.output)
    print(f"Wrote {args.output} with actors: {', '.join(selected_names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
