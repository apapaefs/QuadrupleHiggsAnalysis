#!/usr/bin/env python3
"""Prepare a Sherpa-specific copy of the ``heft_c3d4`` UFO model.

The HEFT parameters ``GH`` and ``Gphi`` already contain two powers of the
strong coupling.  The source UFO deliberately does not count those powers in
its coupling-order metadata.  Sherpa uses that metadata when applying process
order constraints and when deciding which powers of alpha_s run.  This helper
therefore adds two QCD powers to every coupling whose value contains ``GH`` or
``Gphi``.

The source model is validated before an output directory is created.  The
adapted model is always written as ``heft_c3d4_sherpa`` and includes a JSON
provenance record with content hashes and the complete order transformation.
"""

import argparse
import ast
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Set, Tuple


OUTPUT_MODEL_NAME = "heft_c3d4_sherpa"
PROVENANCE_FILENAME = "sherpa_ufo_provenance.json"
QCD_ORDER_INCREMENT = 2
EFFECTIVE_PARAMETERS = frozenset(("GH", "Gphi"))

# This is intentionally an exact snapshot of the source model metadata.  A
# changed or regenerated UFO must be reviewed instead of being patched
# silently using assumptions made for the current heft_c3d4 model.
EXPECTED_ORIGINAL_ORDERS = {
    "GC_13": {"HIG": 1},
    "GC_14": {"HIG": 1, "QCD": 1},
    "GC_15": {"HIG": 1, "QCD": 2},
    "GC_GGHH": {"HIG": 1},
    "GC_GGGHH": {"HIG": 1, "QCD": 1},
    "GC_GGGGHH": {"HIG": 1, "QCD": 2},
    "GC_GGHHH": {"HIG": 1},
    "GC_GGGHHH": {"HIG": 1, "QCD": 1},
    "GC_GGGGHHH": {"HIG": 1, "QCD": 2},
    "GC_16": {"HIG": 1},
    "GC_17": {"HIG": 1, "QCD": 1},
}

REQUIRED_UFO_FILES = (
    "__init__.py",
    "couplings.py",
    "coupling_orders.py",
    "object_library.py",
    "parameters.py",
    "particles.py",
    "vertices.py",
)


class AdapterError(RuntimeError):
    """Raised when the source model or requested output is unsafe."""


class EffectiveCoupling(NamedTuple):
    name: str
    value: str
    parameters: Tuple[str, ...]
    order: Dict[str, int]
    order_start: int
    order_end: int


def sha256_file(path: Path) -> str:
    """Return the hexadecimal SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_transient(path: Path) -> bool:
    return "__pycache__" in path.parts or path.suffix in (".pyc", ".pyo")


def model_files(root: Path, exclude_provenance: bool = True) -> Iterable[Path]:
    """Yield reproducible UFO content, excluding Python cache artifacts."""

    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file() or _is_transient(path.relative_to(root)):
            continue
        if exclude_provenance and path.name == PROVENANCE_FILENAME:
            continue
        yield path


def sha256_tree(root: Path, exclude_provenance: bool = True) -> str:
    """Hash file names and contents for a model tree deterministically."""

    digest = hashlib.sha256()
    for path in model_files(root, exclude_provenance=exclude_provenance):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        contents = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
    return digest.hexdigest()


def _absolute_offsets(text: str) -> List[int]:
    offsets = [0]
    for line in text.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def _call_name(call: ast.Call) -> Optional[str]:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _keyword(call: ast.Call, name: str) -> ast.AST:
    matches = [keyword.value for keyword in call.keywords if keyword.arg == name]
    if len(matches) != 1:
        raise AdapterError("Coupling call must have exactly one {!r} keyword".format(name))
    return matches[0]


def _string_literal(node: ast.AST, description: str) -> str:
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError) as error:
        raise AdapterError("{} must be a string literal".format(description)) from error
    if not isinstance(value, str):
        raise AdapterError("{} must be a string literal".format(description))
    return value


def _order_literal(node: ast.AST, coupling_name: str) -> Dict[str, int]:
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError) as error:
        raise AdapterError("{} has a non-literal order dictionary".format(coupling_name)) from error
    if not isinstance(value, dict):
        raise AdapterError("{} order metadata is not a dictionary".format(coupling_name))
    if any(not isinstance(key, str) or not isinstance(power, int) for key, power in value.items()):
        raise AdapterError("{} order metadata must map strings to integers".format(coupling_name))
    return dict(value)


def _parameter_names(value: str, coupling_name: str) -> Set[str]:
    try:
        expression = ast.parse(value, mode="eval")
    except SyntaxError as error:
        raise AdapterError("{} has an invalid value expression".format(coupling_name)) from error
    return {node.id for node in ast.walk(expression) if isinstance(node, ast.Name)}


def discover_effective_couplings(text: str) -> Dict[str, EffectiveCoupling]:
    """Find all Coupling assignments whose value uses ``GH`` or ``Gphi``."""

    try:
        module = ast.parse(text)
    except SyntaxError as error:
        raise AdapterError("couplings.py is not valid Python") from error
    offsets = _absolute_offsets(text)
    found: Dict[str, EffectiveCoupling] = {}

    for statement in module.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if not isinstance(target, ast.Name) or not isinstance(statement.value, ast.Call):
            continue
        call = statement.value
        if _call_name(call) != "Coupling":
            continue

        value = _string_literal(_keyword(call, "value"), "{}.value".format(target.id))
        parameters = _parameter_names(value, target.id) & EFFECTIVE_PARAMETERS
        if not parameters:
            continue

        declared_name = _string_literal(_keyword(call, "name"), "{}.name".format(target.id))
        if declared_name != target.id:
            raise AdapterError(
                "Coupling assignment {!r} declares name {!r}".format(target.id, declared_name)
            )
        order_node = _keyword(call, "order")
        if not hasattr(order_node, "end_lineno") or order_node.end_lineno is None:
            raise AdapterError("Python AST does not expose source ranges for order metadata")
        start = offsets[order_node.lineno - 1] + order_node.col_offset
        end = offsets[order_node.end_lineno - 1] + order_node.end_col_offset
        found[target.id] = EffectiveCoupling(
            name=target.id,
            value=value,
            parameters=tuple(sorted(parameters)),
            order=_order_literal(order_node, target.id),
            order_start=start,
            order_end=end,
        )

    return found


def validate_original_metadata(couplings: Mapping[str, EffectiveCoupling]) -> None:
    """Require the exact reviewed effective-coupling set and source orders."""

    expected_names = set(EXPECTED_ORIGINAL_ORDERS)
    actual_names = set(couplings)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        raise AdapterError(
            "Effective-coupling set drifted (missing: {}; unexpected: {})".format(
                ", ".join(missing) or "none", ", ".join(unexpected) or "none"
            )
        )

    mismatches = []
    for name in sorted(expected_names):
        expected = EXPECTED_ORIGINAL_ORDERS[name]
        actual = couplings[name].order
        if actual != expected:
            mismatches.append("{}: expected {}, found {}".format(name, expected, actual))
    if mismatches:
        raise AdapterError("Original coupling-order metadata drifted: " + "; ".join(mismatches))


def adapted_order(order: Mapping[str, int]) -> Dict[str, int]:
    result = dict(order)
    result["QCD"] = result.get("QCD", 0) + QCD_ORDER_INCREMENT
    return result


def _format_order(order: Mapping[str, int]) -> str:
    # Preserve the UFO's semantic key order, appending QCD only when absent.
    return "{" + ",".join("{!r}:{}".format(key, value) for key, value in order.items()) + "}"


def adapt_couplings_text(text: str) -> Tuple[str, List[Dict[str, object]]]:
    """Validate and adapt coupling orders, returning text and provenance rows."""

    couplings = discover_effective_couplings(text)
    validate_original_metadata(couplings)
    replacements = []
    transformations: List[Dict[str, object]] = []
    for name in sorted(couplings):
        coupling = couplings[name]
        new_order = adapted_order(coupling.order)
        replacements.append((coupling.order_start, coupling.order_end, _format_order(new_order)))
        transformations.append(
            {
                "name": name,
                "effective_parameters": list(coupling.parameters),
                "original_order": coupling.order,
                "adapted_order": new_order,
            }
        )

    adapted = text
    for start, end, replacement in sorted(replacements, reverse=True):
        adapted = adapted[:start] + replacement + adapted[end:]

    discovered_adapted = discover_effective_couplings(adapted)
    for name, original in couplings.items():
        expected = adapted_order(original.order)
        actual = discovered_adapted[name].order
        if actual != expected:
            raise AdapterError(
                "Internal verification failed for {}: expected {}, found {}".format(
                    name, expected, actual
                )
            )
    return adapted, transformations


def _validate_source(source: Path) -> None:
    if not source.is_dir():
        raise AdapterError("Source UFO is not a directory: {}".format(source))
    missing = [name for name in REQUIRED_UFO_FILES if not (source / name).is_file()]
    if missing:
        raise AdapterError("Source UFO is missing required files: {}".format(", ".join(missing)))


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_output_path(source: Path, output: Path) -> None:
    if output.name != OUTPUT_MODEL_NAME:
        raise AdapterError("Output directory must be named {!r}".format(OUTPUT_MODEL_NAME))
    source_resolved = source.resolve()
    output_resolved = output.resolve(strict=False)
    if output_resolved == source_resolved or _is_within(output_resolved, source_resolved):
        raise AdapterError("Output must not be the source UFO or a directory inside it")
    if output.is_symlink():
        raise AdapterError("Refusing to use a symbolic link as the output directory")
    if output.exists() and not output.is_dir():
        raise AdapterError("Output exists and is not a directory: {}".format(output))


def _output_is_nonempty(output: Path) -> bool:
    return output.exists() and next(output.iterdir(), None) is not None


def _copy_ufo(source: Path, destination: Path) -> None:
    shutil.copytree(
        str(source),
        str(destination),
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )


def _provenance_source_path(source: Path, source_root: Optional[Path]) -> str:
    """Return an absolute or explicitly rooted source path for provenance."""

    source_resolved = source.resolve()
    if source_root is None:
        return str(source_resolved)

    root_resolved = Path(source_root).resolve()
    try:
        relative = source_resolved.relative_to(root_resolved)
    except ValueError as error:
        raise AdapterError(
            "Source UFO {} is not inside provenance source root {}".format(
                source_resolved, root_resolved
            )
        ) from error
    return relative.as_posix()


def adapt_ufo(
    source: Path,
    output: Path,
    force: bool = False,
    source_root: Optional[Path] = None,
) -> Path:
    """Create and return a validated Sherpa-specific UFO copy."""

    source = Path(source)
    output = Path(output)
    _validate_source(source)
    _validate_output_path(source, output)
    provenance_source_path = _provenance_source_path(source, source_root)

    # Validate all assumptions before creating, deleting, or replacing output.
    source_couplings_text = (source / "couplings.py").read_text(encoding="utf-8")
    adapted_text, transformations = adapt_couplings_text(source_couplings_text)
    source_tree_hash = sha256_tree(source)
    source_couplings_hash = sha256_file(source / "couplings.py")

    if _output_is_nonempty(output) and not force:
        raise AdapterError(
            "Output directory is not empty: {} (pass --force to replace it)".format(output)
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".{}-".format(OUTPUT_MODEL_NAME), dir=str(output.parent)) as tmp:
        staged = Path(tmp) / OUTPUT_MODEL_NAME
        _copy_ufo(source, staged)
        (staged / "couplings.py").write_text(adapted_text, encoding="utf-8")

        provenance = {
            "schema_version": 1,
            "adapter": Path(__file__).name,
            "adapter_sha256": sha256_file(Path(__file__)),
            "output_model_name": OUTPUT_MODEL_NAME,
            "source": {
                "path": provenance_source_path,
                "tree_sha256": source_tree_hash,
                "couplings_py_sha256": source_couplings_hash,
            },
            "adapted": {
                "tree_sha256_excluding_provenance": sha256_tree(staged),
                "couplings_py_sha256": sha256_file(staged / "couplings.py"),
            },
            "transformation": {
                "effective_parameters": sorted(EFFECTIVE_PARAMETERS),
                "qcd_order_increment": QCD_ORDER_INCREMENT,
                "couplings": transformations,
            },
        }
        (staged / PROVENANCE_FILENAME).write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        if output.exists():
            if _output_is_nonempty(output):
                if not force:
                    raise AdapterError(
                        "Output directory became non-empty while preparing the model: {}"
                        .format(output)
                    )
                shutil.rmtree(str(output))
            else:
                output.rmdir()
        os.replace(str(staged), str(output))

    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_ufo", type=Path, help="path to the original heft_c3d4 UFO")
    parser.add_argument(
        "output_ufo",
        type=Path,
        help="fresh output directory; its basename must be heft_c3d4_sherpa",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace a non-empty output directory after source validation",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        help=(
            "record the source UFO path relative to this directory; use the "
            "repository root when regenerating a tracked adapted model"
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = adapt_ufo(
            args.source_ufo,
            args.output_ufo,
            force=args.force,
            source_root=args.source_root,
        )
    except AdapterError as error:
        raise SystemExit("error: {}".format(error))
    provenance = output / PROVENANCE_FILENAME
    print("Prepared Sherpa UFO: {}".format(output))
    print("Provenance: {}".format(provenance))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
