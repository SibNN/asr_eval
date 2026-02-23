from __future__ import annotations
import ast
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast


DefType = Literal[
    "class",
    "function",
    "async_function",
    "method",
    "async_method",
    "variable",
]


@dataclass
class Def:
    """An utility to build documentation for LLM agents.

    Describes a public module-level field or class field.
    """

    module: str
    """Module, such as 'asr_eval.align.alignment'."""

    type: DefType
    """Object type: 'class', 'function', 'async_function', 'method',
    'async_method', or 'variable'."""

    name: str
    """Name, such as MY_CONST, MyClass, my_func of MyClass.my_method."""

    def_path: str
    """A relative path where the object is defined. Typically the same
    as `.module`, but in form of file name
    ('asr_eval.align.matching' -> 'asr_eval/align/matching.py'), may
    differ for object that are defined in one file and exposed in
    `__all__` in another file.
    """

    lines: tuple[int, int]
    """Start line (inclusive) and end line (exclusive) in source code.
    For classes this spans all the class, including all fields. Also
    includes decorators.
    """

    definition: str
    """A set of **full code lines**, extracted from the source file,
    which span decorators and object definition. Common indent
    removed. Names in class bases and type annotations are replaced
    with their fully qualified names. Example:
    ```
    @some_decorator(x=1)
    def render_as_text(
        self,
        color_mode: typing.Literal['ansi', 'html', None] = 'ansi',
        html_add_style: bool = True,
    ) -> str:
    ```
    """

    implementation: str
    """Similar to `definition` field, but also includes implementation
    (for classes includes code for all methods).
    """


def pretty_format_module_def(d: Def, source_path: str) -> str:
    type_ = d.type.replace('_', ' ').capitalize()
    name = d.name.replace('.', '::')
    if d.def_path == source_path:
        where_defined = 'defined'
    else:
         where_defined = f'defined in {d.def_path}'
    where_defined += f' at lines {d.lines[0]}-{d.lines[1]}'

    definition = d.definition
    if d.type != 'variable':
        definition += ' ...'
    definition = f'```\n{definition}\n```'

    return (
        f'{type_} `{name}` ({where_defined})'
        + '\n'
        + definition
    )

def pretty_format_module_defs(defs: list[Def], source_path: str) -> str:
    return '\n\n'.join(
        '## ' + pretty_format_module_def(d, source_path=source_path)
        for d in defs
    )


def extract_definitions(package_root: Path, relative_path: Path) -> list[Def]:
    """An utility to build documentation for LLM agents.

    Accepts a package root, such as `Path(`/home/user/asr_eval_public`)`
    and relative path, such as `Path(`asr_eval/align/alignment.py`)`.

    Reads the `__all__` field and finds all the objects defined in it.
    Analyzes the code and return a `Def` for each object. If the object
    is defined in another file, reads that file and returns a `Def`.

    By Claude Code.
    """
    file_path = package_root / relative_path
    source = file_path.read_text()
    source_lines = source.splitlines()
    tree = ast.parse(source)

    all_names = _find_all_names(tree)
    if not all_names:
        return []

    exposing_module = _path_to_module(relative_path)
    qualify_map = _build_qualify_map(tree, exposing_module)

    local_nodes: dict[str, ast.stmt] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            local_nodes[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    local_nodes[target.id] = node
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                local_nodes[node.target.id] = node

    import_map = _build_import_map(tree, package_root)

    result: list[Def] = []
    for name in all_names:
        if name in local_nodes:
            def_obj = _make_def(
                name=name,
                exposing_module=exposing_module,
                def_rel_path=relative_path,
                node=local_nodes[name],
                source_lines=source_lines,
                qualify_map=qualify_map,
            )
            result.append(def_obj)
            node = local_nodes[name]
            if isinstance(node, ast.ClassDef):
                result.extend(_extract_class_methods(
                    name, node, exposing_module, relative_path,
                    source_lines, qualify_map,
                ))
        elif name in import_map:
            abs_path, _src_module, src_rel_path = import_map[name]
            src_source = abs_path.read_text()
            src_lines = src_source.splitlines()
            src_tree = ast.parse(src_source)
            node = _find_top_level_node(src_tree, name)
            if node is None:
                continue
            src_module = _path_to_module(src_rel_path)
            src_qualify_map = _build_qualify_map(src_tree, src_module)
            def_obj = _make_def(
                name=name,
                exposing_module=exposing_module,
                def_rel_path=src_rel_path,
                node=node,
                source_lines=src_lines,
                qualify_map=src_qualify_map,
            )
            result.append(def_obj)
            if isinstance(node, ast.ClassDef):
                result.extend(_extract_class_methods(
                    name, node, exposing_module, src_rel_path,
                    src_lines, src_qualify_map,
                ))

    return result



def _path_to_module(relative_path: Path) -> str:
    parts = list(relative_path.parts)
    if parts[-1].endswith('.py'):
        parts[-1] = parts[-1][:-3]
    if parts[-1] == '__init__':
        parts = parts[:-1]
    return '.'.join(parts)


def _module_to_file(module: str, package_root: Path) -> Path | None:
    parts = module.split('.')
    path = package_root.joinpath(*parts).with_suffix('.py')
    if path.exists():
        return path
    path = package_root.joinpath(*parts, '__init__.py')
    if path.exists():
        return path
    return None


def _find_all_names(tree: ast.Module) -> list[str] | None:
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == '__all__':
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        return [
                            elt.value
                            for elt in node.value.elts
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                        ]
    return None


def _find_top_level_node(tree: ast.Module, name: str) -> ast.stmt | None:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == name:
                return node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return node
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == name:
                return node
    return None


def _build_import_map(tree: ast.Module, package_root: Path) -> dict[str, tuple[Path, str, Path]]:
    """Returns mapping: name -> (abs_path, module_name, rel_path)"""
    result: dict[str, tuple[Path, str, Path]] = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            source_file = _module_to_file(node.module, package_root)
            if source_file is None:
                continue
            source_rel = source_file.relative_to(package_root)
            for alias in node.names:
                key = alias.asname if alias.asname else alias.name
                result[key] = (source_file, node.module, source_rel)
    return result


def _build_qualify_map(tree: ast.Module, module: str) -> dict[str, str]:
    """Build a mapping from each local name to its fully qualified name.

    Covers locally-defined names (classes, functions, variables) and all
    imported names from both package modules and the standard library.
    """
    result: dict[str, str] = {}

    # Locally-defined names get qualified with the current module
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            result[node.name] = f"{module}.{node.name}"
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    result[target.id] = f"{module}.{target.id}"
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                result[node.target.id] = f"{module}.{node.target.id}"

    # Imported names: `from X import Y [as Z]`
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            if node.level > 0:
                # Relative import → resolve to absolute module
                parts = module.split('.')
                parent_parts = parts[:-node.level] if node.level <= len(parts) else []
                abs_module = (
                    '.'.join(parent_parts + [node.module])
                    if node.module
                    else '.'.join(parent_parts)
                )
            else:
                abs_module = node.module
            for alias in node.names:
                local_name = alias.asname if alias.asname else alias.name
                result[local_name] = f"{abs_module}.{alias.name}"

    return result


def _collect_annotation_replacements(
    node: ast.stmt,
    qualify_map: dict[str, str],
    object_name: str,
) -> list[tuple[int, int, int, str]]:
    """Return (lineno, col_start, col_end, fqn) for every bare name that
    should be replaced with its fully qualified equivalent.

    Visits decorators, class bases, argument annotations, and return
    annotations. The object being defined (``object_name``) is never
    replaced.
    """
    replacements: list[tuple[int, int, int, str]] = []

    def visit(expr: ast.expr) -> None:
        if isinstance(expr, ast.Name):
            if expr.id in qualify_map and expr.id != object_name:
                replacements.append((
                    expr.lineno,
                    expr.col_offset,
                    expr.col_offset + len(expr.id),
                    qualify_map[expr.id],
                ))
        elif isinstance(expr, ast.Attribute):
            # Already a dotted reference – leave it alone
            pass
        elif isinstance(expr, ast.Subscript):
            visit(expr.value)
            visit(expr.slice)
        elif isinstance(expr, (ast.Tuple, ast.List)):
            for elt in expr.elts:
                visit(elt)
        elif isinstance(expr, ast.BinOp):
            visit(expr.left)
            visit(expr.right)
        elif isinstance(expr, ast.Call):
            visit(expr.func)
            for arg in expr.args:
                visit(arg)
            for kw in expr.keywords:
                visit(kw.value)
        elif isinstance(expr, ast.UnaryOp):
            visit(expr.operand)
        elif isinstance(expr, ast.IfExp):
            visit(expr.test)
            visit(expr.body)
            visit(expr.orelse)

    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        for dec in node.decorator_list:
            visit(dec)
        args = node.args
        for arg in args.posonlyargs + args.args + args.kwonlyargs:
            if arg.annotation:
                visit(arg.annotation)
        if args.vararg and args.vararg.annotation:
            visit(args.vararg.annotation)
        if args.kwarg and args.kwarg.annotation:
            visit(args.kwarg.annotation)
        if node.returns:
            visit(node.returns)
    elif isinstance(node, ast.ClassDef):
        for dec in node.decorator_list:
            visit(dec)
        for base in node.bases:
            visit(base)
        for keyword in node.keywords:
            visit(keyword.value)
    elif isinstance(node, ast.AnnAssign):
        if node.annotation:
            visit(node.annotation)

    return replacements


def _apply_replacements(
    source_lines: list[str],
    replacements: list[tuple[int, int, int, str]],
) -> list[str]:
    """Apply ``(lineno, col_start, col_end, replacement)`` edits to a copy
    of *source_lines*.

    Processes replacements from right to left (bottommost line, rightmost
    column first) so that earlier positions are not shifted by later edits.
    """
    sorted_reps = sorted(replacements, key=lambda r: (r[0], r[1]), reverse=True)
    lines = list(source_lines)
    for lineno, col_start, col_end, fqn in sorted_reps:
        line = lines[lineno - 1]  # AST uses 1-indexed lines
        lines[lineno - 1] = line[:col_start] + fqn + line[col_end:]
    return lines


def _extract_text(source_lines: list[str], start: int, end: int) -> str:
    """Extract source_lines[start-1:end] (1-indexed, inclusive) and remove common indent."""
    extracted = source_lines[start - 1:end]
    return textwrap.dedent('\n'.join(extracted))


def _node_start_line(node: ast.stmt) -> int:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.decorator_list:
        return node.decorator_list[0].lineno
    return node.lineno


def _make_def(
    name: str,
    exposing_module: str,
    def_rel_path: Path,
    node: ast.stmt,
    source_lines: list[str],
    qualify_map: dict[str, str] | None = None,
    obj_type: DefType | None = None,
    end_line: int | None = None,
) -> 'Def':
    start = _node_start_line(node)
    end: int = end_line if end_line is not None else cast(int, node.end_lineno)

    implementation = _extract_text(source_lines, start, end)

    # definition: decorators + signature up to the line before the body
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
        first_body = node.body[0]
        if (
            isinstance(first_body, ast.Expr)
            and isinstance(first_body.value, ast.Constant)
            and isinstance(first_body.value.value, str)
        ):
            # Include the docstring in the definition
            def_end: int = cast(int, first_body.end_lineno)
        else:
            def_end = first_body.lineno - 1
    else:
        def_end = end

    if qualify_map is not None and isinstance(
        node,
        (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.AnnAssign),
    ):
        replacements = _collect_annotation_replacements(node, qualify_map, name)
        if replacements:
            modified_lines = _apply_replacements(source_lines, replacements)
            definition = _extract_text(modified_lines, start, def_end)
        else:
            definition = _extract_text(source_lines, start, def_end)
    else:
        definition = _extract_text(source_lines, start, def_end)

    if obj_type is None:
        if isinstance(node, ast.ClassDef):
            obj_type = "class"
        elif isinstance(node, ast.AsyncFunctionDef):
            obj_type = "async_function"
        elif isinstance(node, ast.FunctionDef):
            obj_type = "function"
        else:
            obj_type = "variable"

    return Def(
        module=exposing_module,
        type=obj_type,
        name=name,
        def_path=str(def_rel_path),
        lines=(start, end + 1),
        definition=definition,
        implementation=implementation,
    )


def _extract_class_methods(
    class_name: str,
    class_node: ast.ClassDef,
    exposing_module: str,
    def_rel_path: Path,
    source_lines: list[str],
    qualify_map: dict[str, str] | None = None,
) -> list[Def]:
    """Return a Def for every public method of *class_node*.

    Public means the name does not start with an underscore.
    The Def name is formatted as ``ClassName.method_name``.
    """
    result: list[Def] = []
    for i, item in enumerate(class_node.body):
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not item.name.startswith('_'):
                method_type: DefType = (
                    "async_method" if isinstance(item, ast.AsyncFunctionDef) else "method"
                )
                result.append(_make_def(
                    name=f'{class_name}.{item.name}',
                    exposing_module=exposing_module,
                    def_rel_path=def_rel_path,
                    node=item,
                    source_lines=source_lines,
                    qualify_map=qualify_map,
                    obj_type=method_type,
                ))
        elif isinstance(item, ast.AnnAssign):
            if isinstance(item.target, ast.Name) and not item.target.id.startswith('_'):
                # Check if the next statement is a variable docstring
                end_line: int | None = None
                if i + 1 < len(class_node.body):
                    next_item = class_node.body[i + 1]
                    if (
                        isinstance(next_item, ast.Expr)
                        and isinstance(next_item.value, ast.Constant)
                        and isinstance(next_item.value.value, str)
                    ):
                        end_line = cast(int, next_item.end_lineno)
                result.append(_make_def(
                    name=f'{class_name}.{item.target.id}',
                    exposing_module=exposing_module,
                    def_rel_path=def_rel_path,
                    node=item,
                    source_lines=source_lines,
                    qualify_map=qualify_map,
                    obj_type='variable',
                    end_line=end_line,
                ))
    return result