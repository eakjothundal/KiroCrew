"""The store-side admission pin, as a syntax-tree analyzer.

The consolidator's write gate hands each store method its per-mutation
admission (``admit``); the store asks it as the LAST step inside its own lock,
immediately before its first mutation, and again before a commit it owns. This
module reads that contract off the source of the five store modules, for every
mutation a fronted method performs -- directly, or through any helper it calls,
however deep, in this module or another of the package.

What counts as a mutation is DERIVED, never listed: a statement executed on a
connection that is not a read (a literal that does not start with a read
keyword, or a statement built by a call -- unknown is not read-only), a commit,
a call that reaches the disk (``write_text``, ``atomic_write``, ``os.replace``
...), and every call to a function whose own body does any of these,
transitively. The hand-kept list of writer names this replaces is how the FAISS
dedup's ``_delete_episodic_row`` -- an UPDATE plus a commit two calls deep --
ran ahead of ``write_episodic``'s admission for three rounds while the pin
stayed green. Directories and lock files are containers, not content: ``mkdir``
and lock-file creation are not mutations.

The rules, for every mutation a fronted method reaches:

1. An admission point DOMINATES it: the point sits earlier in the source and on
   no branch the mutation is not on (``if`` arms, loops and ``except`` bodies
   are branches; a ``with`` block, and a ``try`` whose handlers all leave the
   function, are not -- their bodies always ran when what follows them runs).
   The points are the method's own ``admit()`` calls (or a pure local closure
   that makes one), a call that hands the hook on (``..., admit=admit`` -- the
   callee asks it, and is pinned in its own right), and a call to a local
   closure that asks the hook itself ahead of its own mutations (pinned as a
   unit). A ``with`` item that mutates (``self._vector_commit(...)`` commits at
   block exit) is judged at the block's last line.
2. Inside a lock or transaction block, the dominating point must sit INSIDE the
   same block: the store asks under the lock it writes under, so nothing that
   waited for the lock can land un-admitted.
3. A helper that is a STANDALONE writer -- its body commits a transaction of
   its own -- runs behind a lock the caller does not hold, so the caller's
   admission cannot cover it: it must be handed the hook, unless it is one of
   :data:`ROW_COMPLETION_HELPERS`, which complete the row the admission just
   let through (its audit event, its facets) and are never refused apart from
   it -- a refusal after the row's commit would leave a half-row and record a
   denial the row's admission already answered.

A mutation no point dominates fails unless :data:`UNADMITTED_SITES` names it
with its reason (a refusal's own audit row, written before any admission could
be asked); the pin's tests check each named site is exactly one site, and that
each helper the analyzer trusts as read-only (:data:`PARAMETERIZED_READERS`,
whose statement is a parameter) is fed nothing but reads.

Not covered, by design: a writer reached through an object the analyzer cannot
resolve (a stdlib or third-party callee not in :data:`WRITER_ATTRS` /
:data:`OS_WRITERS`), and the ORDER of a helper's own commit relative to the
caller's open transaction -- that is the behavioural order pins' job.
"""

from __future__ import annotations

import ast
from pathlib import Path

import kiro_crew

SRC = Path(kiro_crew.__file__).parent

#: Statements a store may run ahead of its admission: reads and transaction
#: bookkeeping. Anything else executed on a connection is a write.
SQL_READS = ("SELECT", "PRAGMA", "BEGIN", "SAVEPOINT", "RELEASE", "EXPLAIN", "ROLLBACK")
#: Attributes that write when called on anything: a connection's commit, a
#: file object's or path's writers, FAISS's index writer.
WRITER_ATTRS = frozenset(
    {
        "commit",
        "executemany",
        "executescript",
        "write",
        "writelines",
        "write_text",
        "write_bytes",
        "unlink",
        "rename",
        "write_index",
    }
)
#: Module functions of the package that write, called by their bare name.
WRITER_NAMES = frozenset({"atomic_write", "_atomic_write_text", "replace_with_retry"})
#: ``os`` / ``shutil`` writers, by dotted name.
OS_WRITERS = frozenset(
    {
        "os.replace",
        "os.rename",
        "os.write",
        "os.ftruncate",
        "os.unlink",
        "os.remove",
        "shutil.move",
        "shutil.rmtree",
        "shutil.copy",
        "shutil.copy2",
        "shutil.copyfile",
    }
)
#: A ``with`` item that is a lock or a transaction: the block a mutation inside
#: it must be admitted inside. ``_vector_commit`` takes ``_db_lock`` and owns a
#: transaction it commits at block exit.
LOCK_TOKENS = (
    "_db_lock",
    "file_lock(",
    "self._lock",
    "self._locked(",
    "self.db",
    "self._vector_commit(",
)

#: Helpers whose statement is a PARAMETER: read-only by their callers' word.
#: The pin verifies the word -- their bodies commit nothing and every statement
#: handed to them is a read (``verify_parameterized_readers``).
PARAMETERIZED_READERS = frozenset(
    {
        ("vector_memory.py", "VectorMemoryStore", "_fetch_one_locked"),
        ("vector_memory.py", "VectorMemoryStore", "_fetch_all_locked"),
    }
)

#: Standalone writers (rule 3) that COMPLETE an admitted row and are not handed
#: the hook: dominated by the row's admission, never refused apart from it.
ROW_COMPLETION_HELPERS: dict[str, str] = {
    "VectorMemoryStore._log_event": (
        "the audit row of the mutation the admission just let through (or of the "
        "write a duplicate turned away, after that admission); it carries the "
        "admitted row's own text and swallows its failures -- refusing it apart "
        "from the row would record a denial the row's admission already answered"
    ),
    "VectorMemoryStore._stamp_facets": (
        "the carve axes of the row just committed (v2 only): an index projection "
        "of admitted content, contractually never raising"
    ),
    "MemoryStore._index_file": (
        "the FTS index of the file just rewritten under the same lock: a "
        "projection of admitted content"
    ),
    "VectorMemoryStore.save_faiss_index": (
        "the on-disk flush of the FAISS mirror, every _FAISS_SAVE_INTERVAL writes: "
        "it holds the vectors of admitted rows and nothing else"
    ),
}

#: Mutations a fronted method performs with NO admission point ahead of them,
#: each with its reason -- the only way past rule 1. Keyed by class, method, the
#: callee as written and its first argument as written: a site, not a line.
UNADMITTED_SITES: dict[tuple[str, str, str, str], str] = {
    (
        "VectorMemoryStore",
        "write_episodic",
        "self._log_event",
        "SemanticRejectCode.INJECTION.value",
    ): (
        "the audit of a write REFUSED for its content before the embedding: it "
        "records that nothing was written and carries a redacted snippet -- the "
        "XPIA trail main keeps for every rejected episode"
    ),
    ("VectorMemoryStore", "set_semantic", "self.log_reject_event", "code"): (
        "the audit of a write REFUSED by validation before the store is touched: "
        "it records that nothing was written"
    ),
    ("VectorMemoryStore", "_retire_stale_episodic_v1", "self.search_episodic", ""): (
        "a READ: the search's only write is the hit rows' last_accessed_at, the "
        "access bookkeeping every search performs, never scored and never content"
    ),
}

_COMPOUND = (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.ExceptHandler, ast.Match)
_FUNCS = (ast.FunctionDef, ast.AsyncFunctionDef)


def _package_path(dotted: str) -> Path | None:
    """The file of ``kiro_crew.<dotted>``, or ``None`` when it is not a module file."""
    parts = dotted.split(".")
    if parts[0] != "kiro_crew":
        return None
    rel = SRC.joinpath(*parts[1:])
    for candidate in (rel.with_suffix(".py"), rel / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def execution_calls(fn: ast.AST) -> list[ast.Call]:
    """The calls *fn* executes -- its body, minus the bodies of the closures it
    defines (a definition is not an execution point; the closure's call is)."""
    nested = {
        id(x) for n in ast.walk(fn) if isinstance(n, _FUNCS) and n is not fn for x in ast.walk(n)
    }
    return [n for n in ast.walk(fn) if isinstance(n, ast.Call) and id(n) not in nested]


def closures(fn: ast.AST) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    return {n.name: n for n in ast.walk(fn) if isinstance(n, _FUNCS) and n is not fn}


def _calls_admit(fn: ast.AST) -> bool:
    return any(
        isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id == "admit"
        for c in ast.walk(fn)
    )


def delegations(fn: ast.AST) -> list[ast.Call]:
    """Calls that hand the hook on (an ``admit=`` keyword): admitted where they land."""
    return [n for n in execution_calls(fn) if any(k.arg == "admit" for k in n.keywords)]


def is_lock(node: ast.AST) -> bool:
    return isinstance(node, ast.With) and any(
        any(tok in ast.unparse(item.context_expr) for tok in LOCK_TOKENS) for item in node.items
    )


def _leaves(handler: ast.ExceptHandler) -> bool:
    return bool(handler.body) and isinstance(handler.body[-1], (ast.Raise, ast.Return))


def _transparent(node: ast.AST) -> bool:
    """A compound whose body always ran when what follows it runs."""
    if isinstance(node, ast.With):
        return True
    if isinstance(node, ast.Try):
        return all(_leaves(h) for h in node.handlers)
    return isinstance(node, ast.If) and ast.unparse(node.test) == "admit is not None"


def _site(cls_name: str, name: str, call: ast.Call) -> tuple[str, str, str, str]:
    first = ast.unparse(call.args[0]) if call.args else ""
    return (cls_name, name, ast.unparse(call.func), first)


class StoreModule:
    """One module's syntax tree, with the package modules it imports resolvable
    by the name they are bound to here, so a helper's writes are found wherever
    the helper lives."""

    _cache: dict[Path, "StoreModule"] = {}

    def __init__(self, path: Path, source: str | None = None):
        self.path = path
        self.source = source if source is not None else path.read_text(encoding="utf-8")
        self.tree = ast.parse(self.source)
        #: name bound here -> (module file, attribute in it, or None for the module)
        self.bindings: dict[str, tuple[Path, str | None]] = {}
        for node in self.tree.body:
            if isinstance(node, ast.ImportFrom) and node.module:
                for alias in node.names:
                    bound = alias.asname or alias.name
                    file = _package_path(f"{node.module}.{alias.name}")
                    if file is not None:
                        self.bindings[bound] = (file, None)
                        continue
                    file = _package_path(node.module)
                    if file is not None:
                        self.bindings[bound] = (file, alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    file = _package_path(alias.name)
                    if file is not None:
                        self.bindings[alias.asname or alias.name] = (file, None)
        self.functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {
            n.name: n for n in self.tree.body if isinstance(n, _FUNCS)
        }
        self.classes: dict[str, ast.ClassDef] = {
            n.name: n for n in self.tree.body if isinstance(n, ast.ClassDef)
        }
        self._memo: dict[tuple[str, str], bool] = {}

    @classmethod
    def load(cls, path: Path) -> "StoreModule":
        if path not in cls._cache:
            cls._cache[path] = cls(path)
        return cls._cache[path]

    # -- lookups ---------------------------------------------------------------

    def method(self, cls_name: str, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
        cls = self.classes.get(cls_name)
        while cls is not None:
            for node in cls.body:
                if isinstance(node, _FUNCS) and node.name == name:
                    return node
            base = next((b for b in cls.bases if isinstance(b, ast.Name)), None)
            cls = self.classes.get(base.id) if base is not None else None
        return None

    def require(self, cls_name: str, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
        fn = self.method(cls_name, name)
        assert fn is not None, f"{cls_name}.{name} not found in {self.path.name}"
        return fn

    def _reader(self, cls_name: str | None, name: str) -> bool:
        return (self.path.name, cls_name, name) in PARAMETERIZED_READERS

    # -- what mutates ----------------------------------------------------------

    def direct_mutation(self, call: ast.Call) -> bool:
        """The call itself writes: a non-read statement, a commit, a disk writer."""
        func = call.func
        if ast.unparse(func) in OS_WRITERS:
            return True
        if isinstance(func, ast.Attribute):
            if func.attr == "execute":
                first = call.args[0] if call.args else None
                if isinstance(first, ast.JoinedStr):
                    first = first.values[0] if first.values else None
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    return not first.value.lstrip().upper().startswith(SQL_READS)
                return True  # a statement built by a call: unknown is not read-only
            return func.attr in WRITER_ATTRS
        return isinstance(func, ast.Name) and func.id in WRITER_NAMES

    def resolve(
        self, call: ast.Call, cls_name: str | None, scope: dict[str, ast.AST]
    ) -> tuple["StoreModule", str | None, str, ast.AST] | None:
        """The function a call reaches, when the analyzer can see it:
        ``(module, class, qualname, node)``."""
        func = call.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            if func.value.id == "self" and cls_name is not None:
                target = self.method(cls_name, func.attr)
                if target is not None:
                    return (self, cls_name, f"{cls_name}.{func.attr}", target)
                return None
            bound = self.bindings.get(func.value.id)
            if bound is not None and bound[1] is None:
                other = StoreModule.load(bound[0])
                fn = other.functions.get(func.attr)
                return (other, None, func.attr, fn) if fn is not None else None
            return None
        if isinstance(func, ast.Name):
            if func.id in scope:
                return (self, cls_name, func.id, scope[func.id])
            if func.id in self.functions:
                return (self, None, func.id, self.functions[func.id])
            bound = self.bindings.get(func.id)
            if bound is not None and bound[1] is not None:
                other = StoreModule.load(bound[0])
                fn = other.functions.get(bound[1])
                return (other, None, bound[1], fn) if fn is not None else None
        return None

    def call_mutates(
        self,
        call: ast.Call,
        cls_name: str | None,
        scope: dict[str, ast.AST],
        visiting: set[tuple[str, str]],
    ) -> bool:
        """The call writes -- itself, or through the function it reaches."""
        if self.direct_mutation(call):
            return True
        reached = self.resolve(call, cls_name, scope)
        if reached is None:
            return False
        module, r_cls, qualname, fn = reached
        if module._reader(r_cls, qualname.rsplit(".", 1)[-1]):
            return False
        return module.function_mutates(fn, r_cls, qualname, visiting)

    def function_mutates(
        self,
        fn: ast.AST,
        cls_name: str | None,
        qualname: str,
        visiting: set[tuple[str, str]],
    ) -> bool:
        key = (str(self.path), qualname)
        if key in self._memo:
            return self._memo[key]
        if key in visiting:
            return False  # a cycle: decided by the other calls on the path
        visiting.add(key)
        scope = closures(fn)
        result = any(
            self.call_mutates(call, cls_name, scope, visiting) for call in execution_calls(fn)
        )
        visiting.discard(key)
        self._memo[key] = result
        return result

    def call_commits(self, call: ast.Call, cls_name: str | None, scope: dict[str, ast.AST]) -> bool:
        """The call reaches a function whose body commits a transaction of its
        own (transitively): a STANDALONE writer, rule 3."""
        reached = self.resolve(call, cls_name, scope)
        if reached is None:
            return False
        module, r_cls, qualname, fn = reached
        return module._commits(fn, r_cls, qualname, set())

    def _commits(
        self, fn: ast.AST, cls_name: str | None, qualname: str, visiting: set[tuple[str, str]]
    ) -> bool:
        key = (str(self.path), qualname)
        if key in visiting:
            return False
        visiting.add(key)
        scope = closures(fn)
        # ``with self.db:`` is the connection's own context manager: it commits at
        # block exit, so a body that uses it owns a transaction as surely as one
        # that calls ``commit()``.
        if any(
            isinstance(w, ast.With)
            and any(ast.unparse(item.context_expr) == "self.db" for item in w.items)
            for w in ast.walk(fn)
        ):
            return True
        for call in execution_calls(fn):
            if isinstance(call.func, ast.Attribute) and call.func.attr == "commit":
                return True
            reached = self.resolve(call, cls_name, scope)
            if reached is None or reached[2] in ROW_COMPLETION_HELPERS:
                continue  # a row-completion helper's own commit is judged by rule 1
            if reached[0]._commits(reached[3], reached[1], reached[2], visiting):
                return True
        return False


def _check_unit(
    module: StoreModule,
    cls_name: str,
    fn: ast.AST,
    label: str,
    method: str,
    violations: list[str],
) -> None:
    """Rules 1-3 for one unit: a fronted method, or a closure of one that asks
    the hook itself. Appends to *violations*."""
    scope = closures(fn)
    # Local closures: pure admission closures (``_admitted``) are admission
    # points; a closure that asks the hook AND mutates is pinned as a unit and
    # its calls are admission points; any other closure is an ordinary helper.
    pure: set[str] = set()
    units: set[str] = set()
    for name, cl in scope.items():
        if not _calls_admit(cl):
            continue
        if module.function_mutates(cl, cls_name, f"{label}.{name}", set()):
            units.add(name)
            _check_unit(module, cls_name, cl, f"{label}.{name}", method, violations)
        else:
            pure.add(name)
    handed = delegations(fn)
    points = [
        c
        for c in execution_calls(fn)
        if (isinstance(c.func, ast.Name) and (c.func.id == "admit" or c.func.id in pure | units))
        or any(c is d for d in handed)
    ]
    if not points:
        violations.append(f"{label} never asks its admit hook")
        return
    # The sibling a delegation hands the hook to is pinned in its own right.
    for d in handed:
        f = d.func
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "self":
            violations.extend(store_admission_violations(module, cls_name, f.attr))
    mutations = [
        c
        for c in execution_calls(fn)
        if not any(c is p for p in points) and module.call_mutates(c, cls_name, scope, set())
    ]

    parents: dict[int, ast.AST] = {}
    for node in ast.walk(fn):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node

    def with_of(call: ast.Call) -> ast.With | None:
        """The ``with`` whose item this call is, if any."""
        cur = parents.get(id(call))
        if isinstance(cur, ast.withitem):
            outer = parents.get(id(cur))
            if isinstance(outer, ast.With):
                return outer
        return None

    def branches(node: ast.AST) -> tuple[set[tuple[int, str]], list[ast.AST]]:
        """The branch points enclosing *node* -- each compound that may skip its
        body, with the arm the node is in -- and the lock blocks enclosing it."""
        arms: set[tuple[int, str]] = set()
        locks: list[ast.AST] = []
        child, cur = node, parents.get(id(node))
        while cur is not None and cur is not fn:
            if isinstance(cur, _COMPOUND):
                if is_lock(cur) and not (
                    isinstance(cur, ast.With)
                    and any(item.context_expr is child for item in cur.items)
                ):
                    locks.append(cur)
                if not _transparent(cur):
                    field = next(
                        (
                            f
                            for f, v in ast.iter_fields(cur)
                            if (isinstance(v, list) and any(x is child for x in v)) or v is child
                        ),
                        "",
                    )
                    if field != "test":  # a condition runs on every path into its arms
                        arms.add((id(cur), field))
            child, cur = cur, parents.get(id(cur))
        return arms, locks

    admitted = [(p, *branches(p)) for p in points]
    for m in mutations:
        block = with_of(m)
        # A mutating ``with`` item (``_vector_commit``) writes at block exit: an
        # admission inside the block is ahead of it.
        judged_at = block.end_lineno if block is not None and block.end_lineno else m.lineno
        m_arms, m_locks = branches(block if block is not None else m)
        dominating = [
            (p, p_locks)
            for p, p_arms, p_locks in admitted
            if p.lineno < judged_at and p_arms <= m_arms
        ]
        if not dominating:
            if _site(cls_name, method, m) in UNADMITTED_SITES:
                continue
            violations.append(
                f"{label}: the mutation at line {m.lineno} ({ast.unparse(m.func)}) has no "
                "admission ahead of it on its path"
            )
            continue
        if m_locks and not any(
            any(lock is held for held in p_locks for lock in m_locks) for _, p_locks in dominating
        ):
            violations.append(
                f"{label}: the mutation at line {m.lineno} inside the lock block at line "
                f"{m_locks[0].lineno} is admitted only outside that lock"
            )
            continue
        if block is None and module.call_commits(m, cls_name, scope):
            reached = module.resolve(m, cls_name, scope)
            qual = reached[2] if reached is not None else ast.unparse(m.func)
            if qual not in ROW_COMPLETION_HELPERS:
                violations.append(
                    f"{label}: the mutation at line {m.lineno} ({ast.unparse(m.func)}) commits a "
                    "transaction of its own behind its own lock and is not handed the hook"
                )


def store_admission_violations(module: StoreModule, cls_name: str, name: str) -> list[str]:
    """The pin for one store method, as a list of violations (empty = clean)."""
    fn = module.require(cls_name, name)
    params = [a.arg for a in fn.args.args + fn.args.kwonlyargs]
    if "admit" not in params:
        return [f"{cls_name}.{name} takes no admit hook"]
    violations: list[str] = []
    _check_unit(module, cls_name, fn, f"{cls_name}.{name}", name, violations)
    return sorted(set(violations))


def reachable(module: StoreModule, cls_name: str, name: str) -> set[tuple[str, str]]:
    """Every function a fronted method reaches, transitively, as ``(file, qualname)``."""
    seen: set[tuple[str, str]] = set()
    stack: list[tuple[StoreModule, str | None, str, ast.AST]] = [
        (module, cls_name, f"{cls_name}.{name}", module.require(cls_name, name))
    ]
    while stack:
        mod, cls, qual, fn = stack.pop()
        if (str(mod.path), qual) in seen:
            continue
        seen.add((str(mod.path), qual))
        scope = closures(fn)
        for call in execution_calls(fn):
            hit = mod.resolve(call, cls, scope)
            if hit is not None:
                stack.append(hit)
        for cl in scope.values():
            stack.append((mod, cls, f"{qual}.{cl.name}", cl))
    return seen


def verify_parameterized_readers(module: StoreModule, fronted: list[tuple[str, str]]) -> list[str]:
    """Rule 0: each trusted reader's body commits nothing and executes only its
    statement parameter; every statement handed to it on a fronted path is a
    read -- a literal, or a name bound in the same function to a read literal.
    (Off the fronted paths the readers' callers own their statements.)"""
    problems: list[str] = []
    readers = {(c, n) for (f, c, n) in PARAMETERIZED_READERS if f == module.path.name}
    for cls_name, name in readers:
        fn = module.require(cls_name, name)
        params = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
        for call in execution_calls(fn):
            if isinstance(call.func, ast.Attribute) and call.func.attr == "execute":
                first = call.args[0] if call.args else None
                if not (isinstance(first, ast.Name) and first.id in params):
                    problems.append(
                        f"{cls_name}.{name} executes something other than its parameter"
                    )
            elif module.direct_mutation(call):
                problems.append(f"{cls_name}.{name} writes: {ast.unparse(call.func)}")
    on_path: set[str] = set()
    for cls_name, name in fronted:
        on_path |= {q for f, q in reachable(module, cls_name, name) if f == str(module.path)}
    names = {n for _, n in readers}
    for cls in module.classes.values():
        for owner in cls.body:
            if not isinstance(owner, _FUNCS) or f"{cls.name}.{owner.name}" not in on_path:
                continue
            _verify_reader_calls(module, owner, names, problems)
    return sorted(set(problems))


def _verify_reader_calls(
    module: StoreModule, owner: ast.AST, names: set[str], problems: list[str]
) -> None:
    def _heads(value: ast.AST) -> list[str] | None:
        """The statement text(s) a value denotes, when the pin can read them."""
        if isinstance(value, ast.JoinedStr) and value.values:
            value = value.values[0]
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return [value.value]
        if isinstance(value, ast.BinOp) and isinstance(value.left, ast.Constant):
            return [str(value.left.value)]
        if isinstance(value, (ast.Tuple, ast.List)):
            parts = [_heads(e) for e in value.elts]
            if all(parts):
                return [h for part in parts for h in part]  # type: ignore[union-attr]
        return None

    bound: dict[str, list[str]] = {}
    for node in ast.walk(owner):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            heads = _heads(node.value)
            if isinstance(target, ast.Name) and heads is not None:
                bound[target.id] = heads
    for node in ast.walk(owner):  # a loop variable over a bound tuple carries its heads
        if isinstance(node, (ast.For, ast.comprehension)):
            if (
                isinstance(node.target, ast.Name)
                and isinstance(node.iter, ast.Name)
                and node.iter.id in bound
            ):
                bound[node.target.id] = bound[node.iter.id]
    for call in ast.walk(owner):
        if not (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr in names
            and call.args
        ):
            continue
        first = call.args[0]
        texts = _heads(first)
        if texts is None and isinstance(first, ast.Name):
            texts = bound.get(first.id)
        if texts is None:
            problems.append(
                f"line {call.lineno}: {call.func.attr} is handed a statement the pin cannot read"
            )
            continue
        if any(not t.lstrip().upper().startswith(SQL_READS + ("WITH",)) for t in texts):
            problems.append(f"line {call.lineno}: {call.func.attr} is handed a write")
    return sorted(set(problems))


def unadmitted_sites_present(module: StoreModule, cls_name: str, name: str) -> list[tuple]:
    """The allowlisted sites that exist in this method, one entry per occurrence."""
    fn = module.require(cls_name, name)
    return [
        _site(cls_name, name, c)
        for c in execution_calls(fn)
        if _site(cls_name, name, c) in UNADMITTED_SITES
    ]
