"""Walking the TEMPO Online classification tree.

The flat matrix listing carries no classification, which is why
``toc(full_description=True)`` has to ask about every matrix in turn.  The
service also exposes its navigation tree, in two shapes::

    GET context/         an array of branch nodes, each wrapped as
                         {parentCode, level, context: {name, code, childrenUrl,
                         comment, url}}.  One request returns every branch:
                         currently 8 domains, 70 sub-domains and 262 surveys.

    GET context/{code}   a single node, as
                         {context: {...}, ancestors: [...], children: [...]},
                         where children are bare nodes.

A node's ``childrenUrl`` names the endpoint serving its children: ``context``
for a branch, ``matrix`` for a statistical matrix.  Matrices do not appear in
the root listing, so enumerating them costs one request per deepest branch.

Both shapes were confirmed against the live service, but TEMPO Online
publishes no API contract, so the parsing below accepts the plausible field
spellings and reports an unexpected shape instead of guessing.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ._api import (
    base_url,
    first_field,
    parse_json,
    strip_html,
    url,
    validate_flag,
    validate_language,
)
from ._cache import cache_key, cached
from ._http import request, request_many
from .errors import TempoAPIError, TempoHTTPError

__all__ = ["hierarchy"]

_ISSUES = "https://github.com/bogdanoancea/PyTEMPO/issues"

_COLUMNS = ["code", "name", "parent_code", "level", "is_matrix", "path"]


def _unexpected() -> None:
    raise TempoAPIError(
        "The TEMPO Online service returned an unexpected context-tree format. "
        "Use toc(full_description=True) instead, and please report this at "
        f"{_ISSUES}"
    )


def _context_node(node: Any) -> dict[str, str | None] | None:
    if not isinstance(node, dict):
        return None

    code = first_field(node, ("code", "id", "nodeId"))
    if code is None:
        return None

    return {
        "code": code,
        "name": first_field(node, ("name", "label", "context")),
        "children_url": first_field(node, ("childrenUrl", "url")),
    }


def _context_root(parsed: Any) -> pd.DataFrame | None:
    """Parse the root listing: wrapped nodes carrying their parent and level."""
    if not isinstance(parsed, list) or not parsed:
        return None

    entries: list[dict[str, Any]] = []
    for entry in parsed:
        if not isinstance(entry, dict):
            continue

        # Tolerate an unwrapped node as well as the {parentCode, level,
        # context} wrapper the service currently sends.
        wrapped = entry.get("context")
        node = _context_node(wrapped if wrapped is not None else entry)
        if node is None:
            continue

        try:
            level: int | None = int(entry.get("level"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            level = None

        entries.append(
            {
                **node,
                "parent_code": first_field(entry, ("parentCode", "parent")),
                "level": level,
            }
        )

    if not entries:
        return None

    parents = [
        # The service reports the root's parent as "0", which is not a node.
        None if entry["parent_code"] in {"0", "", None} else entry["parent_code"]
        for entry in entries
    ]

    return pd.DataFrame(
        {
            "code": [entry["code"] for entry in entries],
            "name": strip_html([entry["name"] for entry in entries]),
            "parent_code": parents,
            # Levels are 0-based at the service; 1-based is friendlier to report.
            "level": pd.array(
                [
                    None if entry["level"] is None else entry["level"] + 1
                    for entry in entries
                ],
                dtype="Int64",
            ),
            "children_url": [entry["children_url"] for entry in entries],
        }
    )


def _context_children(parsed: Any) -> pd.DataFrame | None:
    """Parse a single node's children, which arrive unwrapped."""
    if not isinstance(parsed, dict):
        return None

    children = parsed.get("children")
    if not isinstance(children, list) or not children:
        return None

    nodes = [_context_node(child) for child in children]
    nodes = [node for node in nodes if node is not None]
    if not nodes:
        return None

    return pd.DataFrame(
        {
            "code": [node["code"] for node in nodes],
            "name": strip_html([node["name"] for node in nodes]),
            "children_url": [node["children_url"] for node in nodes],
        }
    )


def _context_paths(nodes: pd.DataFrame) -> list[str | None]:
    """Build each node's full path from its parent links, level by level."""
    paths: list[str | None] = [None] * len(nodes)

    position_of: dict[str, int] = {}
    for position, code in enumerate(nodes["code"].tolist()):
        position_of.setdefault(code, position)

    codes = nodes["parent_code"].tolist()
    names = nodes["name"].tolist()
    levels = nodes["level"].tolist()

    for level in sorted({value for value in levels if not pd.isna(value)}):
        for position, value in enumerate(levels):
            if pd.isna(value) or value != level:
                continue

            parent = codes[position]
            parent_position = position_of.get(parent) if parent is not None else None
            parent_path = (
                paths[parent_position] if parent_position is not None else None
            )
            name = names[position]

            paths[position] = name if parent_path is None else f"{parent_path} > {name}"

    return paths


def _is_matrix(children_url: pd.Series) -> pd.Series:
    return children_url.notna() & (children_url == "matrix")


def hierarchy(
    language: str = "ro",
    matrices: bool = True,
    cache: bool = True,
) -> pd.DataFrame:
    """Retrieve the TEMPO Online classification tree.

    Returns the service's own navigation tree: the statistical domains,
    sub-domains and surveys, and optionally every matrix beneath them, each
    with its parent and its full classification path.

    This is the cheap way to learn where a matrix sits in the classification.
    The branch structure arrives in a single request.
    ``toc(full_description=True)`` recovers the same three levels by requesting
    the metadata of roughly 1500 matrices, one at a time, because it also
    reports each matrix's publication date.

    Args:
        language: ``"ro"`` for Romanian (the default) or ``"en"`` for English.
        matrices: Whether to also list the matrices under each survey.  The
            service does not include them in the tree listing, so this costs
            one additional request per deepest branch, currently 262 of them;
            they are issued concurrently and cached.  Pass ``False`` for the
            branch structure alone, which is one request.
        cache: Whether to reuse and populate the response cache described in
            :func:`tempo_ins.cache_dir`.

    Returns:
        A data frame with one row per node and the columns ``code`` (for a
        matrix, the code accepted by :func:`tempo_ins.get` and
        :func:`tempo_ins.bulk`), ``name``, ``parent_code`` (``None`` at the
        root), ``level`` (1 for a statistical domain), ``is_matrix`` and
        ``path`` (the full classification path, levels separated by ``" > "``).

    Stability:
        TEMPO Online publishes no API contract.  The two response shapes this
        function reads have been confirmed against the live service, and both
        the field names and the wrapper are matched tolerantly, but a change at
        the service could still break the walk.  It then raises a
        :class:`~tempo_ins.errors.TempoAPIError` rather than returning
        something half-built, and ``toc(full_description=True)`` remains
        available as the longer-established route to the same three
        classification levels.

    Example:
        >>> import tempo_ins as tempo
        >>> branches = tempo.hierarchy(language="en", matrices=False)
        ...                                                   # doctest: +SKIP
        >>> tree = tempo.hierarchy(language="en")             # doctest: +SKIP
        >>> tree.loc[tree["code"] == "POP105A", ["level", "path"]]
        ...                                                   # doctest: +SKIP
    """
    language = validate_language(language)
    matrices = validate_flag(matrices, "matrices")
    cache = validate_flag(cache, "cache")

    def walk() -> pd.DataFrame:
        try:
            response = request(url("context/", language=language))
        except TempoHTTPError as error:
            if error.status_code == 404:
                _unexpected()
            raise

        nodes = _context_root(parse_json(response, "context-tree"))
        if nodes is None or not len(nodes):
            _unexpected()
            raise AssertionError  # pragma: no cover - _unexpected always raises

        nodes["path"] = _context_paths(nodes)
        nodes["is_matrix"] = _is_matrix(nodes["children_url"])

        if not matrices:
            return nodes[_COLUMNS].reset_index(drop=True)

        collected = [nodes[_COLUMNS].reset_index(drop=True)]

        # Expand the branches whose children the root listing did not include.
        # Bounded, so an unexpectedly deep or cyclic tree cannot loop for ever.
        known_parents = set(nodes["parent_code"].dropna().tolist())
        frontier = nodes[
            (~nodes["is_matrix"]) & (~nodes["code"].isin(known_parents))
        ].reset_index(drop=True)

        for round_number in range(1, 9):
            if not len(frontier):
                break

            urls = [
                url(f"context/{code}", language=language)
                for code in frontier["code"].tolist()
            ]
            responses = request_many(
                urls,
                progress_label=f"Listing matrices (level {round_number + 1})",
            )

            found = []
            for index, response in enumerate(responses):
                children = _context_children(parse_json(response, "context-node"))
                if children is None:
                    continue

                parent = frontier.iloc[index]
                parent_path = parent["path"]
                names = children["name"].tolist()

                found.append(
                    pd.DataFrame(
                        {
                            "code": children["code"],
                            "name": names,
                            "parent_code": parent["code"],
                            "level": pd.array(
                                [
                                    None
                                    if pd.isna(parent["level"])
                                    else int(parent["level"]) + 1
                                ]
                                * len(children),
                                dtype="Int64",
                            ),
                            "is_matrix": _is_matrix(children["children_url"]),
                            "path": [
                                name
                                if parent_path is None
                                else f"{parent_path} > {name}"
                                for name in names
                            ],
                        }
                    )
                )

            if not found:
                break

            discovered = pd.concat(found, ignore_index=True)
            collected.append(discovered)
            frontier = discovered[~discovered["is_matrix"]].reset_index(drop=True)

        return pd.concat(collected, ignore_index=True)[_COLUMNS]

    if cache:
        return cached(
            cache_key("hierarchy", language, matrices, base_url()), walk
        ).copy()

    return walk()
