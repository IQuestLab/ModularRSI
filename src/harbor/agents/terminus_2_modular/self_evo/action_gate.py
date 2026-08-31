"""C3 — the declared action has to be what was actually built.

The router decides two axes: which *lane* a finding belongs to, and which
*action* it calls for — ``modify`` this incumbent, ``replace`` it, or ``add``
something new. The implementer is told which one. Until now nothing checked
that it obeyed, so the routing was a suggestion written into a prompt:

* a ``novelty``/``add`` proposal could come back as an edit to the incumbent;
* a ``replace`` could come back as more logic piled into the 1659-line variant
  it was supposed to retire;
* an implementer could rewrite a *second* active variant on the side;

and every downstream gate would still pass. The damage is not tidiness. The
archive then records an action the tree does not implement, the niche cell the
change was supposed to open stays empty, and "does adding variants help more
than modifying them?" becomes unanswerable because half the adds were modifies.

Everything here is mechanical: the declared action against the library before
and after, plus the files touched. No model is asked anything, and each check is
written so it can only fire on unambiguous evidence — a false reject costs a
whole generation, so where the evidence is ambiguous the gate stays quiet.

**The path convention.** A variant lives at ``modules/<type>/<name>.py`` (the
same rule :func:`library.build_default_library` discovers by), reported either
package-rooted or relative to the modules root depending on the run's mode.
Anything deeper is a helper — nothing registers it, so nothing selects it — and
anything outside ``modules/`` is kernel surface, judged by its own gate.
"""

from __future__ import annotations

from harbor.agents.terminus_2_modular import archive as _archive

MODIFY = "modify"
REPLACE = "replace"
ADD = "add"


def qual_for_path(rel: str) -> str | None:
    """``[NEW:][modules/]<type>/<name>.py`` → ``"<type>/<name>"``; else None.

    None means "this file is not a variant": a helper under a subpackage, an
    ``__init__.py``, or kernel surface. Those are not evidence of anything this
    gate judges.

    The ``NEW:`` marker identifies a file absent from the parent tree and is
    removed before deriving the variant qualifier.
    """
    rel = str(rel)
    if rel.startswith("NEW:"):
        rel = rel[4:]
    parts = [p for p in rel.replace("\\", "/").split("/") if p and p != "."]
    if parts and parts[0] == "modules":
        parts = parts[1:]
    if len(parts) != 2 or not parts[1].endswith(".py"):
        return None
    stem = parts[1][:-3]
    if stem.startswith("__"):
        return None
    return f"{parts[0]}/{stem}"


def _touched_variants(files_changed) -> set[str]:
    return {q for q in (qual_for_path(f) for f in files_changed or []) if q}


def _supersede_targets(variant_meta_text: str) -> set[str]:
    """Everything the staged ``<variant_meta>`` blocks claim to retire.

    Both spellings count: editors write ``SUPERSEDES: confirm_exit`` at least as
    often as the qualified form, and refusing the bare one would reject correct
    work over punctuation.
    """
    out: set[str] = set()
    for meta in _archive.parse_variant_meta(variant_meta_text or ""):
        for ref in meta.get("supersedes") or []:
            out.add(ref)
            out.add(ref.split("/")[-1])
    return out


def check_action(
    *,
    action: str,
    target_variant: str,
    files_changed,
    variant_meta_text: str,
    parent_quals: set[str],
    staged_quals: set[str],
) -> str | None:
    """Reject reason, or None when the build matches what was routed."""
    act = (action or "").strip().lower()
    if act not in (MODIFY, REPLACE, ADD):
        return None  # no declared action — the legacy path has no proposal

    new_variants = set(staged_quals) - set(parent_quals)
    touched = _touched_variants(files_changed)
    touched_incumbents = touched & set(parent_quals)

    if act == MODIFY:
        if target_variant and target_variant not in touched:
            return (
                f"action=modify was routed at {target_variant}, but nothing in "
                "the change touches it: "
                + (", ".join(sorted(touched)) or "no variant file changed")
            )
        strays = sorted(touched_incumbents - {target_variant})
        if strays:
            return (
                "action=modify may only change its own target; this also "
                "rewrote " + ", ".join(strays)
            )
        return None

    if act == ADD:
        if not new_variants:
            return (
                "action=add registered no new variant — the library after the "
                "change offers exactly what it offered before"
            )
        if touched_incumbents:
            return "action=add must not edit an incumbent; this changed " + ", ".join(
                sorted(touched_incumbents)
            )
        return None

    # REPLACE
    if not new_variants:
        return (
            "action=replace registered no new variant — there is nothing for "
            f"{target_variant or 'the incumbent'} to be replaced BY"
        )
    if target_variant:
        declared = _supersede_targets(variant_meta_text)
        if target_variant not in declared and target_variant.split("/")[-1] not in (
            declared
        ):
            return (
                f"action=replace must declare SUPERSEDES: {target_variant} in "
                "its <variant_meta>, or the archive keeps that variant active "
                "and the composer goes on selecting the thing this replaces"
            )
        # "Retired" is a statement about the LIBRARY, not about the file: the
        # composer selects what `build_default_library` offers, and the archive
        # records what SUPERSEDES declares. So an incumbent whose file was
        # edited on the way OUT — emptied until it no longer `register()`s — is
        # retired, and rejecting that costs a generation for doing the job more
        # thoroughly than the minimum. Only an incumbent that is still on offer
        # AND still being edited is the "kept growing it in place" this catches.
        if target_variant in touched and target_variant in staged_quals:
            return (
                f"action=replace must retire {target_variant}, not keep growing "
                "it in place"
            )
    strays = sorted(touched_incumbents - {target_variant})
    if strays:
        return (
            "action=replace may only retire its own target; this also rewrote "
            + ", ".join(strays)
        )
    return None
