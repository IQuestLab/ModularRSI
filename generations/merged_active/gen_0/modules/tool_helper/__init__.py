"""Solver helper tools — extra ACTIONS the agent can call, one file each.

Intentionally EMPTY at gen_0: the agent starts with exactly terminus-2's action
surface, and every helper here is something evolution added. An empty dir means
the mechanism is dormant, not broken.

A helper file is an ordinary module file (`NAME`, `USAGE`, `DESCRIPTION`,
`NICHE`, `async def run(args, ctx)`, `register(library)` registering under
`type_="tool_helper"`), so it gets an archive entry, a niche cell, DAG lineage
and a status like any module variant. See the `adding-a-solver-tool` skill.

Unlike the five module types, helpers are not pick-one: a task is given a SUBSET
of the active helpers, all at once.
"""
