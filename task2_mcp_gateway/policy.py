"""Tool-level authorization policy.

The task states the rule as "if ``params.name`` starts with ``admin_``, require
the admin role". That rule is implemented, but as *data* rather than as an
``if`` buried in the request handler, for two reasons: a policy you can print is
a policy you can audit, and the next rule ("billing_ tools need the finance
role") should not require touching the proxy.

The subtle part is name normalisation. ``"admin_reset_key".startswith("admin_")``
is easy; the attack is everything that is not exactly that string but that the
*downstream* server may still treat as that tool:

    "Admin_reset_key"     -- downstream lookup may be case-insensitive
    " admin_reset_key"    -- leading whitespace
    "ADMIN_RESET_KEY"     -- shouting
    "\u0410dmin_reset_key"    -- Cyrillic capital A, renders identically to "A"

So the check runs on an NFKC-normalised, case-folded, stripped copy of the name,
while the *original* string is what gets forwarded. Normalising the forwarded
name instead would be its own bug: the gateway would be silently rewriting the
caller's request.

Normalisation alone is not enough, and it is worth being precise about why.
NFKC folds *compatibility* equivalents -- full-width "\uff41" becomes "a" -- but Cyrillic
"\u0410" and Latin "A" are genuinely different characters, so no normal form will
ever unify them. (In this gateway the charset guard runs first, so a full-width
name is rejected with -32602 before normalisation is ever consulted; the folding
is defence in depth for a deployment that relaxes the guard.) Chasing homoglyphs is unwinnable. The answer is to constrain
the input instead: ``is_wellformed_tool_name`` restricts names to the ASCII set
MCP tool names actually use, and the gateway rejects anything else with -32602
before the policy is consulted. Deny by charset, not by lookalike table.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from auth import Principal

# MCP tool names in the wild are ASCII identifiers. Anything outside this set is
# rejected before the policy runs -- see the module docstring.
#
# \Z, not $. In Python's re, "$" also matches immediately before a trailing
# newline, so r"^[a-z_]+$" happily accepts "admin_reset_key\n". A name carrying a
# newline is exactly the shape you want for log forging or for slipping past a
# downstream that strips whitespace before lookup. \Z anchors at the true end of
# the string with no exception.
TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,128}\Z")


def is_wellformed_tool_name(name: str) -> bool:
    """Whether a tool name is inside the character set the gateway will reason about."""
    return isinstance(name, str) and TOOL_NAME_RE.match(name) is not None


# A JSON-RPC method name in MCP is an ASCII path like "tools/call".
METHOD_NAME_RE = re.compile(r"^[A-Za-z0-9_./-]{1,128}\Z")


def is_wellformed_method(method: object) -> bool:
    """Whether a method name is inside the character set the gateway reasons about."""
    return isinstance(method, str) and METHOD_NAME_RE.match(method) is not None


def normalize_method(method: str) -> str:
    """Fold a method name to the form routing decisions are made against.

    This exists because the tool-name lesson applies one layer up and was missed
    the first time. ``authorize()`` originally gated on ``method != "tools/call"``
    with an exact ASCII compare, so ``Tools/Call``, ``TOOLS/CALL``,
    ``"tools/call "`` and ``" tools/call"`` skipped the authorization branch
    entirely -- and were then *forwarded* to the downstream, audited as
    ``forwarded`` rather than ``denied``. A viewer could reach ``admin_reset_key``
    that way. The shipped mock downstream happens to reject those spellings
    because its own dispatch is exact-match, which is precisely the downstream
    leniency this gateway is not allowed to depend on.

    JSON-RPC method names are case-sensitive, so a strict downstream would treat
    ``Tools/Call`` as unknown. That is not the point: the gateway must not let a
    request it did not understand through, whatever the downstream would do
    with it.
    """
    return unicodedata.normalize("NFKC", method).strip().casefold()


def normalize_tool_name(name: str) -> str:
    """Fold a tool name to the form the policy is evaluated against."""
    return unicodedata.normalize("NFKC", name).strip().casefold()


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str
    rule: str


@dataclass(frozen=True)
class ToolPolicy:
    """Prefix-based tool authorization.

    ``prefix_roles`` maps a normalised tool-name prefix to the set of roles
    permitted to call it. A tool matching no prefix is allowed, which matches
    the task's stated rule; flip ``default_allow`` to ``False`` for a
    deny-by-default posture, which is what you would run in production once the
    tool inventory is known.
    """

    prefix_roles: dict[str, frozenset[str]] = field(
        default_factory=lambda: {"admin_": frozenset({"admin"})}
    )
    default_allow: bool = True

    def evaluate(self, tool_name: str, principal: Principal) -> Decision:
        normalized = normalize_tool_name(tool_name)

        for prefix, allowed_roles in self.prefix_roles.items():
            if normalized.startswith(prefix):
                if principal.role in allowed_roles:
                    return Decision(
                        allowed=True,
                        reason=f"role '{principal.role}' satisfies prefix rule '{prefix}'",
                        rule=f"prefix:{prefix}",
                    )
                return Decision(
                    allowed=False,
                    reason=(
                        f"tool '{tool_name}' requires one of "
                        f"{sorted(allowed_roles)}; caller has role '{principal.role}'"
                    ),
                    rule=f"prefix:{prefix}",
                )

        if self.default_allow:
            return Decision(allowed=True, reason="no restrictive rule matched", rule="default_allow")
        return Decision(
            allowed=False,
            reason=f"tool '{tool_name}' is not on the allow-list",
            rule="default_deny",
        )

    def visible_to(self, tool_name: str, principal: Principal) -> bool:
        """Whether this tool should appear in ``tools/list`` for this caller."""
        return self.evaluate(tool_name, principal).allowed
