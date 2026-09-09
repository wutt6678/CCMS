#!/usr/bin/env python3
"""Render the Iteration 11 paper's tables into LaTeX and Markdown, from the numbers file.

``scripts/iter11_paper_numbers.py`` filed one document holding every number the
paper quotes, with a rendering spec per column and, for the claim tables, a
rendering spec per row. It also filed the requirement this stage exists to meet:

    not_rendered: this file holds values and a rendering spec per column. The
    .tex and Markdown renderers consume it and hold no numbers of their own, so
    rounding is decided once and two renderings of one table cannot disagree

So this script holds no numbers. It reads the numbers file, turns each cell into
one canonical string under the spec that cell was filed with, and encodes that
string twice: once for LaTeX and once for Markdown. Rounding happens in exactly
one function, :func:`canonical_cell`, which means the two renderings cannot drift
apart by rounding differently -- there is only one rounding. The only permitted
difference between them is the escaping each format's syntax forces, and
``--verify`` checks that difference by unescaping the LaTeX and requiring the
Markdown back, cell by cell, for all of them.

Three things this stage decides, each one measured out of the numbers file rather
than chosen by eye and each one filed in ``paper/tables/renderings.json``:

The column layout. Every column's printed width is measured -- the longer of its
header and its widest cell, with LaTeX macros discounted because ``\\kappa``
costs one glyph and six characters -- and the table's natural width is compared
with the width of a text block. A table with a cell too long for a fixed column
becomes a ``tabularx`` whose prose columns wrap, weighted by the square root of
their measured width so that a 347-character value column does not starve the
column that names it. A table with too many columns and no long cell becomes a
``resizebox``. A table that fits is left alone. Which of the three fired, and the
measurements that decided it, are filed per table.

What is escaped and what is not. Column headers and captions are AUTHORED LaTeX:
they carry ``$\\Delta_{TV}$``, ``$p > \\alpha$`` and ``95\\% CI``, and escaping
them would print the mathematics as literal dollar signs and backslashes. Cell
values, notes, claims and artifact names are LITERAL TEXT: 108 of them carry an
underscore, one carries a shell ``&&``, four carry a JSON brace pair, and any of
them unescaped either fails to compile or, worse, silently moves a column
boundary at the ``&``. The split is therefore not a convention but a measurement,
and it is checked in both directions: nothing this stage escapes may contain
``$`` or ``\\``, and nothing it passes through may contain a bare ``&``, a bare
``%``, an unbalanced ``$`` or ``{``, or an underscore outside math mode.

What an absent value prints as. Eleven cells are filed as ``null`` under an
``optional_*`` spec. They print as ``--`` rather than as nothing, because an
empty cell is indistinguishable from a cell the renderer dropped, and a reader
cannot tell a missing measurement from a bug in the print.

The renderer also refuses to guess. Every spec in the numbers stage's own
``FORMATS`` vocabulary is implemented here and the two sets are compared, so a
spec added upstream becomes a finding instead of a cell rendered by whatever
``str()`` does. A cell whose Python type disagrees with its spec is a finding
rather than a coercion, because coercing a float filed under ``int`` or a count
filed under ``str`` is how a table starts printing 0.49498327759197325. A
p-value that would print as ``0.0000`` is refused: the numbers file establishes
that the smallest p this evidence can report is 1/5000, and printing a nonzero p
as zero overstates it. An interval whose lower end exceeds its upper end is
refused. A cell containing a newline is refused, because neither format can put
one inside a table row.

Writing is gated on the numbers themselves. ``--write`` first asks the numbers
stage to re-derive its own document and refuses to print from a numbers file that
does not, since that is the one moment a number enters the paper and the last
moment anybody can stop it. ``--verify`` does not repeat that check: it verifies
this stage's own contract, and the closeout's ``--deep`` runs the numbers verifier
beside it, so the composition covers a stale numbers file without this stage
re-reading twenty-two artifacts on every invocation.

Exit codes, and each one is tested:
    0  verified, or written
    1  a rendering disagrees with the numbers file, a cell contradicts the spec it
       was filed under, a spec is not implemented, or the numbers file this stage
       was asked to print from does not itself re-derive
    2  no numbers file to render from, or no renderings filed to verify against
    3  not reachable for this stage: its one input is a committed JSON file, so no
       checkout that can run this lacks a section of it
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import textwrap
from decimal import Decimal
from pathlib import Path
from typing import NoReturn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import iter11_paper_numbers as numbers  # noqa: E402

NUMBERS_PATH = numbers.OUT_PATH
OUT_DIR = REPO_ROOT / "paper" / "tables"
RENDERINGS_NAME = "renderings.json"
TEX_SUBDIR = "tex"
MARKDOWN_SUBDIR = "markdown"
ALL_TEX_NAME = "00_all.tex"

#: What this document is, in the ``kind`` field every artifact in this iteration
#: carries. Bumping it is how a change to the schema is announced to a verifier.
KIND = "iteration_11_paper_table_renderings_v1"

QUESTION = (
    "what the twelve filed tables look like in print, in both the formats the "
    "paper is written in, derived from the numbers file and agreeing with each "
    "other cell by cell"
)

#: The companion key a claim-table row carries to say how its own value renders.
#: It is never a declared column key in any of the twelve tables, so a row-level
#: ``format`` is always a rendering spec and never data.
ROW_FORMAT_KEY = "format"

#: The column a row-level spec governs. ``key_value_table`` in the numbers stage
#: builds rows as ``(claim, value, format, where)`` and declares all three
#: columns ``str``, because one column mixes counts, hashes, booleans and prose
#: and a single spec for all of them would either round a hash or print a count
#: as a decimal. The row's own spec is therefore the truth about its value cell.
ROW_FORMAT_GOVERNS = "value"

#: What an absent value prints as, in both formats. See the module docstring.
ABSENT = "--"

#: The characters LaTeX gives a meaning to. A single pass over this class is what
#: makes escaping safe: replacing ``\\`` first and then the others would re-escape
#: the braces inside ``\\textbackslash{}``.
LATEX_HOSTILE = re.compile(r"[\\&%$#_{}~^]")

LATEX_ESCAPES = {
    "\\": "\\textbackslash{}",
    "&": "\\&",
    "%": "\\%",
    "$": "\\$",
    "#": "\\#",
    "_": "\\_",
    "{": "\\{",
    "}": "\\}",
    "~": "\\textasciitilde{}",
    "^": "\\textasciicircum{}",
}

_LATEX_UNESCAPES = {escaped: raw for raw, escaped in LATEX_ESCAPES.items()}
LATEX_UNESCAPE = re.compile(
    "|".join(re.escape(item) for item in
             sorted(_LATEX_UNESCAPES, key=len, reverse=True)))

#: The character Markdown gives a meaning to inside a table row. A newline would
#: end the row, but one is refused rather than substituted, so it is not here.
MARKDOWN_HOSTILE = "|"

#: An underscore that Markdown could read as the start of emphasis rather than as
#: part of an identifier. CommonMark will not open emphasis with an underscore
#: that sits inside a word, which is where every underscore this stage has met
#: sits -- ``dependency_lock``, ``outputs/iteration_11``, ``CMST_456921/text_only``
#: -- so nothing here is escaped for it. That is a measurement and not a
#: convention, and this pattern is what re-measures it: an underscore that is not
#: preceded by a word character CAN open emphasis, and one that does has to be
#: escaped or the note would print part of an identifier in italics.
MARKDOWN_EMPHASIS_RISK = re.compile(r"(?<!\w)_\S")

#: A note is broken into source lines no longer than this.
NOTE_SOURCE_WIDTH = 99

#: Fields the numbers stage authored as LaTeX, which pass through untouched.
AUTHORED_LATEX_FIELDS = ("caption", "header")

#: Fields that are literal text and are escaped. ``note`` is here rather than
#: above because every filed note is prose: three carry an underscore from a
#: literal identifier such as ``tab:panel_exclusions`` and none carries a single
#: ``$`` or ``\``, which is what makes escaping them correct rather than
#: destructive. The check below re-measures that on every run.
ESCAPED_FIELDS = ("note", "cell")

#: A text block is about this many characters wide at 10pt Computer Modern in a
#: standard article: 6.5in of \textwidth at roughly 5pt per character. It is an
#: estimate and is filed as one, because the alternative -- compiling the table
#: and measuring the box -- would make this document depend on a TeX
#: installation, and the figures were kept out of the numbers file for exactly
#: that reason.
CHARS_PER_TEXTWIDTH = 93.0

#: \tabcolsep is 6pt on each side of every column, so each column costs about
#: this much of the text block before it holds a single character.
TABCOLSEP_CHARS = 2.4

#: A column whose widest printed cell exceeds this cannot sit in a fixed ``l``
#: column: TeX will not break it and it will run out of the text block. Measured
#: against the twelve tables as filed, the widest such cell is 347 characters
#: (the ``pip install`` line in tab:environment) and the widest cell in any table
#: that takes the resizebox route is 56, so the two cases do not overlap.
WRAP_ABOVE_CHARS = 60

#: Once a table is wrapping at all, a column wider than this FRACTION of the text
#: block wraps too. ``tabularx`` gives a fixed column its natural width and the
#: wrapping columns whatever is left, so without this the column that triggered
#: the wrap would share the block with nothing and the columns beside it would
#: still eat it: tab:environment's 47-character claim column left fixed takes half
#: the block and leaves its 347-character value column one word per line.
WRAP_SHARE_OF_BUDGET = 0.25

#: Wrapping columns share the text block in proportion to the SQUARE ROOT of
#: their measured width. Proportional to the width itself would give
#: tab:environment's 347-character value column 88% of the block and leave its
#: claim column one word per line; the square root keeps the longer column wider
#: without starving the one that names it.
WRAP_WEIGHT_POWER = 0.5

#: How many decimal places each rounding spec prints. Filed beside the rounding
#: block so that the precision is stated where the convention is, rather than left
#: to be inferred from reading :func:`_base_renderers`.
ROUNDING_DIGITS = {"float2": 2, "float4": 4, "signed4": 4, "p4": 4, "pct1": 1}

#: Where each base spec sits in a column. Numbers right-align so that their
#: decimal points line up, booleans centre because they are a mark rather than a
#: quantity, and text left-aligns.
ALIGNMENT = {
    "str": "l",
    "int": "r",
    "float2": "r",
    "float4": "r",
    "signed4": "r",
    "pct1": "r",
    "p4": "r",
    "sha8": "r",
    "ci4": "r",
    "bool": "c",
}

#: How many hex characters ``sha8`` prints. The filed values are whole digests --
#: 64 characters for a sha256 and 40 for the resolved model revision, which is a
#: git id -- so this spec is a prefix and the renderings file says so beside the
#: full length of every value it shortened.
SHA8_CHARS = 8

MECHANISMS = ("plain_tabular", "resizebox", "tabularx")

#: The LaTeX packages the emitted tables need, filed because r20 has to put them
#: in the preamble and a missing one is a compilation failure nobody can trace
#: back to a table.
PACKAGE_FOR_MECHANISM = {
    "plain_tabular": ("booktabs",),
    "resizebox": ("booktabs", "graphicx"),
    "tabularx": ("booktabs", "tabularx"),
}


def _rel(path: Path | str) -> str:
    try:
        return str(Path(path).relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def fatal(message: str, code: int) -> NoReturn:
    """Exit with the code the docstring says this failure means.

    ``raise SystemExit("text")`` exits 1 whatever the text says, and 1 is this
    script's code for a CONTRADICTION. A numbers file that is not there reported
    as a contradiction sends whoever reads the exit code to look for a
    disagreement between a table and an artifact that was never read.
    """
    print(f"FATAL: {message}", file=sys.stderr)
    raise SystemExit(code)


def load_numbers(path: Path) -> dict:
    """The one input. Everything printed is read out of it."""
    if not path.exists():
        fatal(f"no paper numbers at {_rel(path)}; this stage renders and holds no "
              f"numbers of its own, so run "
              f"`python scripts/iter11_paper_numbers.py --write` first", 2)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fatal(f"{_rel(path)} is not readable JSON ({exc}); it is the paper's only "
              f"source of numbers and this stage does not repair it", 1)
        raise  # unreachable; keeps the return type honest for the type checker


def numbers_re_derive(path: Path) -> tuple[int, str, list[str]]:
    """Ask the numbers stage whether its own document still re-derives.

    Wrapped so that a test can stub the upstream stage without stubbing the whole
    of it, and so that the dependency is visible as a call rather than buried in
    a code path.
    """
    return numbers.verify(path)


# ---------------------------------------------------------------------------
# Canonical cells: the only place a number is rounded
# ---------------------------------------------------------------------------

def _is_number(value: object) -> bool:
    """A float or an int that is not a bool.

    ``isinstance(True, int)`` is True in Python, so a boolean filed under a
    numeric spec would otherwise be accepted and printed as ``1``.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _fixed(value: float, digits: int) -> str:
    return f"{value:.{digits}f}"


def _base_renderers() -> dict:
    """One callable per base spec, each returning canonical plain text.

    Each raises ``ValueError`` with a sentence that says what was filed and what
    the spec required. The caller turns that into a finding rather than a crash,
    so one bad cell is reported beside the rest instead of ending the run.
    """

    def as_str(value):
        if not isinstance(value, str):
            raise ValueError(
                f"filed as {type(value).__name__} {value!r} under the spec 'str', "
                f"which is literal text; coercing it would print a float at full "
                f"precision or a boolean as 'True', and the row-level spec exists "
                f"precisely so that this cell does not have to be guessed")
        return value

    def as_int(value):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"filed as {type(value).__name__} {value!r} under the "
                             f"spec 'int', which prints a count")
        return f"{value:d}"

    def as_bool(value):
        if not isinstance(value, bool):
            raise ValueError(f"filed as {type(value).__name__} {value!r} under a "
                             f"boolean spec")
        # The filed spelling, so that a reader can grep the artifact for exactly
        # what the table printed. 'yes'/'no' or a check mark would read better and
        # would be the renderer's first piece of vocabulary of its own.
        return "true" if value else "false"

    def as_float(digits):
        def render(value):
            if not _is_number(value):
                raise ValueError(f"filed as {type(value).__name__} {value!r} under "
                                 f"a {digits}-decimal spec")
            return _fixed(float(value), digits)
        return render

    def as_signed4(value):
        if not _is_number(value):
            raise ValueError(f"filed as {type(value).__name__} {value!r} under the "
                             f"spec 'signed4', which prints an explicit sign")
        return f"{float(value):+.4f}"

    def as_pct1(value):
        if not _is_number(value):
            raise ValueError(f"filed as {type(value).__name__} {value!r} under the "
                             f"spec 'pct1'")
        # The filed value is a FRACTION: the five shares in
        # tab:vision_ablation_shift sum to exactly 1.0, which is the measurement
        # that licenses multiplying by 100 rather than assuming it. Reading it
        # the other way would print 76.6% as 0.8%, a hundredfold error that no
        # reader of the table could detect.
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"filed as {value!r} under the spec 'pct1', which is a "
                             f"fraction of a whole; {value} is outside [0, 1], so "
                             f"either it is already a percentage or it is not a "
                             f"share, and this stage does not guess which")
        return f"{float(value) * 100.0:.1f}%"

    def as_p4(value):
        if not _is_number(value):
            raise ValueError(f"filed as {type(value).__name__} {value!r} under a "
                             f"p-value spec")
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"filed as {value!r} under a p-value spec, which is a "
                             f"probability")
        text = _fixed(float(value), 4)
        if text == "0.0000" and float(value) != 0.0:
            raise ValueError(
                f"filed as {value!r}, which 'p4' would print as 0.0000: a nonzero "
                f"p printed as zero claims an impossibility rather than a small "
                f"probability, and the numbers file establishes that the smallest "
                f"p this evidence can report is 1/5000 = 0.0002, so nothing filed "
                f"here should be under it")
        return text

    def as_sha8(value):
        if not isinstance(value, str):
            raise ValueError(f"filed as {type(value).__name__} {value!r} under the "
                             f"spec 'sha8', which prints a hex digest prefix")
        if len(value) < SHA8_CHARS:
            raise ValueError(f"filed as {value!r}, which is {len(value)} characters "
                             f"and shorter than the {SHA8_CHARS} 'sha8' prints")
        if any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"filed as {value!r}, which is not lowercase hex, so "
                             f"it is not a digest this spec can shorten")
        return value[:SHA8_CHARS]

    def as_ci4(value):
        if not isinstance(value, list):
            raise ValueError(f"filed as {type(value).__name__} {value!r} under the "
                             f"spec 'ci4', which prints an interval")
        if len(value) != 2:
            raise ValueError(f"filed as {value!r}, which has {len(value)} ends; an "
                             f"interval has two")
        low, high = value
        if not (_is_number(low) and _is_number(high)):
            raise ValueError(f"filed as {value!r}, whose ends are "
                             f"{type(low).__name__} and {type(high).__name__}")
        if float(low) > float(high):
            raise ValueError(f"filed as [{low}, {high}], whose lower end exceeds its "
                             f"upper end, so it is not an interval")
        return f"[{_fixed(float(low), 4)}, {_fixed(float(high), 4)}]"

    return {
        "str": as_str,
        "int": as_int,
        "bool": as_bool,
        "float2": as_float(2),
        "float4": as_float(4),
        "signed4": as_signed4,
        "pct1": as_pct1,
        "p4": as_p4,
        "sha8": as_sha8,
        "ci4": as_ci4,
    }


RENDERERS = _base_renderers()

#: Every spec :func:`canonical_cell` can render: each base spec, and the optional
#: form of each. This is the renderer's CAPABILITY and it is deliberately wider
#: than the vocabulary the numbers stage declares, because ``optional_pct1`` costs
#: nothing to support once ``pct1`` exists and refusing it would be a restriction
#: nobody asked for. What is not permitted is the other direction -- a spec
#: declared upstream and not renderable here -- and that comparison is recomputed
#: on every run rather than remembered.
CAPABLE_FORMATS = frozenset(
    list(RENDERERS) + [f"optional_{name}" for name in RENDERERS])


def declared_but_not_renderable() -> list[str]:
    """Specs the numbers stage declares that this stage could not print.

    Recomputed rather than filed as a constant, because the numbers stage's
    vocabulary is not this module's to freeze. A spec added upstream has to arrive
    here as a finding; a renderer that quietly did not know it would print the
    cell with ``str()``, and the two renderings of one table would start to differ
    in the one place the numbers file says they cannot.
    """
    return sorted(set(numbers.FORMATS) - CAPABLE_FORMATS)


def split_spec(spec: str) -> tuple[bool, str]:
    """Whether a spec tolerates an absent value, and the spec underneath it."""
    if spec.startswith("optional_"):
        return True, spec[len("optional_"):]
    return False, spec


def canonical_cell(spec: str, value: object, where: str) -> tuple[str, str | None]:
    """One cell as plain text, and the finding if it cannot be rendered.

    Total: it never raises. A cell that contradicts its spec is returned as a
    finding so that it is reported beside every other finding, because a renderer
    that stops at the first bad cell tells the author about one problem and hides
    the rest.
    """
    if spec not in CAPABLE_FORMATS:
        return "", (f"{where}: filed under the rendering spec {spec!r}, which is "
                    f"not one of the {len(CAPABLE_FORMATS)} this stage can render; "
                    f"a renderer that meets an unknown spec has to guess, and two "
                    f"renderers guess differently")
    optional, base = split_spec(spec)
    if value is None:
        if not optional:
            return "", (f"{where}: filed as null under the spec {spec!r}, which does "
                        f"not tolerate an absent value; either the cell is missing "
                        f"or the spec should be 'optional_{base}'")
        return ABSENT, None
    if optional and isinstance(value, str) and value == ABSENT:
        return "", (f"{where}: filed as the literal string {ABSENT!r}, which is what "
                    f"this stage prints an absent value as; a reader could not tell "
                    f"the two apart, so an absent value is filed as null")
    try:
        return RENDERERS[base](value), None
    except ValueError as exc:
        return "", f"{where}: {exc}"


# ---------------------------------------------------------------------------
# Escaping: the one place the two formats are allowed to differ
# ---------------------------------------------------------------------------

def latex_text(text: str) -> str:
    """Escape literal text for LaTeX, in a single pass.

    One pass matters: replacing ``\\\\`` first and then the others would re-escape
    the braces this function just introduced in ``\\\\textbackslash{}``.
    """
    return LATEX_HOSTILE.sub(lambda match: LATEX_ESCAPES[match.group(0)], text)


def latex_unescape(text: str) -> str:
    """The inverse of :func:`latex_text`, longest replacement first."""
    return LATEX_UNESCAPE.sub(lambda match: _LATEX_UNESCAPES[match.group(0)], text)


def markdown_text(text: str) -> str:
    """Escape literal text for a Markdown table row."""
    return text.replace(MARKDOWN_HOSTILE, "\\" + MARKDOWN_HOSTILE)


def markdown_unescape(text: str) -> str:
    return text.replace("\\" + MARKDOWN_HOSTILE, MARKDOWN_HOSTILE)


def _math_segments(text: str) -> list[tuple[str, bool]]:
    """Split authored LaTeX into (segment, in_math) pairs on ``$``."""
    parts = text.split("$")
    return [(part, index % 2 == 1) for index, part in enumerate(parts)]


def check_authored_latex(text: str, where: str, field: str) -> list[str]:
    """That a field passed through untouched will actually compile.

    These fields are authored LaTeX and this stage does not escape them, which is
    the right thing only for as long as they stay well-formed. Each check below is
    a way a header or a caption can break a table silently or loudly, and every
    one of them is a finding rather than a repair: this stage does not rewrite
    prose the numbers stage filed.
    """
    issues: list[str] = []
    if text.count("$") % 2:
        issues.append(f"{where}: the authored {field} {text!r} has "
                      f"{text.count('$')} '$' characters, which is odd, so it opens "
                      f"a math group it never closes and LaTeX will read the rest "
                      f"of the table as mathematics")
    if text.count("{") != text.count("}"):
        issues.append(f"{where}: the authored {field} {text!r} has "
                      f"{text.count('{')} '{{' and {text.count('}')} '}}', so a "
                      f"group it opens is never closed")
    if "&" in text:
        issues.append(f"{where}: the authored {field} {text!r} contains '&', which "
                      f"ends a table cell; this stage passes authored LaTeX "
                      f"through unescaped, so an '&' here would move every "
                      f"following column of this row one place to the left and the "
                      f"table would still compile")
    for index, character in enumerate(text):
        if character == "%" and (index == 0 or text[index - 1] != "\\"):
            issues.append(f"{where}: the authored {field} {text!r} contains a bare "
                          f"'%' at position {index}, which comments out the rest of "
                          f"the line; it has to be written '\\%'")
            break
    for segment, in_math in _math_segments(text):
        if in_math:
            continue
        for index, character in enumerate(segment):
            if character == "_" and (index == 0 or segment[index - 1] != "\\"):
                issues.append(f"{where}: the authored {field} {text!r} contains an "
                              f"underscore outside math mode, which LaTeX reads as "
                              f"a subscript and refuses with 'Missing $ inserted'; "
                              f"it has to be written '\\_' or moved inside "
                              f"mathematics")
                break
    return issues


def check_escaped_field(text: str, where: str, field: str) -> list[str]:
    """That a field this stage escapes carries no authored mathematics.

    The other half of the asymmetry. Escaping ``$\\Delta_{TV}$`` would print
    ``$Delta_TV$`` as literal characters, so a note or a cell that grows
    mathematics is a decision the author has to make in the numbers stage, where
    the field can be moved to the authored side deliberately. It is not a thing
    this stage should discover by having quietly ruined it.
    """
    issues = []
    for character in ("$", "\\"):
        if character in text:
            issues.append(
                f"{where}: the {field} {text[:80]!r} contains {character!r}, which "
                f"means it carries authored LaTeX; this stage escapes every "
                f"{field} as literal text, so the mathematics would print as "
                f"punctuation. Move it to a field listed in "
                f"AUTHORED_LATEX_FIELDS or restate it in words")
    return issues


# ---------------------------------------------------------------------------
# The grid: every cell of every table, as canonical text
# ---------------------------------------------------------------------------

def effective_spec(table: dict, row: dict, key: str) -> tuple[str, str]:
    """The spec that governs one cell, and where it came from.

    A claim-table row carries its own spec for its value cell. The column it is
    declared under says ``str`` for all three columns, which is true of the claim
    and of the artifact name and wrong for the value, so the row wins there and
    the column wins everywhere else.
    """
    column_spec = {column["key"]: column["format"] for column in table["columns"]}
    spec = column_spec[key]
    if key == ROW_FORMAT_GOVERNS and ROW_FORMAT_KEY in row:
        return row[ROW_FORMAT_KEY], "row"
    return spec, "column"


REQUIRED_TABLE_FIELDS = ("label", "caption", "columns", "rows")
REQUIRED_COLUMN_FIELDS = ("key", "header", "format")


def unprintable_reasons(name: str, table: object) -> list[str]:
    """What is missing from a table that would stop it being printed at all.

    ONE definition of printable, used by the renderer, by the agreement check and
    by the census, so that a malformed table is skipped by all three rather than
    reported by one and crashed on by another. A census that raises turns a finding
    the renderer already filed into a traceback, and a traceback is the one output
    that tells a reader nothing about which table was wrong.
    """
    if not isinstance(table, dict):
        return [f"{name}: filed as {type(table).__name__}, not an object"]
    reasons = [f"{name}: files no {field!r}, which a table has to have to be "
               f"printable" for field in REQUIRED_TABLE_FIELDS if field not in table]
    if reasons:
        return reasons
    if not isinstance(table["columns"], list) or not isinstance(table["rows"], list):
        return [f"{name}: files columns or rows that are not a list"]
    for index, column in enumerate(table["columns"]):
        if not isinstance(column, dict):
            reasons.append(f"{name}: column {index} is {type(column).__name__}, not "
                           f"an object with a key, a header and a format")
            continue
        reasons.extend(f"{name}: column {index} files no {field!r}, so there is "
                       f"nothing to print under it"
                       for field in REQUIRED_COLUMN_FIELDS if field not in column)
    return reasons


def is_printable(table: object) -> bool:
    return not unprintable_reasons("", table)


def render_grid(name: str, table: dict) -> tuple[list[list[str]], list[str], dict]:
    """Every cell of one table as canonical text, plus what it took to get there.

    Returns the grid, the findings, and the counts: how many cells were governed
    by a row-level spec, which specs were used, and which row keys were dropped
    for not being declared columns. Nothing is silently discarded: a companion key
    is provenance the numbers stage filed beside a row and the paper does not
    print, so the ones dropped are named here rather than vanishing.
    """
    findings: list[str] = []
    column_keys = [column["key"] for column in table["columns"]]
    declared = set(column_keys)
    rows = table["rows"]

    if not rows:
        findings.append(f"{name}: filed with no rows, so there is nothing to print; "
                        f"a table with a header and no body is a hole in the paper "
                        f"and not a smaller table")
    if not column_keys:
        findings.append(f"{name}: filed with no columns")

    if ROW_FORMAT_KEY not in declared:
        with_row_spec = [index for index, row in enumerate(rows)
                         if ROW_FORMAT_KEY in row]
        if with_row_spec and ROW_FORMAT_GOVERNS not in declared:
            findings.append(
                f"{name}: {len(with_row_spec)} row(s) carry their own rendering "
                f"spec under {ROW_FORMAT_KEY!r} but the table has no "
                f"{ROW_FORMAT_GOVERNS!r} column for it to govern, so those specs "
                f"would be read and then ignored")

    grid: list[list[str]] = []
    specs_used: dict[str, int] = {}
    row_governed = 0
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            findings.append(f"{name} row {index}: filed as "
                            f"{type(row).__name__}, not an object")
            continue
        for key in column_keys:
            if key not in row:
                findings.append(f"{name} row {index}: declares the column {key!r} "
                                f"but files no value for it, so the cell would be "
                                f"blank and indistinguishable from an absent one")
        line: list[str] = []
        for key in column_keys:
            where = f"{name} row {index} column {key!r}"
            spec, origin = effective_spec(table, row, key)
            if origin == "row":
                row_governed += 1
            specs_used[spec] = specs_used.get(spec, 0) + 1
            text, finding = canonical_cell(spec, row.get(key), where)
            if finding:
                findings.append(finding)
            if isinstance(text, str) and any(c in text for c in "\n\r\t"):
                findings.append(f"{where}: renders to {text!r}, which contains a "
                                f"newline or a tab; neither format can hold one "
                                f"inside a table row, and substituting a space "
                                f"would print something the artifact does not say")
                text = ""
            line.append(text)
        grid.append(line)

    companions = sorted({key for row in rows if isinstance(row, dict)
                         for key in row} - declared - {ROW_FORMAT_KEY})
    metadata = {
        "n_columns": len(column_keys),
        "n_rows": len(grid),
        "n_cells": sum(len(line) for line in grid),
        "n_cells_governed_by_a_row_level_spec": row_governed,
        "formats_used": dict(sorted(specs_used.items())),
        "companion_keys_filed_but_not_printed": companions,
    }
    return grid, findings, metadata


# ---------------------------------------------------------------------------
# Layout, measured
# ---------------------------------------------------------------------------

def _printed_width(text: str) -> int:
    """How wide authored LaTeX prints, not how many characters it is typed in.

    ``$\\kappa$ (refusal)`` is 19 characters and prints as 12. Counting source
    characters would over-widen every column whose header carries mathematics and
    push a table into a resizebox it did not need.
    """
    stripped = re.sub(r"\\[a-zA-Z]+", lambda match: match.group(0)[1:], text)
    for character in ("$", "{", "}", "\\"):
        stripped = stripped.replace(character, "")
    return len(stripped)


def column_alignment(table: dict, column: dict) -> tuple[str, bool]:
    """Where one column sits, and whether its cells are of mixed kinds.

    A claim table's ``value`` column takes its spec row by row, so it holds a
    count in one row, a digest in the next and a sentence after that. There is no
    single alignment for a column like that: right-aligning it would line up the
    decimal points of the numbers and leave the prose hanging, and taking the
    alignment from whichever row happened to be first would make the column's
    position in the paper depend on the order the claims were filed in. A column
    whose cells are all one kind gets that kind's alignment; a column whose cells
    are mixed is text and sits left.
    """
    bases = set()
    for row in table["rows"]:
        spec, _ = effective_spec(table, row, column["key"])
        bases.add(split_spec(spec)[1])
    if not bases:
        bases = {split_spec(column["format"])[1]}
    if len(bases) > 1:
        return "l", True
    return ALIGNMENT.get(next(iter(bases)), "l"), False


def column_layout(name: str, table: dict, grid: list[list[str]]) -> dict:
    """Which of the three LaTeX mechanisms this table needs, and why.

    Measured rather than chosen: the widths come from the rendered cells and the
    authored headers, the natural width is their sum plus what the column
    separators cost, and the mechanism follows from two comparisons. Filed in full
    so that a reviewer can see the layout is a consequence of the data and not of
    somebody's eye.
    """
    columns = table["columns"]
    widths: dict[str, int] = {}
    alignment: dict[str, str] = {}
    mixed: list[str] = []
    for position, column in enumerate(columns):
        key = column["key"]
        cells = [grid[index][position] for index in range(len(grid))]
        widest_cell = max((len(cell) for cell in cells), default=0)
        widths[key] = max(_printed_width(column["header"]), widest_cell)
        alignment[key], is_mixed = column_alignment(table, column)
        if is_mixed:
            mixed.append(key)

    natural = sum(widths.values()) + TABCOLSEP_CHARS * len(columns)
    # Two different questions, and conflating them is what starves a column. The
    # first is whether this table has to wrap at all: does any cell exceed what a
    # fixed column can hold. The second, asked only once the answer is yes, is
    # WHICH columns wrap: tabularx gives a fixed column its natural width and the
    # X columns whatever is left, so a 47-character claim column left fixed would
    # take half the text block and leave the 347-character column beside it one
    # word per line.
    unbreakable = [column["key"] for column in columns
                   if widths[column["key"]] > WRAP_ABOVE_CHARS]
    over_budget = natural > CHARS_PER_TEXTWIDTH
    share_floor = WRAP_SHARE_OF_BUDGET * CHARS_PER_TEXTWIDTH
    over_the_floor = [column["key"] for column in columns
                      if widths[column["key"]] > share_floor]
    wrapping: list[str] = []
    weights: dict[str, float] | None = None
    if unbreakable:
        mechanism = "tabularx"
        # WHICH columns wrap is a second question from whether the table wraps at
        # all, and conflating them is what starves a column: tabularx gives a fixed
        # column its natural width and the X columns whatever is left, so a
        # 47-character claim column left fixed would take half the text block and
        # leave the 347-character column beside it one word per line.
        wrapping = list(over_the_floor)
        widths_of = ", ".join(f"{key} at {widths[key]}" for key in unbreakable)
        sharing = ", ".join(f"{key} at {widths[key]}" for key in wrapping)
        why = (f"{len(unbreakable)} column(s) hold a cell wider than "
               f"{WRAP_ABOVE_CHARS} characters -- {widths_of} -- and LaTeX will not "
               f"break a fixed column, so this table wraps. {len(wrapping)} "
               f"column(s) measure over {WRAP_SHARE_OF_BUDGET} of the text block "
               f"({share_floor:.1f} characters) and share it: {sharing}")
        # Normalised so that they sum to the number of wrapping columns, which is
        # what tabularx requires of \hsize factors. The last one takes whatever is
        # left so that the sum is exact rather than exact to two decimals.
        raw = {key: math.pow(widths[key], WRAP_WEIGHT_POWER) for key in wrapping}
        total = sum(raw.values()) or 1.0
        scaled = {key: len(wrapping) * value / total for key, value in raw.items()}
        weights = {}
        for key in wrapping[:-1]:
            weights[key] = round(scaled[key], 2)
        weights[wrapping[-1]] = round(len(wrapping) - sum(weights.values()), 2)
        weight_text = ", ".join(f"{key} {weights[key]}" for key in wrapping)
        why += (f"; the widths are shared in proportion to the "
                f"{WRAP_WEIGHT_POWER} power of the measured width, {weight_text}, "
                f"because sharing them in proportion to the width itself would "
                f"give the widest column almost the whole block and leave the "
                f"others one word per line")
    elif over_budget:
        mechanism = "resizebox"
        why = (f"the widest column measures {max(widths.values(), default=0)} "
               f"characters against {WRAP_ABOVE_CHARS}, so no cell needs breaking, "
               f"but the columns together measure {natural:.1f} against a text "
               f"block of about {CHARS_PER_TEXTWIDTH:.0f}: the table is scaled to "
               f"fit and NO column wraps, so a column filed as wider than the "
               f"share floor is scaled with the rest rather than wrapped")
    else:
        mechanism = "plain_tabular"
        why = (f"the columns measure {natural:.1f} against a text block of about "
               f"{CHARS_PER_TEXTWIDTH:.0f} and no cell exceeds "
               f"{WRAP_ABOVE_CHARS} characters, so the table fits as it is and no "
               f"column wraps")

    return {
        "mechanism": mechanism,
        "why": why,
        "column_widths_chars": widths,
        "natural_width_chars": round(natural, 1),
        "text_block_budget_chars": CHARS_PER_TEXTWIDTH,
        "over_budget": over_budget,
        "unbreakable_columns": unbreakable,
        "columns_over_the_share_floor": over_the_floor,
        "wrapping_columns": wrapping,
        "wrap_weights": weights,
        "alignment": alignment,
        "mixed_spec_columns": mixed,
        "what_a_mixed_spec_column_is": (
            "a column whose cells take their rendering spec row by row, which is "
            "what a claim table's value column does. It is aligned as text because "
            "no single alignment fits a column that holds a count in one row, a "
            "digest in the next and a sentence after that"),
        "latex_packages_required": list(PACKAGE_FOR_MECHANISM[mechanism]),
    }


def _latex_column_spec(layout: dict, columns: list[dict]) -> str:
    parts = ["@{}"]
    weights = layout["wrap_weights"] or {}
    for column in columns:
        key = column["key"]
        if key in weights:
            parts.append(">{\\raggedright\\arraybackslash"
                         f"\\hsize={weights[key]:.2f}\\hsize}}X")
        else:
            parts.append(layout["alignment"][key])
    parts.append("@{}")
    return "".join(parts)


# ---------------------------------------------------------------------------
# The two encodings
# ---------------------------------------------------------------------------

def _provenance_lines(name: str, numbers_rel: str, numbers_sha: str,
                      source: str) -> list[str]:
    return [
        "% Rendered by scripts/iter11_paper_tables.py -- do not edit.",
        f"% Table {name!r}.",
        f"% Every cell is derived from {numbers_rel}",
        f"%   sha256 {numbers_sha}",
        f"% whose values are read out of the filed artifact {source!r}",
        "%   by scripts/iter11_paper_numbers.py.",
        "% Re-render with: python scripts/iter11_paper_tables.py --write",
    ]


def wrapped_note(text: str) -> str:
    """The note broken into source lines, when breaking it cannot change the print.

    The longest filed note is 2343 characters and would otherwise be one source
    line of 2396, which compiles fine and is unreviewable: changing one word of it
    would show as a whole-line diff in an artifact this repository re-derives and
    compares. A newline is a space to LaTeX and a soft break is a space to
    Markdown, so the wrap is invisible in both -- but only while the note's
    whitespace is already exactly one space between words. A note filed with a
    double space would be silently normalised by the wrap, so such a note is left
    on one line rather than quietly reworded.
    """
    if " ".join(text.split()) != text:
        return text
    return textwrap.fill(text, width=NOTE_SOURCE_WIDTH, break_long_words=False,
                         break_on_hyphens=False)


def check_markdown_literal(text: str, where: str, field: str) -> list[str]:
    """That Markdown will print this literal text as literal text.

    Markdown is escaped for '|' only. Underscores are deliberately left alone,
    because CommonMark will not open emphasis with an underscore that sits inside
    a word and every underscore this stage has met sits inside one. That is a
    measurement rather than a convention, so it is re-measured here: an underscore
    that is NOT inside a word can open emphasis, and one that does would print
    part of an identifier in italics with nothing else in the chain noticing.
    """
    match = MARKDOWN_EMPHASIS_RISK.search(text)
    if match is None:
        return []
    return [f"{where}: the {field} {text[:70]!r} has an underscore at position "
            f"{match.start()} that is not inside a word, which Markdown can read "
            f"as the start of emphasis and print in italics; escape it there or "
            f"restate it, because this stage escapes Markdown for '|' only"]


def latex_table(name: str, table: dict, grid: list[list[str]], layout: dict,
                numbers_rel: str, numbers_sha: str) -> str:
    """One table as a standalone booktabs float."""
    columns = table["columns"]
    mechanism = layout["mechanism"]
    lines = _provenance_lines(name, numbers_rel, numbers_sha, table.get("source", ""))
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{table['caption']}}}\\label{{{table['label']}}}")

    if mechanism == "resizebox":
        lines.append("\\resizebox{\\textwidth}{!}{%")
        environment, arguments = "tabular", "{" + _latex_column_spec(layout, columns) + "}"
    elif mechanism == "tabularx":
        environment = "tabularx"
        arguments = ("{\\textwidth}{" + _latex_column_spec(layout, columns) + "}")
    else:
        environment, arguments = "tabular", "{" + _latex_column_spec(layout, columns) + "}"

    lines.append(f"\\begin{{{environment}}}{arguments}")
    lines.append("\\toprule")
    lines.append(" & ".join(column["header"] for column in columns) + " \\\\")
    lines.append("\\midrule")
    for line in grid:
        lines.append(" & ".join(latex_text(cell) for cell in line) + " \\\\")
    lines.append("\\bottomrule")
    lines.append(f"\\end{{{environment}}}")
    if mechanism == "resizebox":
        lines.append("}")
    note = table.get("note")
    if note:
        lines.append("\\par\\smallskip")
        lines.append("{\\footnotesize\\textit{Note.}")
        for source_line in wrapped_note(latex_text(note)).splitlines():
            lines.append(f"  {source_line}")
        lines.append("\\par}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def markdown_table(name: str, table: dict, grid: list[list[str]], layout: dict,
                   numbers_rel: str, numbers_sha: str) -> str:
    """One table as a GitHub-flavoured Markdown fragment."""
    columns = table["columns"]
    marks = {"l": ":---", "r": "---:", "c": ":---:"}
    lines = [
        "<!--",
        "  Rendered by scripts/iter11_paper_tables.py -- do not edit.",
        f"  Table {name!r}.",
        f"  Every cell is derived from {numbers_rel}",
        f"    sha256 {numbers_sha}",
        "  whose values are read out of the filed artifact",
        f"  {table.get('source', '')!r} by scripts/iter11_paper_numbers.py.",
        "  Re-render with: python scripts/iter11_paper_tables.py --write",
        "-->",
        "",
        f"### {table['caption']}",
        "",
        "| " + " | ".join(column["header"] for column in columns) + " |",
        "| " + " | ".join(marks[layout["alignment"][column["key"]]]
                          for column in columns) + " |",
    ]
    for line in grid:
        lines.append("| " + " | ".join(markdown_text(cell) for cell in line) + " |")
    lines.append("")
    note = table.get("note")
    if note:
        for index, source_line in enumerate(
                wrapped_note(markdown_text(note)).splitlines()):
            lines.append(source_line if index else f"*Note.* {source_line}")
        lines.append("")
    lines.append(f"`{table['label']}` -- source `{table.get('source', '')}`, "
                 f"rendered from `{numbers_rel}`.")
    return "\n".join(lines) + "\n"


def all_tables_tex(names: list[str], prefix: str) -> str:
    r"""One file that \inputs every table.

    Alphabetical, because the renderer does not know where the paper will cite
    each table and an order derived from the names is the only order that is a
    function of the evidence. main.tex cites tables by \label in whatever order
    the prose wants; this file exists so that a draft can pull all of them in
    while that order is still being written.

    ``prefix`` is the output directory's own name, because LaTeX resolves
    ``\input`` against the directory the MAIN document is compiled from and not
    against the directory the including file sits in. This file sits in
    ``<prefix>/tex/`` and main.tex sits one level above ``<prefix>/``, so the
    paths have to carry the prefix: emitting ``tex/blinding`` here would send
    LaTeX looking for ``paper/tex/blinding.tex``, which does not exist, and the
    failure would name a path this stage never wrote.
    """
    lines = [
        "% Every Iteration 11 table, rendered by scripts/iter11_paper_tables.py.",
        "% Alphabetical by table name: the citation order is the paper's decision",
        "% and not the renderer's, so this file does not make one.",
        f"% Paths carry the {prefix}/ prefix because LaTeX resolves \\input",
        "% against the directory main.tex is compiled from, not against the",
        "% directory this file sits in.",
    ]
    for name in names:
        lines.append(f"\\input{{{prefix}/{TEX_SUBDIR}/{name}}}")
    lines.append("")
    lines.append(f"% {len(names)} tables, rendered from "
                 f"paper/numbers/iteration_11_paper_numbers.json.")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Rendering the whole document
# ---------------------------------------------------------------------------

def render_document(doc: dict, out_dir: Path,
                    numbers_rel: str, numbers_sha: str) -> tuple[dict, dict, list]:
    """Every output file this stage produces, and the findings.

    Total: it returns whatever it could render together with everything that was
    wrong, rather than raising at the first bad cell. A stage that stops early
    reports one problem and hides the rest, and the fix for that is not a
    try/except but a function that finishes.
    """
    findings: list[str] = []
    tables = doc.get("tables")
    if not isinstance(tables, dict) or not tables:
        fatal(f"{numbers_rel} files no tables, so there is nothing to render; this "
              f"stage holds no numbers of its own and cannot supply any", 1)

    unknown = declared_but_not_renderable()
    if unknown:
        findings.append(
            f"the numbers stage declares the rendering specs {', '.join(unknown)}, "
            f"which this stage cannot render; a column filed under one of them "
            f"would be printed by guessing, and two renderers guess differently")

    files: dict[str, str] = {}
    per_table: dict[str, dict] = {}
    for name in sorted(tables):
        table = tables[name]
        unprintable = unprintable_reasons(name, table)
        if unprintable:
            findings.extend(unprintable)
            continue

        # A caption is authored LaTeX and a note is literal prose, and each gets
        # the check that belongs to it: the caption is asked whether it will
        # compile as the mathematics it carries, the note whether it carries any
        # mathematics that escaping would destroy. Asking either question of the
        # other field reports the field's own nature as a defect.
        findings.extend(check_authored_latex(table["caption"], name, "caption"))
        for column in table["columns"]:
            header = column["header"]
            where = f"{name} column {column['key']!r}"
            findings.extend(check_authored_latex(header, where, "header"))
            if column["format"] not in numbers.FORMATS:
                findings.append(f"{where}: declares the rendering spec "
                                f"{column['format']!r}, which is not one the numbers "
                                f"stage declares, so nothing upstream could have "
                                f"filed a value for it under that name")
        note = table.get("note")
        if note:
            findings.extend(check_escaped_field(note, name, "note"))
            findings.extend(check_markdown_literal(note, name, "note"))

        grid, cell_findings, metadata = render_grid(name, table)
        findings.extend(cell_findings)
        for index, row in enumerate(grid):
            for position, cell in enumerate(row):
                where = (f"{name} row {index} column "
                         f"{table['columns'][position]['key']!r}")
                findings.extend(check_escaped_field(cell, where, "cell"))
                findings.extend(check_markdown_literal(cell, where, "cell"))
        layout = column_layout(name, table, grid)
        per_table[name] = dict(metadata, label=table["label"],
                               source=table.get("source", ""), layout=layout,
                               tex=f"{_rel(out_dir)}/{TEX_SUBDIR}/{name}.tex",
                               markdown=f"{_rel(out_dir)}/{MARKDOWN_SUBDIR}/{name}.md")
        files[f"{TEX_SUBDIR}/{name}.tex"] = latex_table(
            name, table, grid, layout, numbers_rel, numbers_sha)
        files[f"{MARKDOWN_SUBDIR}/{name}.md"] = markdown_table(
            name, table, grid, layout, numbers_rel, numbers_sha)

    files[f"{TEX_SUBDIR}/{ALL_TEX_NAME}"] = all_tables_tex(sorted(tables),
                                                           out_dir.name)
    return files, per_table, findings


def the_renderings_agree(doc: dict, numbers_rel: str, numbers_sha: str) -> list[str]:
    """That the two encodings of every cell are the same cell.

    This is the requirement the numbers file states about this stage -- "two
    renderings of one table cannot disagree" -- turned into a check. It is not
    structural: the LaTeX is unescaped and compared with the Markdown, so an
    encoder that rounded differently, dropped a sign or invented a digit would
    fail here even though both files were produced by one pass over one grid.
    """
    issues: list[str] = []
    tables = doc.get("tables") or {}
    for name in sorted(tables):
        table = tables[name]
        grid, _, _ = render_grid(name, table)
        layout = column_layout(name, table, grid)
        tex = latex_table(name, table, grid, layout, numbers_rel, numbers_sha)
        markdown = markdown_table(name, table, grid, layout, numbers_rel, numbers_sha)
        tex_rows = _latex_body_rows(tex)
        markdown_rows = _markdown_body_rows(markdown)
        if len(tex_rows) != len(grid) or len(markdown_rows) != len(grid):
            issues.append(f"{name}: rendered {len(tex_rows)} LaTeX body rows and "
                          f"{len(markdown_rows)} Markdown rows from {len(grid)} "
                          f"filed rows, so a row was lost or split in one of the "
                          f"two formats")
            continue
        for index, canonical in enumerate(grid):
            tex_cells = _split_latex_row(tex_rows[index])
            markdown_cells = [markdown_unescape(cell)
                              for cell in _split_markdown_row(markdown_rows[index])]
            if len(tex_cells) != len(canonical) or len(markdown_cells) != len(canonical):
                issues.append(f"{name} row {index}: {len(tex_cells)} LaTeX cells and "
                              f"{len(markdown_cells)} Markdown cells from "
                              f"{len(canonical)} columns")
                continue
            for position, expected in enumerate(canonical):
                key = table["columns"][position]["key"]
                if latex_unescape(tex_cells[position]) != expected:
                    issues.append(f"{name} row {index} column {key!r}: the LaTeX cell "
                                  f"unescapes to "
                                  f"{latex_unescape(tex_cells[position])!r} and the "
                                  f"filed cell renders to {expected!r}, so the two "
                                  f"print different numbers")
                if markdown_cells[position] != expected:
                    issues.append(f"{name} row {index} column {key!r}: the Markdown "
                                  f"cell is {markdown_cells[position]!r} and the "
                                  f"filed cell renders to {expected!r}, so the two "
                                  f"print different numbers")
        note = table.get("note")
        if note:
            # The note is printed by both formats as well, so it is part of what
            # has to agree. Compared with whitespace normalised, because the note
            # is wrapped into source lines and a line break is a space in both
            # formats -- which is precisely what makes the wrap safe to do, and
            # what this comparison therefore also tests.
            expected = " ".join(note.split())
            from_tex = " ".join(latex_unescape(_latex_note(tex)).split())
            from_markdown = " ".join(
                markdown_unescape(_markdown_note(markdown)).split())
            if from_tex != expected:
                issues.append(f"{name}: the LaTeX note prints {from_tex[:120]!r} and "
                              f"the filed note is {expected[:120]!r}")
            if from_markdown != expected:
                issues.append(f"{name}: the Markdown note prints "
                              f"{from_markdown[:120]!r} and the filed note is "
                              f"{expected[:120]!r}")
    return issues


def _latex_note(tex: str) -> str:
    """The note as it was emitted into the LaTeX float."""
    marker = "{\\footnotesize\\textit{Note.}"
    if marker not in tex:
        return ""
    return tex.split(marker, 1)[1].rsplit("\\par}", 1)[0]


def _markdown_note(markdown: str) -> str:
    """The note as it was emitted into the Markdown fragment."""
    lines = markdown.splitlines()
    start = next((index for index, line in enumerate(lines)
                  if line.startswith("*Note.* ")), None)
    if start is None:
        return ""
    prefix = "*Note.* "
    collected = [lines[start][len(prefix):]]
    for line in lines[start + 1:]:
        if not line.strip():
            break
        collected.append(line)
    return " ".join(collected)


def _latex_body_rows(tex: str) -> list[str]:
    """The rows between ``\\midrule`` and ``\\bottomrule``."""
    if "\\midrule" not in tex or "\\bottomrule" not in tex:
        return []
    body = tex.split("\\midrule", 1)[1].split("\\bottomrule", 1)[0]
    return [line for line in body.strip().splitlines() if line.strip()]


def _split_latex_row(line: str) -> list[str]:
    """Split a LaTeX row on the ``&`` that are not escaped."""
    body = line.strip()
    if body.endswith("\\\\"):
        body = body[:-2]
    cells: list[str] = []
    current = ""
    index = 0
    while index < len(body):
        character = body[index]
        if character == "\\" and index + 1 < len(body):
            current += body[index:index + 2]
            index += 2
            continue
        if character == "&":
            cells.append(current.strip())
            current = ""
            index += 1
            continue
        current += character
        index += 1
    cells.append(current.strip())
    return cells


def _markdown_body_rows(markdown: str) -> list[str]:
    lines = [line for line in markdown.splitlines() if line.startswith("| ")]
    # The first two are the header row and the alignment row.
    return lines[2:] if len(lines) >= 2 else []


def _split_markdown_row(line: str) -> list[str]:
    body = line.strip()
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|"):
        body = body[:-1]
    cells: list[str] = []
    current = ""
    index = 0
    while index < len(body):
        character = body[index]
        if character == "\\" and index + 1 < len(body):
            current += body[index:index + 2]
            index += 2
            continue
        if character == "|":
            cells.append(current.strip())
            current = ""
            index += 1
            continue
        current += character
        index += 1
    cells.append(current.strip())
    return cells


# ---------------------------------------------------------------------------
# What the numbers file actually contains, measured
# ---------------------------------------------------------------------------

def cells_of(doc: dict):
    """Every cell of every table, with the spec that governs it.

    One walk, so that every measurement this stage files about the numbers -- how
    many cells, how many are governed by a row-level spec, how many are absent,
    which characters need escaping -- is a count of the same enumeration rather
    than four separate passes that could disagree with each other.
    """
    for name in sorted(doc.get("tables") or {}):
        table = doc["tables"][name]
        if not is_printable(table):
            # render_document has already filed this as a finding. Enumerating the
            # cells of a table that cannot be printed would turn that finding into
            # an AttributeError naming no table, in every census that walks this.
            continue
        columns = table.get("columns") or []
        for index, row in enumerate(table.get("rows") or []):
            if not isinstance(row, dict):
                continue
            for position, column in enumerate(columns):
                key = column.get("key")
                spec, origin = effective_spec(table, row, key)
                yield (name, key, index, position, spec, origin, row.get(key))


def _spec_census(doc: dict) -> dict:
    used: dict[str, int] = {}
    row_governed = 0
    total = 0
    absent = 0
    for _, _, _, _, spec, origin, value in cells_of(doc):
        used[spec] = used.get(spec, 0) + 1
        total += 1
        if origin == "row":
            row_governed += 1
        if value is None:
            absent += 1
    return {
        "n_cells": total,
        "n_cells_governed_by_a_row_level_spec": row_governed,
        "n_absent_cells": absent,
        "formats_the_tables_use": dict(sorted(used.items())),
        "formats_declared_but_unused_by_any_table": sorted(
            set(numbers.FORMATS) - set(used)),
    }


def _sha8_census(doc: dict) -> dict:
    lengths: dict[str, int] = {}
    count = 0
    for _, _, _, _, spec, _, value in cells_of(doc):
        if spec != "sha8" or not isinstance(value, str):
            continue
        count += 1
        key = str(len(value))
        lengths[key] = lengths.get(key, 0) + 1
    return {
        "n_chars_printed": SHA8_CHARS,
        "n_cells": count,
        "filed_lengths": dict(sorted(lengths.items(), key=lambda kv: int(kv[0]))),
        "what_it_prints": (
            f"the first {SHA8_CHARS} characters of the filed digest. The whole "
            f"digest stays in the numbers file and in the artifact it came from; "
            f"a paper table carries a prefix a reader can match against the "
            f"artifact, not the 64 characters that would not fit in the column"),
    }


def _pct1_confirmation(doc: dict) -> list[dict]:
    """Which filed columns show that ``pct1`` is a fraction of a whole.

    The renderer multiplies by 100. That is a hundredfold error if the filed
    values are already percentages, and no reader of the printed table could tell.
    So the reading is corroborated against the data rather than taken from the
    spec's name: a column of shares of one whole sums to 1.
    """
    confirmed = []
    columns: dict[tuple[str, str], list] = {}
    for name, key, _, _, spec, _, value in cells_of(doc):
        if spec != "pct1":
            continue
        columns.setdefault((name, key), []).append(value)
    for (name, key), values in sorted(columns.items()):
        numeric = [value for value in values if _is_number(value)]
        total = float(sum(numeric))
        confirmed.append({
            "table": name,
            "column": key,
            "n_values": len(values),
            "n_numeric": len(numeric),
            "sum": total,
            "sums_to_one_whole": math.isclose(total, 1.0, abs_tol=1e-9),
            "printed_as": "a percentage, so a filed 0.75 prints as 75.0%",
        })
    return confirmed


def _pct1_block(doc: dict) -> dict:
    """The ``pct1`` reading and the measurement that corroborates it."""
    confirmed = _pct1_confirmation(doc)
    return {
        "what_the_renderer_does": "multiplies the filed value by 100 and prints "
                                  "one decimal",
        "measured_over": confirmed,
        "columns_that_sum_to_one_whole": sorted(
            f"{row['table']}.{row['column']}" for row in confirmed
            if row["sums_to_one_whole"]),
        "why_a_column_may_not_sum_to_one": (
            "a column of shares of DIFFERENT denominators -- five separate 'share "
            "of the items where this happened' figures, say -- need not sum to a "
            "whole, and one that does not is not a defect. What the sum establishes "
            "is the unit: a column that does sum to 1 is a partition, and a "
            "partition is filed as fractions, which is what licenses multiplying by "
            "100 at all. Every value is separately required to lie in [0, 1], "
            "which is the check that catches a percentage filed as a fraction"),
        "why_it_is_not_read_off_the_name": (
            "a spec called pct1 does not say whether 0.75 means 0.75% or 75%, and "
            "getting it backwards is a hundredfold error no reader of the printed "
            "table could detect, so the reading is corroborated against the filed "
            "values rather than taken from the name of the spec"),
    }


def _rounding_block(doc: dict) -> dict:
    """Which rounding this stage applies, and whether the mode is observable here.

    Python's format rounds half to even on a value whose decimal expansion is
    exactly representable in binary -- ``f"{0.125:.2f}"`` is ``0.12`` and not
    ``0.13`` -- and to the nearest representable value otherwise. Neither is a
    choice this stage makes; both are what the format does, and this stage does not
    substitute a convention of its own because a renderer with its own rounding is
    a renderer that can disagree with the other format.

    What this stage CAN establish is whether any filed value sits exactly half way
    at the precision it prints, which is the only case where half-to-even and
    half-up differ. That is measured rather than asserted, because a reader
    hand-checking a printed cell is doing exactly this arithmetic and an empty list
    means they get the table's digits whatever convention they assume.
    """
    on_a_boundary: list[str] = []
    rounded = 0
    for name, key, _, _, spec, _, value in cells_of(doc):
        base = split_spec(spec)[1]
        digits = ROUNDING_DIGITS.get(base)
        if digits is None or not _is_number(value):
            continue
        rounded += 1
        exact = Decimal(repr(float(value)))
        if base == "pct1":
            exact *= Decimal(100)
        exact *= Decimal(10) ** digits
        if abs(exact) % Decimal(1) == Decimal("0.5"):
            on_a_boundary.append(f"{name}.{key} = {value!r} prints at {digits} "
                                 f"decimals and is exactly half way")
    return {
        "what_it_is": (
            "Python's format rounding, at the precision the spec names, applied in "
            "the one function both renderings call. This stage does not substitute "
            "a rounding convention of its own, because a renderer with its own "
            "rounding is a renderer that can disagree with the other format"),
        "digits_per_spec": dict(sorted(ROUNDING_DIGITS.items())),
        "n_cells_rounded": rounded,
        "cells_exactly_half_way_at_the_precision_they_print": on_a_boundary,
        "why_that_list_is_measured": (
            "a value exactly half way is the ONLY case where round-half-to-even and "
            "round-half-up give different digits, so while this list is empty a "
            "reader hand-checking any printed cell gets the table's digits whatever "
            "convention they assume. It is measured rather than asserted because a "
            "new artifact could put a value on a boundary, and at that point the "
            "convention stops being an implementation detail and becomes a claim "
            "the paper has to make"),
    }


def _hostile_census(doc: dict) -> dict:
    """How much escaping the literal fields actually need, measured.

    Filed because the escaping asymmetry is the one place this stage can silently
    ruin the paper: escape an authored header and the mathematics prints as
    punctuation, pass a literal cell through and an underscore stops the compile
    or an '&' moves a column boundary without any error at all.

    Counted over the RENDERED cells and not over the filed values, because the two
    are not the same text: a share filed as 0.006688963210702341 carries no
    hostile character at all and renders to ``0.7%``, whose percent sign is the one
    character in the whole document that LaTeX would read as the start of a
    comment. A census of the filed values would report that this stage needs no
    percent escaping while it escapes ten of them.
    """
    counts: dict[str, int] = {}
    authored = 0
    notes = 0
    cells = 0
    emphasis_risks = 0
    for name in sorted(doc.get("tables") or {}):
        table = doc["tables"][name]
        if not is_printable(table):
            continue  # already a finding; a census does not get to crash on it
        authored += 1  # the caption
        for column in table.get("columns") or []:
            authored += 1  # each header
        note = table.get("note")
        if isinstance(note, str):
            notes += 1
            emphasis_risks += len(MARKDOWN_EMPHASIS_RISK.findall(note))
            for character in note:
                if character in LATEX_ESCAPES:
                    counts[character] = counts.get(character, 0) + 1
        grid, _, _ = render_grid(name, table)
        for line in grid:
            for cell in line:
                cells += 1
                emphasis_risks += len(MARKDOWN_EMPHASIS_RISK.findall(cell))
                for character in cell:
                    if character in LATEX_ESCAPES:
                        counts[character] = counts.get(character, 0) + 1
    return {
        "n_literal_strings_escaped": cells + notes,
        "n_of_them_cells": cells,
        "n_of_them_notes": notes,
        "n_authored_latex_fields_passed_through": authored,
        "hostile_characters_found_in_literal_text": dict(sorted(counts.items())),
        "n_markdown_emphasis_risks": emphasis_risks,
        "what_that_count_is": (
            "underscores that are not inside a word, which Markdown could read as "
            "the start of emphasis. Zero is why Markdown is escaped for '|' only "
            "and not for underscores: every underscore in this document sits "
            "inside an identifier, where CommonMark leaves it alone. The count is "
            "filed so that the day one does not is a number and not a surprise"),
        "n_dollars_or_backslashes_in_literal_text":
            counts.get("$", 0) + counts.get("\\", 0),        "what_that_last_number_has_to_be": (
            "zero. A literal field carrying '$' or '\\' is carrying authored "
            "mathematics, and escaping it would print the mathematics as "
            "punctuation; this stage refuses instead of choosing"),
    }


# ---------------------------------------------------------------------------
# The filed document
# ---------------------------------------------------------------------------

def render_everything(numbers_path: Path | None = None,
                      out_dir: Path | None = None) -> tuple[dict, dict, list]:
    """The renderings document, every file it describes, and the findings."""
    numbers_path = numbers_path or NUMBERS_PATH
    out_dir = out_dir or OUT_DIR
    doc = load_numbers(numbers_path)
    numbers_rel = _rel(numbers_path)
    numbers_sha = numbers.sha256_file(numbers_path)
    if numbers_sha is None:
        fatal(f"{numbers_rel} disappeared between being read and being hashed", 2)

    files, per_table, findings = render_document(doc, out_dir, numbers_rel, numbers_sha)

    # Only tables that are whole enough to render. One missing a caption is already
    # a finding; asking the agreement check to render it anyway would turn a
    # reported problem into a crash.
    complete = {name: table for name, table in (doc.get("tables") or {}).items()
                if is_printable(table)}
    findings.extend(the_renderings_agree({"tables": complete}, numbers_rel, numbers_sha))

    census = _spec_census(doc)
    mechanisms: dict[str, int] = {}
    packages: set[str] = set()
    for block in per_table.values():
        mechanism = block["layout"]["mechanism"]
        mechanisms[mechanism] = mechanisms.get(mechanism, 0) + 1
        packages.update(block["layout"]["latex_packages_required"])

    document = {
        "kind": KIND,
        "produced_by": "scripts/iter11_paper_tables.py",
        "question": QUESTION,
        "inputs": {
            "path": numbers_rel,
            "sha256": numbers_sha,
            "n_tables": len(per_table),
            "why_bound_by_hash": (
                "every cell printed by this stage is a function of this one file "
                "and of nothing else, so the hash is what makes 'rendered from the "
                "filed numbers' a checkable statement rather than a claim"),
            "why_nothing_else_is_an_input": (
                "this stage reads no artifact of the analysis. The numbers stage "
                "already read twenty-two of them and re-derives itself against "
                "them; a renderer that went back to the artifacts would be a "
                "second place the paper's numbers come from, and the two could "
                "differ"),
        },
        "the_rendering_contract": {
            "formats_the_numbers_stage_declares": list(numbers.FORMATS),
            "formats_this_stage_can_render": sorted(CAPABLE_FORMATS),
            "formats_declared_but_not_renderable": declared_but_not_renderable(),
            "formats_renderable_but_not_declared": sorted(
                CAPABLE_FORMATS - set(numbers.FORMATS)),
            "n_cells": census["n_cells"],
            "formats_the_tables_use": census["formats_the_tables_use"],
            "formats_declared_but_unused_by_any_table":
                census["formats_declared_but_unused_by_any_table"],
            "why_the_two_vocabularies_are_both_filed": (
                "the declared list is compared with what this stage can render on "
                "every run, and 'declared but not renderable' has to be empty. The "
                "other difference is filed rather than treated as a defect: the "
                "renderer can print an optional form of any spec it can print at "
                "all, and four of those are declared by nobody. Stating that "
                "plainly is how a reader knows the wider list is generality and "
                "not a vocabulary that has drifted away from the numbers file"),
            "n_cells_governed_by_a_row_level_spec":
                census["n_cells_governed_by_a_row_level_spec"],
            "what_a_row_level_spec_is": (
                f"a claim table files one spec per row under {ROW_FORMAT_KEY!r} "
                f"because its {ROW_FORMAT_GOVERNS!r} column mixes counts, hashes, "
                f"booleans and prose, and one spec for all of them would either "
                f"round a hash or print a count as a decimal. The row's spec "
                f"governs that cell and the column's spec governs the rest"),
            "absent_value_mark": ABSENT,
            "n_absent_cells": census["n_absent_cells"],
            "why_absent_is_a_mark_and_not_an_empty_cell": (
                "an empty cell is indistinguishable from a cell the renderer "
                "dropped, so a reader cannot tell a measurement that was not made "
                "from a bug in the print"),
            "pct1_is_a_fraction": _pct1_block(doc),
            "sha8_prints_a_prefix": _sha8_census(doc),
            "rounding": _rounding_block(doc),
            "note_source_width_chars": NOTE_SOURCE_WIDTH,
            "why_notes_are_wrapped": (
                "the longest filed note is one string of 2343 characters, which "
                "would be a single source line of 2396 in the rendered table. It "
                "compiles either way, but this repository re-derives and compares "
                "its artifacts and one long line turns a changed word into a "
                "whole-line diff. A line break is a space to LaTeX and a soft break "
                "is a space to Markdown, so the wrap cannot change the print -- and "
                "--verify compares the note of both renderings with the filed note "
                "after normalising whitespace, which is that claim tested rather "
                "than asserted. A note whose whitespace is not already single "
                "spaces is left on one line rather than being normalised"),
            "the_minus_sign_is_ascii": (
                "a negative number prints with the ASCII hyphen-minus the artifact "
                "was written with, not with a typographic minus in math mode. Math "
                "mode would make the LaTeX rendering differ from the Markdown one "
                "by more than escaping, and the invariant that the two renderings "
                "of a table cannot disagree is worth more than the glyph"),
            "what_is_escaped_and_what_is_not": dict(
                _hostile_census(doc),
                authored_latex_fields=list(AUTHORED_LATEX_FIELDS),
                escaped_fields=list(ESCAPED_FIELDS),
                why_they_differ=(
                    "a header or a caption is written by the numbers stage as "
                    "LaTeX and carries mathematics; a cell, a claim, an artifact "
                    "name or a note is literal text and carries identifiers with "
                    "underscores in them. Escaping the first destroys the "
                    "mathematics and not escaping the second breaks the table, so "
                    "the split is measured on every run rather than assumed"),
            ),
            "layout_is_measured": {
                "text_block_budget_chars": CHARS_PER_TEXTWIDTH,
                "tabcolsep_chars_per_column": TABCOLSEP_CHARS,
                "wrap_a_column_above_chars": WRAP_ABOVE_CHARS,
                "wrap_share_of_budget": WRAP_SHARE_OF_BUDGET,
                "wrap_weight_power": WRAP_WEIGHT_POWER,
                "mechanisms": list(MECHANISMS),
                "mechanisms_used": dict(sorted(mechanisms.items())),
                "how_a_width_is_measured": (
                    "the longer of a column's header and its widest rendered cell, "
                    "with LaTeX macros discounted, because \\kappa costs one glyph "
                    "and six characters and counting source characters would push "
                    "a table into a resizebox it did not need"),
                "why_the_budget_is_an_estimate": (
                    "compiling the table and measuring the box would make this "
                    "document depend on a TeX installation, and the figures were "
                    "kept out of the numbers file for exactly that reason. The "
                    "estimate decides which of three standard mechanisms to use, "
                    "and the measurements behind each choice are filed per table"),
            },
            "latex_packages_required": sorted(packages),
            "what_the_input_paths_are_relative_to": (
                f"the directory above the output directory, which is "
                f"{_rel(OUT_DIR.parent)} for the filed rendering and is where "
                f"main.tex has to sit. {ALL_TEX_NAME} therefore emits "
                f"\\input{{{OUT_DIR.name}/{TEX_SUBDIR}/<table>}} rather than "
                f"\\input{{{TEX_SUBDIR}/<table>}}, because LaTeX resolves \\input "
                f"against the directory the main document is compiled from and not "
                f"against the directory the including file sits in"),
            "why_the_packages_are_filed": (
                "a missing \\usepackage is a compilation failure that points at "
                "none of these files, so the preamble the paper needs is stated "
                "here beside the mechanism that needs it"),
        },
        "tables": per_table,
        "totals": {
            "n_tables": len(per_table),
            "n_columns": sum(block["n_columns"] for block in per_table.values()),
            "n_rows": sum(block["n_rows"] for block in per_table.values()),
            "n_cells": sum(block["n_cells"] for block in per_table.values()),
            "n_rendered_files": len(files),
            "n_tex_files": sum(1 for rel in files if rel.endswith(".tex")),
            "n_markdown_files": sum(1 for rel in files if rel.endswith(".md")),
            "output_paths": sorted(files),
        },
        "findings": sorted(findings),
        "what_a_finding_here_means": (
            "a cell that contradicts the spec it was filed under, a spec nobody "
            "implements, a rendering whose two formats disagree, or a field whose "
            "LaTeX will not compile. Empty is the only acceptable value, and "
            "--write refuses to write a document that has any"),
        "what_this_file_is_not": {
            "not_a_source_of_numbers": (
                "every value printed by this stage was read out of the numbers "
                "file. There is one function that turns a filed value into a "
                "printed cell and both formats call it, so rounding is decided "
                "once and this file holds no number the numbers file does not"),
            "not_a_second_hash_of_the_outputs": (
                "the .tex and .md files are not hashed here. The closeout manifest "
                "binds every file under paper/ by hash already, so recording those "
                "hashes a second time would be a second place for them to go stale; "
                "--verify compares the bytes on disk with a fresh rendering "
                "instead, which is a stronger check than comparing two hashes of "
                "one file"),
            "not_the_paper": (
                "no prose, no citation order and no figure is here. main.tex cites "
                "these tables by \\label, and the order they appear in the paper is "
                "the paper's decision rather than the renderer's, which is why "
                f"{ALL_TEX_NAME} is alphabetical"),
            "not_environment_sensitive": (
                "no clock, no commit, no host and no library version enters this "
                "document or any file it writes, so a rendering re-derives byte for "
                "byte anywhere. The figures are specified as data upstream for the "
                "same reason"),
        },
    }
    return document, files, findings


def build(numbers_path: Path | None = None, out_dir: Path | None = None) -> dict:
    """The renderings document, and nothing but it."""
    document, _, _ = render_everything(numbers_path, out_dir)
    return document


BANNED_KEYS = frozenset({
    "generated_at", "timestamp", "git_commit", "git_dirty", "code_commit",
    "python", "host", "generated_from_commit", "code_tree_dirty",
    "untracked_code_paths",
})


def _purity_issues(node: object, path: str) -> list[str]:
    """That nothing in the document records when or where it was written."""
    issues = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key in BANNED_KEYS:
                issues.append(f"{path}/{key} records machine state, so this document "
                              f"would re-derive differently on another machine or "
                              f"after a commit and could never be compared with "
                              f"what is filed")
            issues.extend(_purity_issues(value, f"{path}/{key}"))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            issues.extend(_purity_issues(value, f"{path}[{index}]"))
    return issues


def check_the_renderings(doc: dict) -> list[str]:
    """Internal agreement, and the document against this module's constants.

    Every comparison here takes its expected side from something outside the
    document: a module constant, the numbers stage's own vocabulary, or another
    field of the same document that was computed a different way. A field
    compared with a re-derivation of itself would pass whatever it said.
    """
    issues = _purity_issues(doc, "")
    if doc.get("kind") != KIND:
        issues.append(f"kind is {doc.get('kind')!r} and this stage writes {KIND!r}")
    if doc.get("produced_by") != "scripts/iter11_paper_tables.py":
        issues.append(f"produced_by is {doc.get('produced_by')!r}")
    if doc.get("findings"):
        issues.extend(f"filed finding: {finding}" for finding in doc["findings"])

    contract = doc.get("the_rendering_contract") or {}
    if contract.get("formats_this_stage_can_render") != sorted(CAPABLE_FORMATS):
        issues.append(
            f"the_rendering_contract.formats_this_stage_can_render is "
            f"{contract.get('formats_this_stage_can_render')!r} and this stage can "
            f"render {sorted(CAPABLE_FORMATS)!r}: the filed vocabulary is not the "
            f"one the code holds, so a reader trusting the document would trust a "
            f"list nobody is running")
    if contract.get("formats_declared_but_not_renderable") != []:
        issues.append(
            f"the_rendering_contract.formats_declared_but_not_renderable is "
            f"{contract.get('formats_declared_but_not_renderable')!r}: the numbers "
            f"stage declares specs this stage cannot print, and a column filed "
            f"under one of them would be rendered by guessing")
    if contract.get("formats_the_numbers_stage_declares") != list(numbers.FORMATS):
        issues.append(
            f"the_rendering_contract.formats_the_numbers_stage_declares is "
            f"{contract.get('formats_the_numbers_stage_declares')!r} and the "
            f"numbers stage declares {list(numbers.FORMATS)!r}")
    if contract.get("absent_value_mark") != ABSENT:
        issues.append(f"the_rendering_contract.absent_value_mark is "
                      f"{contract.get('absent_value_mark')!r} and this stage prints "
                      f"{ABSENT!r} for an absent value")
    prefix = contract.get("sha8_prints_a_prefix") or {}
    if prefix.get("n_chars_printed") != SHA8_CHARS:
        issues.append(f"sha8_prints_a_prefix.n_chars_printed is "
                      f"{prefix.get('n_chars_printed')!r} and this stage prints "
                      f"{SHA8_CHARS}")
    rounding = contract.get("rounding") or {}
    if rounding.get("digits_per_spec") != dict(sorted(ROUNDING_DIGITS.items())):
        issues.append(f"rounding.digits_per_spec is "
                      f"{rounding.get('digits_per_spec')!r} and this stage rounds at "
                      f"{dict(sorted(ROUNDING_DIGITS.items()))!r}")
    escaping = contract.get("what_is_escaped_and_what_is_not") or {}
    if escaping.get("authored_latex_fields") != list(AUTHORED_LATEX_FIELDS):
        issues.append("what_is_escaped_and_what_is_not.authored_latex_fields is "
                      f"{escaping.get('authored_latex_fields')!r} and this stage "
                      f"passes through {list(AUTHORED_LATEX_FIELDS)!r}")
    if escaping.get("escaped_fields") != list(ESCAPED_FIELDS):
        issues.append("what_is_escaped_and_what_is_not.escaped_fields is "
                      f"{escaping.get('escaped_fields')!r} and this stage escapes "
                      f"{list(ESCAPED_FIELDS)!r}")
    if escaping.get("n_dollars_or_backslashes_in_literal_text") != 0:
        issues.append(
            f"{escaping.get('n_dollars_or_backslashes_in_literal_text')} literal "
            f"fields carry '$' or '\\', which this stage would escape and thereby "
            f"turn authored mathematics into punctuation")
    if escaping.get("n_markdown_emphasis_risks") != 0:
        issues.append(
            f"{escaping.get('n_markdown_emphasis_risks')} literal field(s) hold an "
            f"underscore that is not inside a word, which Markdown can read as the "
            f"start of emphasis; this stage escapes Markdown for '|' only, so that "
            f"underscore would print part of an identifier in italics")
    if contract.get("note_source_width_chars") != NOTE_SOURCE_WIDTH:
        issues.append(f"the_rendering_contract.note_source_width_chars is "
                      f"{contract.get('note_source_width_chars')!r} and this stage "
                      f"wraps notes at {NOTE_SOURCE_WIDTH}")
    layout = contract.get("layout_is_measured") or {}
    for field, expected in (("text_block_budget_chars", CHARS_PER_TEXTWIDTH),
                            ("tabcolsep_chars_per_column", TABCOLSEP_CHARS),
                            ("wrap_a_column_above_chars", WRAP_ABOVE_CHARS),
                            ("wrap_share_of_budget", WRAP_SHARE_OF_BUDGET),
                            ("wrap_weight_power", WRAP_WEIGHT_POWER)):
        if layout.get(field) != expected:
            issues.append(f"layout_is_measured.{field} is {layout.get(field)!r} and "
                          f"this stage lays out with {expected!r}")
    if layout.get("mechanisms") != list(MECHANISMS):
        issues.append(f"layout_is_measured.mechanisms is {layout.get('mechanisms')!r} "
                      f"and this stage chooses between {list(MECHANISMS)!r}")

    tables = doc.get("tables") or {}
    totals = doc.get("totals") or {}
    for field, expected in (("n_tables", len(tables)),
                            ("n_columns", sum(b.get("n_columns", 0)
                                              for b in tables.values())),
                            ("n_rows", sum(b.get("n_rows", 0) for b in tables.values())),
                            ("n_cells", sum(b.get("n_cells", 0) for b in tables.values()))):
        if totals.get(field) != expected:
            issues.append(f"totals.{field} is {totals.get(field)!r} and the "
                          f"{len(tables)} filed tables add up to {expected}: a total "
                          f"that is not the sum of its parts is a number somebody "
                          f"typed")

    per_mechanism: dict[str, int] = {}
    for name in sorted(tables):
        block = tables[name]
        block_layout = block.get("layout") or {}
        mechanism = block_layout.get("mechanism")
        if mechanism not in MECHANISMS:
            issues.append(f"{name}: layout.mechanism is {mechanism!r}, which is not "
                          f"one of {', '.join(MECHANISMS)}, so this document "
                          f"describes a table shape the renderer cannot emit")
            continue
        per_mechanism[mechanism] = per_mechanism.get(mechanism, 0) + 1
        if block_layout.get("latex_packages_required") != \
                list(PACKAGE_FOR_MECHANISM[mechanism]):
            issues.append(f"{name}: layout.latex_packages_required is "
                          f"{block_layout.get('latex_packages_required')!r} and "
                          f"{mechanism} needs "
                          f"{list(PACKAGE_FOR_MECHANISM[mechanism])!r}")
        wrapping = block_layout.get("wrapping_columns") or []
        unbreakable = block_layout.get("unbreakable_columns") or []
        weights = block_layout.get("wrap_weights")
        widths_filed = block_layout.get("column_widths_chars") or {}
        floor = WRAP_SHARE_OF_BUDGET * CHARS_PER_TEXTWIDTH
        # Re-derived from the widths the document itself filed, for every mechanism
        # and not only the wrapping one: this is the measurement the layout decision
        # is made from, so a document whose filed widths do not produce its filed
        # column lists is describing a table it did not measure.
        expected_over_floor = sorted(key for key, width in widths_filed.items()
                                     if width > floor)
        if sorted(block_layout.get("columns_over_the_share_floor") or []) != \
                expected_over_floor:
            issues.append(
                f"{name}: files columns_over_the_share_floor "
                f"{sorted(block_layout.get('columns_over_the_share_floor') or [])} "
                f"and its own filed widths put {expected_over_floor} over "
                f"{floor:.1f} characters")
        expected_unbreakable = sorted(key for key, width in widths_filed.items()
                                      if width > WRAP_ABOVE_CHARS)
        if sorted(unbreakable) != expected_unbreakable:
            issues.append(f"{name}: files unbreakable_columns {sorted(unbreakable)} "
                          f"and its own filed widths put {expected_unbreakable} over "
                          f"{WRAP_ABOVE_CHARS} characters")
        if mechanism == "tabularx":
            if not unbreakable:
                issues.append(f"{name}: takes the {mechanism} route with no column "
                              f"too wide for a fixed one, which is the route a table "
                              f"takes BECAUSE LaTeX cannot break a cell")
            if not wrapping:
                issues.append(f"{name}: takes the {mechanism} route with no wrapping "
                              f"column, which is the route a table takes BECAUSE a "
                              f"column has to wrap")
            elif sorted(wrapping) != expected_over_floor:
                issues.append(
                    f"{name}: wraps {sorted(wrapping)} but the filed widths select "
                    f"{expected_over_floor}, so the wrapping columns are not the "
                    f"ones the measurements pick out")
            if wrapping and weights is None:
                issues.append(f"{name}: wraps {len(wrapping)} column(s) and files no "
                              f"weights for them")
            elif weights is not None:
                if sorted(weights) != sorted(wrapping):
                    issues.append(f"{name}: weights {sorted(weights)} are not for the "
                                  f"wrapping columns {sorted(wrapping)}")
                elif not math.isclose(sum(weights.values()), float(len(wrapping)),
                                      abs_tol=0.011):
                    issues.append(f"{name}: the wrap weights sum to "
                                  f"{sum(weights.values())} and tabularx needs "
                                  f"\\hsize factors summing to the number of X "
                                  f"columns, {len(wrapping)}")
        else:
            # A scaled table and a table that fits both leave every column at its
            # natural width. Filing wrapping columns for either would say the paper
            # breaks a cell that it in fact shrinks or leaves alone.
            if unbreakable:
                issues.append(f"{name}: takes the {mechanism} route while "
                              f"{', '.join(unbreakable)} hold a cell over "
                              f"{WRAP_ABOVE_CHARS} characters, which LaTeX will not "
                              f"break and which will run out of the text block")
            if wrapping:
                issues.append(f"{name}: takes the {mechanism} route and files "
                              f"{sorted(wrapping)} as wrapping columns; nothing wraps "
                              f"unless the table is a tabularx, so the field says the "
                              f"paper breaks cells it does not break")
            if weights is not None:
                issues.append(f"{name}: takes the {mechanism} route and files wrap "
                              f"weights {weights}, which that mechanism does not use")
        unknown = sorted(set(block.get("formats_used") or {}) - CAPABLE_FORMATS)
        if unknown:
            issues.append(f"{name}: files the specs {', '.join(unknown)}, which this "
                          f"stage cannot render")
        mixed = block_layout.get("mixed_spec_columns") or []
        for key, align in sorted((block_layout.get("alignment") or {}).items()):
            if align not in ("l", "r", "c"):
                issues.append(f"{name}: column {key!r} is aligned {align!r}, which is "
                              f"not a LaTeX column alignment")
            if key in mixed and align != "l":
                issues.append(f"{name}: column {key!r} is filed as holding cells "
                              f"whose specs differ row by row and aligned {align!r}; "
                              f"a mixed column has no single alignment and sits "
                              f"left, otherwise its position depends on which row "
                              f"happened to be filed first")
        for key in mixed:
            if key not in (block_layout.get("alignment") or {}):
                issues.append(f"{name}: files {key!r} as a mixed-spec column and no "
                              f"alignment for it, so it is not a column of this "
                              f"table")
        for field in ("tex", "markdown"):
            path = block.get(field) or ""
            if f"/{name}." not in path:
                issues.append(f"{name}: {field} is {path!r}, which is not named after "
                              f"the table, so a rendered file could not be traced "
                              f"back to the table it prints")
    if layout.get("mechanisms_used") != dict(sorted(per_mechanism.items())):
        issues.append(f"layout_is_measured.mechanisms_used is "
                      f"{layout.get('mechanisms_used')!r} and the "
                      f"{len(tables)} filed tables use {per_mechanism!r}")

    output_paths = totals.get("output_paths") or []
    expected_paths = sorted(
        [f"{TEX_SUBDIR}/{name}.tex" for name in tables]
        + [f"{MARKDOWN_SUBDIR}/{name}.md" for name in tables]
        + [f"{TEX_SUBDIR}/{ALL_TEX_NAME}"])
    if output_paths != expected_paths:
        missing = sorted(set(expected_paths) - set(output_paths))
        extra = sorted(set(output_paths) - set(expected_paths))
        issues.append(f"totals.output_paths does not match the {len(tables)} filed "
                      f"tables: missing {missing}, unexpected {extra}")
    if totals.get("n_rendered_files") != len(expected_paths):
        issues.append(f"totals.n_rendered_files is {totals.get('n_rendered_files')!r} "
                      f"and {len(tables)} tables render to {len(expected_paths)} "
                      f"files")
    if totals.get("n_tex_files") + totals.get("n_markdown_files") != \
            totals.get("n_rendered_files"):
        issues.append(f"totals files {totals.get('n_tex_files')} .tex and "
                      f"{totals.get('n_markdown_files')} .md against "
                      f"{totals.get('n_rendered_files')} rendered files, so a file "
                      f"is counted in one list and not the other")
    if doc.get("inputs", {}).get("n_tables") != len(tables):
        issues.append(f"inputs.n_tables is {doc.get('inputs', {}).get('n_tables')!r} "
                      f"and this document files {len(tables)} tables")
    return issues


def verify(path: Path | None = None, numbers_path: Path | None = None,
           out_dir: Path | None = None) -> tuple[int, str, list[str]]:
    """Re-render everything and compare it with what is on disk.

    Two comparisons, and they catch different things. The document is compared
    with a fresh one, which catches a rendering filed from numbers that have since
    moved. Every file is compared with its fresh bytes, which catches a table
    edited by hand after it was rendered -- the one thing a hash of the document
    alone would not see, because the document does not carry the outputs' hashes.
    """
    out_dir = out_dir or OUT_DIR
    path = path or out_dir / RENDERINGS_NAME
    numbers_path = numbers_path or NUMBERS_PATH
    if not numbers_path.exists():
        return 2, "no_numbers", [
            f"no paper numbers at {_rel(numbers_path)}, so there is nothing to "
            f"render and nothing to compare a rendering with"]
    if not path.exists():
        return 2, "not_filed", [
            f"no table renderings at {_rel(path)}; run "
            f"`python scripts/iter11_paper_tables.py --write` to file them"]
    try:
        filed = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return 1, "unreadable", [
            f"{_rel(path)} is not readable JSON ({exc}); it describes the paper's "
            f"tables and this stage does not repair it"]
    try:
        fresh, files, findings = render_everything(numbers_path, out_dir)
    except SystemExit as exc:
        return (exc.code if isinstance(exc.code, int) else 1), "could_not_render", [
            f"the tables could not be re-rendered from {_rel(numbers_path)}: "
            f"exit {exc.code}"]

    issues = list(findings)
    for key in sorted(set(filed) | set(fresh)):
        if filed.get(key) != fresh.get(key):
            issues.append(
                f"{key}: filed {json.dumps(filed.get(key), sort_keys=True)[:200]}"
                f" != re-rendered {json.dumps(fresh.get(key), sort_keys=True)[:200]}")
    for relative, text in sorted(files.items()):
        target = out_dir / relative
        if not target.exists():
            issues.append(f"{relative}: this stage renders it and it is not on disk, "
                          f"so the paper is missing a table the renderings describe")
            continue
        on_disk = target.read_text(encoding="utf-8")
        if on_disk != text:
            issues.append(
                f"{relative}: the file on disk is {len(on_disk)} bytes and a fresh "
                f"rendering of {numbers.sha256_file(numbers_path)[:12]} is "
                f"{len(text)} bytes, so it was edited after it was rendered or the "
                f"numbers moved underneath it")
    on_disk_files = sorted(
        str(found.relative_to(out_dir))
        for found in out_dir.rglob("*")
        if found.is_file() and found.name != RENDERINGS_NAME
        and found.suffix in (".tex", ".md"))
    for relative in sorted(set(on_disk_files) - set(files)):
        issues.append(f"{relative}: on disk under {_rel(out_dir)} and not a file "
                      f"this stage renders, so it is either a table the numbers file "
                      f"no longer has or one somebody wrote by hand")
    issues.extend(check_the_renderings(filed))
    if issues:
        return 1, "differs", issues
    return 0, "reproduced_exactly", []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify", action="store_true",
                      help="re-render every table and compare with the files on "
                           "disk, writing nothing")
    mode.add_argument("--write", action="store_true",
                      help="write the renderings. Explicit, because it overwrites "
                           "committed files the paper cites")
    parser.add_argument("--numbers", type=Path, default=NUMBERS_PATH,
                        help="the numbers file to render from")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR,
                        help="where the renderings and the tables go")
    args = parser.parse_args(argv)

    if args.verify or not args.write:
        code, conclusion, issues = verify(numbers_path=args.numbers,
                                          out_dir=args.out_dir)
        if code == 2:
            print(f"PAPER TABLES: NOT FILED -- {issues[0]}")
            return 2
        if code == 1:
            print(f"PAPER TABLES: FAIL ({len(issues)} issue(s))")
            for issue in issues:
                print(f"  - {issue}")
            return 1
        filed = json.loads((args.out_dir / RENDERINGS_NAME).read_text(encoding="utf-8"))
        print(f"PAPER TABLES: VERIFIED -- {conclusion.replace('_', ' ')}")
        _print_summary(filed)
        return 0

    # Writing is the moment a number enters the paper, so it is gated on the
    # numbers still being the numbers the evidence produces. --verify does not
    # repeat this: it checks this stage's own contract, and the closeout's --deep
    # runs the numbers verifier beside it.
    code, conclusion, upstream = numbers_re_derive(args.numbers)
    if code != 0:
        print(f"FAIL: refusing to render the paper's tables from numbers that do "
              f"not re-derive ({conclusion.replace('_', ' ')})", file=sys.stderr)
        for issue in upstream:
            print(f"  - {issue}", file=sys.stderr)
        print("  Fix the numbers stage first: rendering would put a number into "
              "print that nothing upstream can vouch for", file=sys.stderr)
        return 1

    document, files, findings = render_everything(args.numbers, args.out_dir)
    problems = findings + check_the_renderings(document)
    if problems:
        print("FAIL: refusing to file renderings that disagree with the numbers "
              "they were rendered from", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    for relative, text in sorted(files.items()):
        target = args.out_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    # Written last: a run interrupted before it leaves orphan tables and no
    # document describing them, which --verify reports as not filed, rather than a
    # document describing files that were never written.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / RENDERINGS_NAME).write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(files)} table files and {_rel(args.out_dir / RENDERINGS_NAME)}")
    _print_summary(document)
    return 0


def _print_summary(doc: dict) -> None:
    """What was rendered, printed so a reader can see the paper's coverage."""
    contract = doc["the_rendering_contract"]
    totals = doc["totals"]
    layout = contract["layout_is_measured"]
    print(f"  numbers                      {doc['inputs']['path']} "
          f"({doc['inputs']['sha256'][:12]})")
    print(f"  tables                       {totals['n_tables']}, "
          f"{totals['n_columns']} columns, {totals['n_rows']} rows, "
          f"{totals['n_cells']} cells")
    print(f"  files                        {totals['n_rendered_files']} "
          f"({totals['n_tex_files']} .tex, {totals['n_markdown_files']} .md)")
    print(f"  specs                        {len(contract['formats_the_numbers_stage_declares'])} "
          f"declared, {len(contract['formats_the_tables_use'])} used, "
          f"{contract['n_cells_governed_by_a_row_level_spec']} cell(s) governed by "
          f"a row-level spec")
    print(f"  absent cells                 {contract['n_absent_cells']}, printed as "
          f"{contract['absent_value_mark']!r}")
    escaping = contract["what_is_escaped_and_what_is_not"]
    print(f"  escaping                     {escaping['n_literal_strings_escaped']} "
          f"literal string(s) escaped, "
          f"{escaping['n_authored_latex_fields_passed_through']} authored LaTeX "
          f"field(s) passed through")
    mechanisms_used = ", ".join(
        f"{key} {value}" for key, value in sorted(layout["mechanisms_used"].items()))
    print(f"  mechanisms                   {mechanisms_used}")
    for name in sorted(doc["tables"]):
        block = doc["tables"][name]
        block_layout = block["layout"]
        print(f"  {block['label']:34s} {block['n_rows']:3d} x "
              f"{block['n_columns']:2d}  {block_layout['mechanism']:14s} "
              f"{block_layout['natural_width_chars']:6.1f} chars")


if __name__ == "__main__":
    raise SystemExit(main())
