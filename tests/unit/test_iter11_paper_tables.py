"""Iteration 11: the paper's tables, rendered from the numbers file and pinned.

The numbers stage filed one document holding every value the paper quotes, with a
rendering spec per column and -- for the claim tables -- a rendering spec per row.
It also filed what it requires of this stage:

    not_rendered: this file holds values and a rendering spec per column. The
    .tex and Markdown renderers consume it and hold no numbers of their own, so
    rounding is decided once and two renderings of one table cannot disagree

That sentence is the contract, and it is two claims rather than one. A renderer
that holds no numbers of its own is a renderer where every printed cell can be
traced to a filed value, so the first half of this file perturbs each filed cell
in turn and requires the print to move with it. Two renderings that cannot
disagree is not a property of having written them in one pass, so the second half
parses the emitted LaTeX and the emitted Markdown back into cells, unescapes the
LaTeX, and requires both to equal the canonical text -- which is the only place
rounding happens.

The rest pins the decisions a renderer has to make that are not rounding, each of
them measured out of the numbers file rather than chosen by eye:

* what is escaped and what is not. Headers and captions are authored LaTeX and
  carry mathematics; cells and notes are literal text and carry identifiers with
  underscores in them. Getting that backwards either prints ``$\\Delta_{TV}$`` as
  punctuation or stops the compile at the first ``_``, and an unescaped ``&`` is
  worse than either because it moves a column boundary without any error at all;
* which cells are governed by a row-level spec, which is what stops a claim table
  printing a count as a decimal or rounding a hash;
* what an absent value prints as, so that a measurement nobody made is not
  indistinguishable from a cell the renderer dropped;
* what the specs MEAN -- that ``pct1`` is a fraction and not a percentage, that
  ``sha8`` prints a prefix of a longer digest, that a p-value must not print as
  zero -- each of which is a hundredfold or a rounding error if read the other way
  and none of which a reader of the printed table could detect;
* the column layout, which is derived from measured widths and filed with the
  measurements that decided it.

CI-safe: every test reads the committed numbers file or synthesises the shape it
has. Nothing renders a figure, calls a model, calls a judge, touches the network,
or needs the media, and nothing needs credentials.
"""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(
        f"{name}_under_test", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


tables = _load_script("iter11_paper_tables")
#: The SAME module object the renderer reads. Loading iter11_paper_numbers a second
#: time under its own name would give these tests a copy whose ``FORMATS`` the
#: renderer never looks at, so monkeypatching it would change nothing and a test
#: that meant to prove the renderer notices a new spec would pass by proving
#: nothing at all.
numbers = tables.numbers

FILED_NUMBERS = tables.NUMBERS_PATH
FILED_RENDERINGS = tables.OUT_DIR / tables.RENDERINGS_NAME

#: The four claim tables, whose value column takes its spec row by row.
CLAIM_TABLES = ("environment", "evidence_binding", "selection", "vision_ablation")


def _numbers_doc() -> dict:
    assert FILED_NUMBERS.is_file(), f"{FILED_NUMBERS} is committed evidence"
    return json.loads(FILED_NUMBERS.read_text(encoding="utf-8"))


def _filed_renderings() -> dict:
    assert FILED_RENDERINGS.is_file(), (
        f"{FILED_RENDERINGS} is committed evidence; run "
        f"`python scripts/iter11_paper_tables.py --write`")
    return json.loads(FILED_RENDERINGS.read_text(encoding="utf-8"))


def _column(key: str, header: str, fmt: str) -> dict:
    return {"key": key, "header": header, "format": fmt}


def _table(label: str, caption: str, columns: list[dict], rows: list[dict],
           note: str | None = None, source: str = "a_test_artifact") -> dict:
    out = {"label": label, "caption": caption, "columns": columns,
           "rows": rows, "source": source}
    if note is not None:
        out["note"] = note
    return out


def _one_table_doc(table: dict, name: str = "t") -> dict:
    """The smallest document this stage will render: one table."""
    return {"kind": numbers.KIND, "tables": {name: table}}


def _render(doc: dict, tmp_path: Path) -> tuple[dict, dict, list[str]]:
    """Render a numbers-shaped document somewhere throwaway."""
    source = tmp_path / "numbers.json"
    source.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    return tables.render_everything(source, tmp_path / "out")


def _render_one(table: dict, tmp_path: Path, name: str = "t"):
    return _render(_one_table_doc(table, name), tmp_path)


def _all_specs_table() -> dict:
    """One table with a column per base spec, so every renderer is exercised."""
    columns = [
        _column("row", "Row", "str"),
        _column("count", "Count", "int"),
        _column("two", "Two dp", "float2"),
        _column("four", "Four dp", "float4"),
        _column("signed", "Signed", "signed4"),
        _column("share", "Share", "pct1"),
        _column("p", "$p$", "p4"),
        _column("digest", "Digest", "sha8"),
        _column("interval", "Interval", "ci4"),
        _column("flag", "Flag", "bool"),
    ]
    row = {
        "row": "an_identifier_with_underscores",
        "count": 588,
        "two": 0.125,
        "four": 0.49498327759197325,
        "signed": -0.05080663265306112,
        "share": 0.006688963210702341,
        "p": 0.0002,
        "digest": "c03a5800ca95b02003c97f35c25db744"
                  "c357f6d35b216c4836b8cb32e9f91014",
        "interval": [-0.15121212121212124, -0.14111111111111113],
        "flag": True,
    }
    return _table("tab:every_spec", "Every rendering spec in one table",
                  columns, [row])


def _sentinel_for(spec: str) -> object:
    """A value of the right type for ``spec`` that cannot occur in the evidence."""
    _, base = tables.split_spec(spec)
    return {
        "str": "SENTINELTEXT",
        "int": 424242,
        "float2": 0.4242,
        "float4": 0.4242,
        "signed4": 0.4242,
        "pct1": 0.4242,
        "p4": 0.4242,
        "sha8": "ab" * 32,
        "ci4": [0.4242, 0.8484],
        "bool": True,
    }[base]


def _canonical_sentinel(spec: str) -> str:
    text, finding = tables.canonical_cell(spec, _sentinel_for(spec), "sentinel")
    assert finding is None, finding
    return text


def _walk_strings(node):
    if isinstance(node, dict):
        for value in node.values():
            yield from _walk_strings(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk_strings(value)
    elif isinstance(node, str):
        yield node


def _cell_count(doc: dict) -> int:
    return sum(len(table["rows"]) * len(table["columns"])
               for table in doc["tables"].values())


def _latex_header_row(tex: str) -> str:
    """The header row, which is the first line after ``\\toprule``.

    Not reachable through :func:`_latex_body_rows`, which returns what sits between
    ``\\midrule`` and ``\\bottomrule`` and therefore excludes it.
    """
    after = tex.split("\\toprule", 1)[1]
    return next(line for line in after.splitlines() if line.strip())


# ---------------------------------------------------------------------------
# The renderer holds no numbers of its own
# ---------------------------------------------------------------------------

class TestTheRendererHoldsNoNumbersOfItsOwn:
    """The first half of the numbers file's ``not_rendered`` requirement."""

    def test_perturbing_any_filed_cell_moves_the_print(self, tmp_path):
        """Every column of every table, one at a time.

        This is the test that would fail if a value had been typed into this
        script: a hardcoded cell does not move when the artifact moves. It is run
        per column rather than per cell because a column's spec is what decides how
        its cells print, and one sentinel per column exercises that decision.
        """
        doc = _numbers_doc()
        perturbed = 0
        for name in sorted(doc["tables"]):
            table = doc["tables"][name]
            _, baseline, findings = _render(
                {"kind": numbers.KIND, "tables": {name: table}}, tmp_path)
            assert not findings, findings
            before = baseline[f"tex/{name}.tex"]
            for column in table["columns"]:
                key = column["key"]
                moved = json.loads(json.dumps(table))
                spec, _ = tables.effective_spec(moved, moved["rows"][0], key)
                moved["rows"][0][key] = _sentinel_for(spec)
                _, after_files, moved_findings = _render(
                    {"kind": numbers.KIND, "tables": {name: moved}}, tmp_path)
                assert not moved_findings, moved_findings
                after = after_files[f"tex/{name}.tex"]
                expected = tables.latex_text(_canonical_sentinel(spec))
                assert expected in after, (
                    f"{name}.{key}: perturbing the filed value did not put "
                    f"{expected!r} in the LaTeX, so that cell is not coming out of "
                    f"the numbers file")
                if expected not in before:
                    assert before != after, f"{name}.{key}: the print did not move"
                perturbed += 1
        assert perturbed == sum(len(doc["tables"][n]["columns"])
                                for n in doc["tables"])

    def test_the_same_perturbation_moves_the_markdown_too(self, tmp_path):
        doc = _numbers_doc()
        moved = json.loads(json.dumps(doc["tables"]["sign_transport"]))
        moved["rows"][0]["mean"] = 0.4242
        _, files, findings = _render(
            {"kind": numbers.KIND, "tables": {"sign_transport": moved}}, tmp_path)
        assert not findings, findings
        assert "0.4242" in files["markdown/sign_transport.md"]
        assert "0.4242" in files["tex/sign_transport.tex"]

    def test_no_digit_in_a_table_body_is_absent_from_the_canonical_grid(
            self, tmp_path):
        """The printed digits are the filed digits, and there are no others.

        Restricted to the rows between ``\\midrule`` and ``\\bottomrule``: outside
        them the renderer legitimately writes numbers of its own, because the
        column-width weights it hands tabularx are computed from measurements and
        have to appear in the column specification.
        """
        doc = _numbers_doc()
        _, files, findings = _render(doc, tmp_path)
        assert not findings, findings
        for name in sorted(doc["tables"]):
            grid, cell_findings, _ = tables.render_grid(name, doc["tables"][name])
            assert not cell_findings, cell_findings
            canonical = "".join(cell for row in grid for cell in row)
            body = "".join(tables._latex_body_rows(files[f"tex/{name}.tex"]))
            for character in {c for c in body if c.isdigit()}:
                assert character in canonical, (
                    f"{name}: the LaTeX body prints the digit {character!r} and no "
                    f"cell of the canonical grid contains it, so a numeral entered "
                    f"the paper from this script rather than from the numbers file")

    def test_rendering_is_a_pure_function_of_the_numbers_file(self, tmp_path):
        doc = _numbers_doc()
        first_doc, first_files, _ = _render(doc, tmp_path)
        second_doc, second_files, _ = _render(doc, tmp_path)
        assert json.dumps(first_doc, sort_keys=True) == \
            json.dumps(second_doc, sort_keys=True)
        assert first_files == second_files

    def test_nothing_in_it_records_when_or_where_it_was_written(self, tmp_path):
        banned = {"generated_at", "timestamp", "git_commit", "git_dirty",
                  "code_commit", "python", "host", "generated_from_commit",
                  "code_tree_dirty", "untracked_code_paths"}

        def walk(node, path):
            if isinstance(node, dict):
                for key, value in node.items():
                    assert key not in banned, f"{path}/{key}"
                    walk(value, f"{path}/{key}")
            elif isinstance(node, list):
                for index, value in enumerate(node):
                    walk(value, f"{path}[{index}]")

        doc, _, _ = _render(_numbers_doc(), tmp_path)
        walk(doc, "")

    def test_the_only_hash_it_carries_is_the_one_it_read_the_numbers_from(
            self, tmp_path):
        """No clock and no commit, so the one thing that can move this document is
        the file it renders, and its hash says which bytes those were."""
        doc = _numbers_doc()
        source = tmp_path / "numbers.json"
        source.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
                          encoding="utf-8")
        rendered, _, _ = tables.render_everything(source, tmp_path / "out")
        assert rendered["inputs"]["sha256"] == numbers.sha256_file(source)
        assert rendered["inputs"]["path"] == str(source)

    def test_it_says_what_it_is_not(self, tmp_path):
        doc, _, _ = _render(_numbers_doc(), tmp_path)
        not_this = doc["what_this_file_is_not"]
        assert set(not_this) == {"not_a_source_of_numbers",
                                 "not_a_second_hash_of_the_outputs",
                                 "not_the_paper", "not_environment_sensitive"}
        assert "one function that turns a filed value into a printed cell" \
            in not_this["not_a_source_of_numbers"]

    def test_it_does_not_hash_the_files_it_writes(self, tmp_path):
        """The closeout manifest binds every file under paper/ by hash already.

        Recording those hashes here too would be a second place for them to go
        stale, and comparing two hashes of one file is weaker than comparing the
        bytes, which is what ``--verify`` does instead.
        """
        doc, files, _ = _render(_numbers_doc(), tmp_path)
        digests = {value for value in _walk_strings(doc)
                   if len(value) == 64
                   and all(c in "0123456789abcdef" for c in value)}
        assert digests == {doc["inputs"]["sha256"]}, (
            f"the renderings file carries {len(digests)} digests and the only one "
            f"it should carry is the numbers file's")
        assert sorted(files) == doc["totals"]["output_paths"], (
            "the document does not name the files this stage actually renders")


# ---------------------------------------------------------------------------
# The two renderings cannot disagree
# ---------------------------------------------------------------------------

class TestTheTwoRenderingsOfOneTableCannotDisagree:
    """The second half of the ``not_rendered`` requirement, tested not assumed."""

    def test_the_filed_tables_agree_cell_by_cell(self):
        assert tables.the_renderings_agree(_numbers_doc(), "n.json", "0" * 64) == []

    def test_every_cell_unescapes_to_the_canonical_text_in_both_formats(
            self, tmp_path):
        doc = _numbers_doc()
        _, files, findings = _render(doc, tmp_path)
        assert not findings, findings
        checked = 0
        for name in sorted(doc["tables"]):
            table = doc["tables"][name]
            grid, _, _ = tables.render_grid(name, table)
            tex_rows = tables._latex_body_rows(files[f"tex/{name}.tex"])
            md_rows = tables._markdown_body_rows(files[f"markdown/{name}.md"])
            assert len(tex_rows) == len(grid) == len(md_rows), name
            for index, canonical in enumerate(grid):
                tex_cells = tables._split_latex_row(tex_rows[index])
                md_cells = tables._split_markdown_row(md_rows[index])
                assert len(tex_cells) == len(canonical), (name, index)
                assert len(md_cells) == len(canonical), (name, index)
                for position, expected in enumerate(canonical):
                    assert tables.latex_unescape(tex_cells[position]) == expected, (
                        name, index, position)
                    assert tables.markdown_unescape(md_cells[position]) == expected, (
                        name, index, position)
                    checked += 1
        assert checked == _cell_count(doc)

    def test_the_notes_agree_too(self, tmp_path):
        doc = _numbers_doc()
        _, files, findings = _render(doc, tmp_path)
        assert not findings, findings
        for name in sorted(doc["tables"]):
            note = doc["tables"][name].get("note")
            if not note:
                continue
            expected = " ".join(note.split())
            from_tex = " ".join(tables.latex_unescape(
                tables._latex_note(files[f"tex/{name}.tex"])).split())
            from_md = " ".join(tables.markdown_unescape(
                tables._markdown_note(files[f"markdown/{name}.md"])).split())
            assert from_tex == expected, name
            assert from_md == expected, name

    def test_the_agreement_check_parses_the_emitted_text(self, tmp_path):
        """It is not structural: the rows are split back out of the files.

        A check that compared the in-memory grid with itself would pass whatever
        the encoders did, which is the shape of a pin that enforces nothing.
        """
        doc = _numbers_doc()
        _, files, _ = _render(doc, tmp_path)
        table = doc["tables"]["sign_transport"]
        tex_rows = tables._latex_body_rows(files["tex/sign_transport.tex"])
        md_rows = tables._markdown_body_rows(files["markdown/sign_transport.md"])
        assert len(tex_rows) == len(table["rows"])
        assert len(md_rows) == len(table["rows"])
        assert len(tables._split_latex_row(tex_rows[0])) == len(table["columns"])
        assert len(tables._split_markdown_row(md_rows[0])) == len(table["columns"])

    def test_a_cell_the_two_formats_disagree_about_is_reported(self, monkeypatch):
        """Driving the comparison over a doctored encoding.

        ``the_renderings_agree`` renders both formats itself, so the way to show it
        can fail is to make one encoder lie. An encoder that dropped the sign of a
        negative mean is the kind of drift the numbers file says cannot happen.
        """
        doc = _numbers_doc()
        honest = tables.markdown_text

        def lying(text):
            return honest(text).replace("-", "")

        monkeypatch.setattr(tables, "markdown_text", lying)
        issues = tables.the_renderings_agree(doc, "n.json", "0" * 64)
        assert issues, "a Markdown encoder that dropped minus signs was not caught"
        assert any("print different numbers" in issue for issue in issues)

    def test_an_escaped_cell_survives_the_round_trip(self):
        hostile = "".join(sorted(tables.LATEX_ESCAPES))
        assert tables.latex_unescape(tables.latex_text(hostile)) == hostile

    def test_escaping_is_one_pass_so_it_cannot_re_escape_itself(self):
        """The classic failure: replace the backslash first, then escape the braces
        that replacement just introduced."""
        assert tables.latex_text("\\") == "\\textbackslash{}"
        assert tables.latex_text("~") == "\\textasciitilde{}"
        assert tables.latex_text("^") == "\\textasciicircum{}"
        assert tables.latex_unescape(tables.latex_text("\\~^")) == "\\~^"
        assert tables.latex_text("a_b") == "a\\_b"

    def test_markdown_escapes_only_the_pipe(self):
        assert tables.markdown_text("a|b") == "a\\|b"
        assert tables.markdown_unescape("a\\|b") == "a|b"
        assert tables.markdown_text("dependency_lock") == "dependency_lock"



# ---------------------------------------------------------------------------
# Which spec governs a cell
# ---------------------------------------------------------------------------

class TestEveryCellIsRenderedUnderTheSpecItWasFiledWith:
    def test_the_row_level_spec_governs_a_claim_tables_value_column(self):
        doc = _numbers_doc()
        governed = 0
        for name in CLAIM_TABLES:
            table = doc["tables"][name]
            assert {c["key"] for c in table["columns"]} == {"claim", "value", "where"}
            for column in table["columns"]:
                assert column["format"] == "str", (name, column["key"])
            for row in table["rows"]:
                spec, origin = tables.effective_spec(table, row, "value")
                assert origin == "row", (name, row["claim"])
                assert spec == row["format"]
                governed += 1
                _, column_origin = tables.effective_spec(table, row, "claim")
                assert column_origin == "column"
        assert governed == 64

    def test_a_hash_is_not_rounded_and_a_count_is_not_a_decimal(self, tmp_path):
        """What the row-level spec is for, on the two rows that show it.

        tab:environment files a sha256 and a package count in the SAME column. One
        spec for the column would either round the digest to 4 decimals or print
        the count as 100.0000.
        """
        table = _numbers_doc()["tables"]["environment"]
        rows = {row["claim"]: row for row in table["rows"]}
        digest = rows["dependency lock sha256"]
        count = rows["packages in the freeze"]
        assert digest["format"] == "sha8" and count["format"] == "int"
        _, files, findings = _render_one(table, tmp_path, "environment")
        assert not findings, findings
        tex = files["tex/environment.tex"]
        assert digest["value"][:8] in tex
        assert digest["value"] not in tex, "the whole digest was printed"
        assert f" {count['value']} " in tex
        assert f"{count['value']}.0000" not in tex

    def test_a_column_spec_governs_every_other_cell(self):
        doc = _numbers_doc()
        for name in sorted(doc["tables"]):
            if name in CLAIM_TABLES:
                continue
            table = doc["tables"][name]
            for row in table["rows"]:
                assert "format" not in row, name
                for column in table["columns"]:
                    spec, origin = tables.effective_spec(table, row, column["key"])
                    assert origin == "column"
                    assert spec == column["format"]

    def test_a_row_spec_with_no_value_column_to_govern_is_a_finding(self, tmp_path):
        table = _table("tab:x", "A row spec nobody can use",
                       [_column("claim", "Claim", "str"),
                        _column("where", "Filed in", "str")],
                       [{"claim": "a", "where": "b", "format": "int"}])
        _, _, findings = _render_one(table, tmp_path)
        assert any("no 'value' column for it to govern" in f for f in findings), \
            findings

    def test_a_cell_whose_type_disagrees_with_its_spec_is_refused(self):
        cases = [
            ("int", 3.5, "prints a count"),
            ("int", True, "prints a count"),
            ("float4", "0.5", "4-decimal spec"),
            ("bool", "true", "boolean spec"),
            ("str", 0.49498327759197325, "literal text"),
            ("sha8", 12345678, "hex digest prefix"),
            ("ci4", "[-0.1, 0.1]", "prints an interval"),
            ("pct1", "half", "spec 'pct1'"),
        ]
        for spec, value, why in cases:
            text, finding = tables.canonical_cell(spec, value, "a cell")
            assert finding is not None, f"{spec} accepted {value!r}"
            assert text == "", (spec, value)
            assert why.split()[0] in finding or spec in finding, (spec, finding)

    def test_a_str_cell_is_not_coerced_because_that_prints_full_precision(self):
        """The exact failure the row-level spec exists to prevent."""
        text, finding = tables.canonical_cell(
            "str", 0.49498327759197325, "a count filed under str")
        assert finding is not None
        assert "0.49498327759197325" in finding

    def test_a_spec_outside_the_vocabulary_is_a_finding(self):
        text, finding = tables.canonical_cell("float7", 0.5, "a cell")
        assert text == ""
        assert finding is not None and "float7" in finding
        assert "guess" in finding

    def test_null_under_a_spec_that_does_not_tolerate_it_is_a_finding(self):
        for spec in ("str", "int", "bool", "p4", "ci4", "float4"):
            text, finding = tables.canonical_cell(spec, None, "a cell")
            assert text == "" and finding is not None, spec
            assert "null" in finding

    def test_the_absent_mark_filed_as_a_value_is_a_finding(self):
        """Otherwise a reader could not tell the two apart, which is the whole
        reason an absent value gets a mark."""
        text, finding = tables.canonical_cell(
            "optional_str", tables.ABSENT, "a cell")
        assert text == "" and finding is not None
        assert tables.ABSENT in finding


# ---------------------------------------------------------------------------
# What is escaped and what is not
# ---------------------------------------------------------------------------

class TestWhatIsEscapedAndWhatIsNot:
    def test_authored_mathematics_in_a_header_survives_both_formats(self, tmp_path):
        table = _numbers_doc()["tables"]["sign_transport"]
        _, files, findings = _render_one(table, tmp_path, "sign_transport")
        assert not findings, findings
        for name in ("tex", "markdown"):
            extension = "tex" if name == "tex" else "md"
            text = files[f"{name}/sign_transport.{extension}"]
            assert "$\\Delta_{TV}$ mean" in text, name
            assert "95\\% CI" in text, name
            assert "$p$" in text, name
            assert "\\textbackslash" not in text.split("\\midrule")[0], (
                f"{name}: a header was escaped and its mathematics is now "
                f"punctuation")

    def test_a_literal_identifier_in_a_cell_is_escaped_for_latex_only(self, tmp_path):
        table = _table("tab:x", "Escaping", [_column("where", "Filed in", "str")],
                       [{"where": "dependency_lock"}])
        _, files, findings = _render_one(table, tmp_path)
        assert not findings, findings
        assert "dependency\\_lock" in files["tex/t.tex"]
        assert "dependency_lock" in files["markdown/t.md"]

    def test_an_ampersand_is_escaped_because_it_would_move_a_column(self, tmp_path):
        """The worst of the ten: it does not fail to compile, it silently shifts
        every following cell of the row one place to the left."""
        table = _table("tab:x", "A shell command", [_column("value", "Value", "str")],
                       [{"value": "conda activate x && pip install -e ."}])
        _, files, findings = _render_one(table, tmp_path)
        assert not findings, findings
        body = tables._latex_body_rows(files["tex/t.tex"])
        assert len(tables._split_latex_row(body[0])) == 1, (
            "the '&&' created a column boundary")
        assert "\\&\\&" in files["tex/t.tex"]

    def test_braces_in_a_cell_are_escaped(self, tmp_path):
        table = _table("tab:x", "A generation config",
                       [_column("value", "Value", "str")],
                       [{"value": '{"do_sample": false, "seed": 42}'}])
        _, files, findings = _render_one(table, tmp_path)
        assert not findings, findings
        assert "\\{" in files["tex/t.tex"] and "\\}" in files["tex/t.tex"]

    def test_a_note_is_escaped_and_never_passes_through(self, tmp_path):
        """Counted rather than spot-checked, so it cannot pass on the wrong token.

        A note is prose that quotes identifiers: panel_exclusions' carries 19
        underscores in things like ``CMST_456921/text_only``, and judge_reliability's
        cross-references ``tab:panel_exclusions``. Every one of them has to come out
        of the LaTeX escaped and out of the Markdown intact.
        """
        for name in ("panel_exclusions", "judge_reliability", "vision_ablation"):
            table = _numbers_doc()["tables"][name]
            note = table["note"]
            assert "_" in note, name
            _, files, findings = _render_one(table, tmp_path, name)
            assert not findings, findings
            tex_note = tables._latex_note(files[f"tex/{name}.tex"])
            md_note = tables._markdown_note(files[f"markdown/{name}.md"])
            assert tex_note.count("\\_") == note.count("_"), name
            assert "\\_" not in md_note, name
            assert md_note.count("_") == note.count("_"), name
        assert "tab:panel\\_exclusions" in tables._latex_note(
            _render_one(_numbers_doc()["tables"]["judge_reliability"], tmp_path,
                        "judge_reliability")[1]["tex/judge_reliability.tex"])

    def test_a_caption_with_an_odd_number_of_dollars_is_refused(self, tmp_path):
        table = _table("tab:x", "An unclosed $ math group",
                       [_column("a", "A", "str")], [{"a": "x"}])
        _, _, findings = _render_one(table, tmp_path)
        assert any("opens a math group it never closes" in f for f in findings), \
            findings

    def test_a_caption_with_an_ampersand_is_refused(self, tmp_path):
        table = _table("tab:x", "Salt & pepper", [_column("a", "A", "str")],
                       [{"a": "x"}])
        _, _, findings = _render_one(table, tmp_path)
        assert any("ends a table cell" in f for f in findings), findings

    def test_a_caption_with_a_bare_percent_is_refused(self, tmp_path):
        table = _table("tab:x", "50% of items", [_column("a", "A", "str")],
                       [{"a": "x"}])
        _, _, findings = _render_one(table, tmp_path)
        assert any("comments out the rest of the line" in f for f in findings), \
            findings

    def test_a_caption_with_an_escaped_percent_is_accepted(self, tmp_path):
        table = _table("tab:x", r"95\% CI", [_column("a", "A", "str")],
                       [{"a": "x"}])
        _, _, findings = _render_one(table, tmp_path)
        assert not findings, findings

    def test_an_underscore_outside_math_mode_is_refused(self, tmp_path):
        table = _table("tab:x", "The dependency_lock file",
                       [_column("a", "A", "str")], [{"a": "x"}])
        _, _, findings = _render_one(table, tmp_path)
        assert any("Missing $ inserted" in f for f in findings), findings

    def test_an_underscore_inside_math_mode_is_accepted(self, tmp_path):
        """The filed caption carries exactly this, and refusing it would refuse the
        paper's own estimand."""
        table = _table("tab:x", r"The frozen $\Delta_{TV}$ estimand",
                       [_column("a", "A", "str")], [{"a": "x"}])
        _, _, findings = _render_one(table, tmp_path)
        assert not findings, findings

    def test_unbalanced_braces_in_a_header_are_refused(self, tmp_path):
        table = _table("tab:x", "A caption",
                       [_column("a", r"$\Delta_{TV$", "str")], [{"a": "x"}])
        _, _, findings = _render_one(table, tmp_path)
        assert any("never closed" in f for f in findings), findings

    def test_a_note_carrying_mathematics_is_refused(self, tmp_path):
        for note in (r"The mean $\Delta$ moved", "A backslash \\ here"):
            table = _table("tab:x", "A caption", [_column("a", "A", "str")],
                           [{"a": "x"}], note=note)
            _, _, findings = _render_one(table, tmp_path)
            assert any("carries authored LaTeX" in f for f in findings), (note, findings)

    def test_a_cell_carrying_mathematics_is_refused(self, tmp_path):
        table = _table("tab:x", "A caption", [_column("a", "A", "str")],
                       [{"a": r"$\Delta$"}])
        _, _, findings = _render_one(table, tmp_path)
        assert any("carries authored LaTeX" in f for f in findings), findings

    def test_an_underscore_markdown_would_italicise_is_refused(self, tmp_path):
        """CommonMark leaves an underscore inside a word alone, which is where
        every underscore in this document sits. One that is not inside a word can
        open emphasis, and Markdown is escaped for '|' only."""
        table = _table("tab:x", "A caption", [_column("a", "A", "str")],
                       [{"a": "the _emphasis_ trap"}])
        _, _, findings = _render_one(table, tmp_path)
        assert any("start of emphasis" in f for f in findings), findings

    def test_no_underscore_in_the_filed_document_can_open_emphasis(self, tmp_path):
        _, _, findings = _render(_numbers_doc(), tmp_path)
        assert not [f for f in findings if "emphasis" in f], findings

    def test_the_census_is_a_count_of_the_real_document(self, tmp_path):
        doc, _, findings = _render(_numbers_doc(), tmp_path)
        assert not findings, findings
        escaping = doc["the_rendering_contract"]["what_is_escaped_and_what_is_not"]
        numbers_doc = _numbers_doc()
        headers = sum(len(numbers_doc["tables"][n]["columns"])
                      for n in numbers_doc["tables"])
        assert escaping["n_authored_latex_fields_passed_through"] == \
            headers + len(numbers_doc["tables"])
        assert escaping["n_of_them_cells"] == _cell_count(numbers_doc)
        assert escaping["n_of_them_notes"] == sum(
            1 for n in numbers_doc["tables"] if "note" in numbers_doc["tables"][n])
        assert escaping["n_literal_strings_escaped"] == \
            escaping["n_of_them_cells"] + escaping["n_of_them_notes"]
        assert escaping["n_dollars_or_backslashes_in_literal_text"] == 0
        assert escaping["n_markdown_emphasis_risks"] == 0
        assert escaping["authored_latex_fields"] == list(tables.AUTHORED_LATEX_FIELDS)
        assert escaping["escaped_fields"] == list(tables.ESCAPED_FIELDS)

    def test_the_percent_signs_it_escapes_are_the_ones_the_rendering_creates(self,
                                                                           tmp_path):
        """A census of the FILED values would report that this stage needs no
        percent escaping, because a share is filed as 0.006688963210702341 and
        renders to ``0.7%``. The census counts rendered cells for that reason."""
        doc, _, findings = _render(_numbers_doc(), tmp_path)
        assert not findings, findings
        counts = doc["the_rendering_contract"][
            "what_is_escaped_and_what_is_not"]["hostile_characters_found_in_literal_text"]
        pct1_cells = sum(
            1 for _, _, _, _, spec, _, value in tables.cells_of(_numbers_doc())
            if spec in ("pct1", "optional_pct1") and value is not None)
        assert counts["%"] == pct1_cells
        assert "%" not in json.dumps(
            [v for _, _, _, _, _, _, v in tables.cells_of(_numbers_doc())
             if isinstance(v, str)])


# ---------------------------------------------------------------------------
# The layout is measured, not chosen
# ---------------------------------------------------------------------------

class TestTheLayoutIsMeasuredRatherThanChosen:
    def test_the_twelve_tables_take_the_mechanism_their_widths_select(self, tmp_path):
        doc, _, findings = _render(_numbers_doc(), tmp_path)
        assert not findings, findings
        expected = {
            "blinding": "resizebox",
            "bound_configurations": "resizebox",
            "censoring_bound": "resizebox",
            "environment": "tabularx",
            "evidence_binding": "tabularx",
            "judge_reliability": "resizebox",
            "panel_exclusions": "resizebox",
            "selection": "tabularx",
            "sign_transport": "resizebox",
            "truncation": "resizebox",
            "vision_ablation": "resizebox",
            "vision_ablation_shift": "plain_tabular",
        }
        assert {name: block["layout"]["mechanism"]
                for name, block in doc["tables"].items()} == expected
        assert doc["the_rendering_contract"]["layout_is_measured"][
            "mechanisms_used"] == {"plain_tabular": 1, "resizebox": 8, "tabularx": 3}

    def test_a_width_discounts_the_macros_a_header_is_typed_in(self):
        assert tables._printed_width("$\\kappa$ (refusal)") == len("kappa (refusal)")
        assert tables._printed_width("$\\Delta_{TV}$ mean") == len("Delta_TV mean")
        assert tables._printed_width("95\\% CI") == len("95% CI")
        assert tables._printed_width("plain") == 5

    def test_a_width_is_the_longer_of_the_header_and_the_widest_cell(self, tmp_path):
        table = _table("tab:x", "A caption",
                       [_column("a", "A very long header indeed", "str")],
                       [{"a": "short"}])
        _, _, _ = _render_one(table, tmp_path)
        grid = [["short"]]
        layout = tables.column_layout("t", table, grid)
        assert layout["column_widths_chars"]["a"] == len("A very long header indeed")

    def test_a_cell_too_wide_for_a_fixed_column_takes_the_wrapping_route(self,
                                                                        tmp_path):
        table = _table("tab:x", "A caption",
                       [_column("a", "A", "str"), _column("b", "B", "str")],
                       [{"a": "x" * 61, "b": "y"}])
        doc, files, findings = _render_one(table, tmp_path)
        assert not findings, findings
        layout = doc["tables"]["t"]["layout"]
        assert layout["mechanism"] == "tabularx"
        assert layout["unbreakable_columns"] == ["a"]
        assert "\\begin{tabularx}{\\textwidth}" in files["tex/t.tex"]

    def test_too_many_short_columns_takes_the_scaling_route(self, tmp_path):
        columns = [_column(f"c{i}", f"H{i}", "int") for i in range(20)]
        table = _table("tab:x", "A caption", columns,
                       [{f"c{i}": i for i in range(20)}])
        doc, files, findings = _render_one(table, tmp_path)
        assert not findings, findings
        layout = doc["tables"]["t"]["layout"]
        assert layout["mechanism"] == "resizebox"
        assert layout["unbreakable_columns"] == []
        assert "\\resizebox{\\textwidth}{!}{%" in files["tex/t.tex"]
        assert files["tex/t.tex"].count("\\resizebox") == 1

    def test_a_table_that_fits_is_left_alone(self, tmp_path):
        table = _table("tab:x", "A caption", [_column("a", "A", "int")],
                       [{"a": 1}, {"a": 2}])
        doc, files, findings = _render_one(table, tmp_path)
        assert not findings, findings
        assert doc["tables"]["t"]["layout"]["mechanism"] == "plain_tabular"
        assert "\\resizebox" not in files["tex/t.tex"]
        assert "tabularx" not in files["tex/t.tex"]
        assert "\\begin{tabular}{@{}r@{}}" in files["tex/t.tex"]

    def test_the_column_beside_a_wide_one_wraps_too_so_it_is_not_starved(self,
                                                                        tmp_path):
        """The failure this rule exists for.

        tab:environment's claim column measures 47 characters and its value column
        347. Wrapping only the column that triggered the wrap leaves the 47 fixed
        at half the text block and gives the 347 what is left, which is one word
        per line.
        """
        table = _table("tab:x", "A caption",
                       [_column("claim", "Claim", "str"),
                        _column("value", "Value", "str")],
                       [{"claim": "c" * 47, "value": "v" * 347}])
        doc, _, findings = _render_one(table, tmp_path)
        assert not findings, findings
        layout = doc["tables"]["t"]["layout"]
        assert layout["unbreakable_columns"] == ["value"]
        assert layout["wrapping_columns"] == ["claim", "value"]
        assert sorted(layout["wrap_weights"]) == ["claim", "value"]
        assert layout["wrap_weights"]["value"] > layout["wrap_weights"]["claim"]

    def test_the_wrap_weights_are_what_tabularx_requires(self, tmp_path):
        doc, _, findings = _render(_numbers_doc(), tmp_path)
        assert not findings, findings
        for name, block in sorted(doc["tables"].items()):
            layout = block["layout"]
            weights = layout["wrap_weights"]
            if layout["mechanism"] != "tabularx":
                assert weights is None, name
                assert layout["wrapping_columns"] == [], (
                    f"{name}: files wrapping columns while taking the "
                    f"{layout['mechanism']} route, where nothing wraps")
                continue
            assert sorted(weights) == sorted(layout["wrapping_columns"]), name
            assert abs(sum(weights.values()) - len(weights)) < 0.011, (name, weights)

    def test_the_share_is_by_the_square_root_so_the_widest_does_not_take_all(
            self, tmp_path):
        table = _table("tab:x", "A caption",
                       [_column("a", "A", "str"), _column("b", "B", "str")],
                       [{"a": "a" * 40, "b": "b" * 360}])
        doc, _, findings = _render_one(table, tmp_path)
        assert not findings, findings
        weights = doc["tables"]["t"]["layout"]["wrap_weights"]
        ratio = weights["b"] / weights["a"]
        assert ratio == pytest.approx(math.sqrt(360 / 40)), (
            f"the share is not by the {tables.WRAP_WEIGHT_POWER} power of the width")
        assert ratio < 360 / 40, (
            "the share is proportional to the width itself, which starves the "
            "narrower column")
        assert ratio > 1.0, "the wider column did not get the wider share"

    def test_a_scaled_or_fitting_table_files_no_wrapping_columns(self, tmp_path):
        doc, _, findings = _render(_numbers_doc(), tmp_path)
        assert not findings, findings
        for name, block in sorted(doc["tables"].items()):
            layout = block["layout"]
            over = [key for key, width in layout["column_widths_chars"].items()
                    if width > tables.WRAP_SHARE_OF_BUDGET
                    * tables.CHARS_PER_TEXTWIDTH]
            assert layout["columns_over_the_share_floor"] == over, name
            if layout["mechanism"] != "tabularx":
                assert layout["wrapping_columns"] == [], name

    def test_alignment_follows_the_spec_and_not_the_table(self, tmp_path):
        doc, files, findings = _render_one(_all_specs_table(), tmp_path, "specs")
        assert not findings, findings
        alignment = doc["tables"]["specs"]["layout"]["alignment"]
        assert alignment == {"row": "l", "count": "r", "two": "r", "four": "r",
                             "signed": "r", "share": "r", "p": "r", "digest": "r",
                             "interval": "r", "flag": "c"}
        assert "\\begin{tabular}{@{}lrrrrrrrrc@{}}" in files["tex/specs.tex"]

    def test_a_mixed_spec_column_is_text_because_no_alignment_fits_it(self, tmp_path):
        doc, _, findings = _render(_numbers_doc(), tmp_path)
        assert not findings, findings
        for name in CLAIM_TABLES:
            layout = doc["tables"][name]["layout"]
            assert layout["mixed_spec_columns"] == ["value"], name
            assert layout["alignment"]["value"] == "l", name
        for name in sorted(doc["tables"]):
            if name not in CLAIM_TABLES:
                assert doc["tables"][name]["layout"]["mixed_spec_columns"] == [], name

    def test_the_packages_each_mechanism_needs_are_filed(self, tmp_path):
        doc, _, findings = _render(_numbers_doc(), tmp_path)
        assert not findings, findings
        for name, block in sorted(doc["tables"].items()):
            layout = block["layout"]
            assert layout["latex_packages_required"] == \
                list(tables.PACKAGE_FOR_MECHANISM[layout["mechanism"]]), name
        assert doc["the_rendering_contract"]["latex_packages_required"] == \
            ["booktabs", "graphicx", "tabularx"]

    def test_a_document_claiming_a_mechanism_nobody_emits_is_refused(self, tmp_path):
        doc, _, findings = _render(_numbers_doc(), tmp_path)
        assert not findings, findings
        doc["tables"]["blinding"]["layout"]["mechanism"] = "longtable"
        issues = tables.check_the_renderings(doc)
        assert any("longtable" in issue for issue in issues), issues

    def test_a_scaled_table_claiming_wrapping_columns_is_refused(self, tmp_path):
        doc, _, findings = _render(_numbers_doc(), tmp_path)
        assert not findings, findings
        doc["tables"]["blinding"]["layout"]["wrapping_columns"] = ["row"]
        issues = tables.check_the_renderings(doc)
        assert any("nothing wraps unless the table is a tabularx" in i
                   for i in issues), issues

    def test_wrap_weights_that_do_not_sum_to_the_column_count_are_refused(self,
                                                                         tmp_path):
        doc, _, findings = _render(_numbers_doc(), tmp_path)
        assert not findings, findings
        doc["tables"]["environment"]["layout"]["wrap_weights"] = {
            "claim": 0.5, "value": 0.5}
        issues = tables.check_the_renderings(doc)
        assert any("\\hsize factors" in issue for issue in issues), issues

    def test_widths_that_do_not_select_the_filed_wrapping_columns_are_refused(
            self, tmp_path):
        doc, _, findings = _render(_numbers_doc(), tmp_path)
        assert not findings, findings
        doc["tables"]["environment"]["layout"]["wrapping_columns"] = ["value"]
        issues = tables.check_the_renderings(doc)
        assert any("not the ones the measurements pick out" in i for i in issues), \
            issues


# ---------------------------------------------------------------------------
# What the specs mean
# ---------------------------------------------------------------------------

class TestWhatTheSpecsMean:
    def test_every_base_spec_renders_what_it_says(self):
        expected = {
            "str": ("an_identifier", "an_identifier"),
            "int": (588, "588"),
            "float2": (0.125, "0.12"),
            "float4": (0.49498327759197325, "0.4950"),
            "signed4": (-0.05080663265306112, "-0.0508"),
            "pct1": (0.006688963210702341, "0.7%"),
            "p4": (0.0002, "0.0002"),
            "sha8": ("c03a5800ca95b020", "c03a5800"),
            "ci4": ([-0.15121212121212124, -0.14111111111111113],
                    "[-0.1512, -0.1411]"),
            "bool": (False, "false"),
        }
        for spec, (value, want) in expected.items():
            text, finding = tables.canonical_cell(spec, value, "a cell")
            assert finding is None, (spec, finding)
            assert text == want, (spec, text, want)

    def test_the_optional_form_of_every_spec_tolerates_an_absent_value(self):
        for base in tables.RENDERERS:
            spec = f"optional_{base}"
            text, finding = tables.canonical_cell(spec, None, "a cell")
            assert finding is None, (spec, finding)
            assert text == tables.ABSENT, spec

    def test_pct1_is_a_fraction_and_not_already_a_percentage(self):
        """Read the other way this is a hundredfold error, and the printed table
        gives a reader no way to see it."""
        text, finding = tables.canonical_cell("pct1", 0.7658862876254181, "a share")
        assert finding is None and text == "76.6%"

    def test_a_share_outside_zero_to_one_is_refused(self):
        for value in (1.5, -0.2, 76.6):
            text, finding = tables.canonical_cell("pct1", value, "a share")
            assert text == "" and finding is not None, value
            assert "fraction of a whole" in finding

    def test_the_filed_shares_print_as_a_hundred_percent(self, tmp_path):
        """The measurement that corroborates the reading, on the real table."""
        table = _numbers_doc()["tables"]["vision_ablation_shift"]
        _, files, findings = _render_one(table, tmp_path, "shift")
        assert not findings, findings
        grid, _, _ = tables.render_grid("shift", table)
        shares = [row[2].rstrip("%") for row in grid]
        assert sum(float(share) for share in shares) == pytest.approx(100.0, abs=0.05)
        assert table["rows"][0]["n_items"] + table["rows"][1]["n_items"] \
            + table["rows"][2]["n_items"] + table["rows"][3]["n_items"] \
            + table["rows"][4]["n_items"] == 299

    def test_the_reading_is_corroborated_in_the_document_not_assumed(self, tmp_path):
        doc, _, findings = _render(_numbers_doc(), tmp_path)
        assert not findings, findings
        block = doc["the_rendering_contract"]["pct1_is_a_fraction"]
        assert block["columns_that_sum_to_one_whole"] == \
            ["vision_ablation_shift.share"]
        measured = {f"{row['table']}.{row['column']}": row["sums_to_one_whole"]
                    for row in block["measured_over"]}
        assert measured["vision_ablation_shift.share"] is True
        assert "why_a_column_may_not_sum_to_one" in block

    def test_sha8_prints_a_prefix_of_a_longer_digest(self):
        text, finding = tables.canonical_cell("sha8", "a" * 64, "a digest")
        assert finding is None and text == "a" * 8

    def test_sha8_also_shortens_the_forty_character_git_id(self):
        """tab:environment files a resolved model revision, which is a git id and
        not a sha256, under the same spec."""
        text, finding = tables.canonical_cell(
            "sha8", "c202236235762e1c871ad0ccb60c8ee5ba337b9a", "a revision")
        assert finding is None and text == "c2022362"

    def test_the_digest_lengths_it_shortened_are_filed(self, tmp_path):
        doc, _, findings = _render(_numbers_doc(), tmp_path)
        assert not findings, findings
        prefix = doc["the_rendering_contract"]["sha8_prints_a_prefix"]
        assert prefix["n_chars_printed"] == tables.SHA8_CHARS
        assert prefix["n_cells"] == 7
        assert prefix["filed_lengths"] == {"40": 1, "64": 6}

    def test_a_digest_too_short_or_not_hex_is_refused(self):
        for value in ("abc", "z" * 64, "ABCDEF0123456789"):
            text, finding = tables.canonical_cell("sha8", value, "a digest")
            assert text == "" and finding is not None, value

    def test_the_bootstrap_floor_prints_as_itself_and_not_as_zero(self):
        text, finding = tables.canonical_cell("p4", 0.0002, "a p")
        assert finding is None and text == "0.0002"

    def test_a_nonzero_p_that_would_print_as_zero_is_refused(self):
        """The numbers file establishes that the smallest p this evidence can
        report is 1/5000. A p under it is either a different estimator or a bug,
        and printing it as 0.0000 would claim an impossibility."""
        text, finding = tables.canonical_cell("p4", 1e-9, "a p")
        assert text == "" and finding is not None
        assert "0.0000" in finding and "1/5000" in finding

    def test_an_exactly_zero_p_is_not_the_same_case(self):
        text, finding = tables.canonical_cell("p4", 0.0, "a p")
        assert finding is None and text == "0.0000"

    def test_a_p_outside_zero_to_one_is_refused(self):
        for value in (1.5, -0.1):
            text, finding = tables.canonical_cell("p4", value, "a p")
            assert text == "" and finding is not None, value

    def test_an_interval_whose_ends_are_the_wrong_way_round_is_refused(self):
        text, finding = tables.canonical_cell("ci4", [0.2, -0.1], "an interval")
        assert text == "" and finding is not None
        assert "lower end exceeds" in finding

    def test_an_interval_that_is_not_two_ends_is_refused(self):
        for value in ([0.1], [0.1, 0.2, 0.3], 0.5):
            text, finding = tables.canonical_cell("ci4", value, "an interval")
            assert text == "" and finding is not None, value

    def test_a_boolean_prints_the_spelling_the_artifact_uses(self):
        """'yes'/'no' or a check mark would read better and would be the first
        piece of vocabulary this renderer owned. A reader can grep the artifact
        for exactly what the table printed."""
        assert tables.canonical_cell("bool", True, "a flag")[0] == "true"
        assert tables.canonical_cell("bool", False, "a flag")[0] == "false"

    def test_a_negative_mean_keeps_the_ascii_minus_in_both_formats(self):
        """Math mode would print a proper minus and would make the two renderings
        differ by more than escaping."""
        text, finding = tables.canonical_cell(
            "signed4", -0.14672426530612295, "a mean")
        assert finding is None and text == "-0.1467"
        assert tables.latex_text(text) == text
        assert tables.markdown_text(text) == text

    def test_a_cell_containing_a_newline_is_refused_in_either_format(self, tmp_path):
        table = _table("tab:x", "A caption", [_column("a", "A", "str")],
                       [{"a": "one\ntwo"}])
        _, _, findings = _render_one(table, tmp_path)
        assert any("newline or a tab" in f for f in findings), findings

    def test_the_rounding_is_pythons_and_that_is_filed_not_chosen(self):
        """Half to even on a value that is exactly representable.

        ``0.125`` is 1/8, so ``f"{0.125:.2f}"`` is ``0.12`` and a hand-check that
        rounds half up would say ``0.13``. This stage does not substitute a
        convention of its own, because a renderer with its own rounding is a
        renderer that can disagree with the other format -- but the convention has
        to be stated where a reviewer will find it rather than discovered by
        disagreeing with a cell.
        """
        assert tables.canonical_cell("float2", 0.125, "x")[0] == "0.12"
        assert tables.canonical_cell("float2", 0.135, "x")[0] == "0.14", (
            "0.135 is not exactly representable, so it is not a half-way case")

    def test_no_filed_value_sits_on_a_rounding_boundary(self, tmp_path):
        """The measurement that makes the convention unobservable in this paper.

        Half-to-even and half-up differ ONLY on a value exactly half way at the
        precision it prints, so while this list is empty every printed cell is what
        a reader gets under either convention.
        """
        doc, _, findings = _render(_numbers_doc(), tmp_path)
        assert not findings, findings
        rounding = doc["the_rendering_contract"]["rounding"]
        assert rounding["cells_exactly_half_way_at_the_precision_they_print"] == []
        assert rounding["digits_per_spec"] == dict(sorted(tables.ROUNDING_DIGITS.items()))
        assert rounding["n_cells_rounded"] > 0, (
            "no cell was rounded at all, so the measurement is vacuous")

    def test_a_value_on_a_boundary_would_be_named(self, tmp_path):
        """The guard has to be able to fire, or filing an empty list proves nothing."""
        table = _table("tab:x", "A caption", [_column("a", "A", "float2")],
                       [{"a": 0.125}])
        doc, _, findings = _render_one(table, tmp_path)
        assert not findings, findings
        named = doc["the_rendering_contract"]["rounding"][
            "cells_exactly_half_way_at_the_precision_they_print"]
        assert len(named) == 1 and "t.a" in named[0], named

    def test_a_filed_precision_the_document_disagrees_with_is_refused(self, tmp_path):
        doc, _, findings = _render(_numbers_doc(), tmp_path)
        assert not findings, findings
        doc["the_rendering_contract"]["rounding"]["digits_per_spec"]["p4"] = 3
        issues = tables.check_the_renderings(doc)
        assert any("rounding.digits_per_spec" in issue for issue in issues), issues



# ---------------------------------------------------------------------------
# The vocabulary is the numbers stage's, not this one's
# ---------------------------------------------------------------------------

class TestTheVocabularyIsTheNumbersStagesOwn:
    def test_every_spec_the_numbers_stage_declares_can_be_rendered(self):
        assert tables.declared_but_not_renderable() == []
        assert set(numbers.FORMATS) <= tables.CAPABLE_FORMATS

    def test_every_declared_spec_renders_a_value_of_its_own(self):
        """Including the two no table uses.

        ``optional_int`` and ``optional_ci4`` are declared upstream and filed by
        nothing today. A renderer written only against the tables in front of it
        would not implement them, and the first artifact to use one would be
        printed by guessing.
        """
        values = {"int": 7, "ci4": [0.1, 0.2], "str": "x", "bool": True,
                  "float2": 0.5, "float4": 0.5, "signed4": 0.5, "pct1": 0.5,
                  "p4": 0.5, "sha8": "a" * 64}
        for spec in numbers.FORMATS:
            optional, base = tables.split_spec(spec)
            value = None if optional else values[base]
            text, finding = tables.canonical_cell(spec, value, "a cell")
            assert finding is None, (spec, finding)
            assert text, spec

    def test_a_spec_declared_upstream_and_not_renderable_here_is_a_finding(
            self, tmp_path, monkeypatch):
        """The drift this comparison exists to catch, made to happen."""
        monkeypatch.setattr(numbers, "FORMATS", numbers.FORMATS + ("float9",))
        _, _, findings = _render(_numbers_doc(), tmp_path)
        assert any("float9" in f and "cannot render" in f for f in findings), findings

    def test_a_column_naming_a_spec_nobody_declares_is_a_finding(self, tmp_path):
        table = _table("tab:x", "A caption", [_column("a", "A", "float9")],
                       [{"a": 0.5}])
        _, _, findings = _render_one(table, tmp_path)
        assert any("not one the numbers stage declares" in f for f in findings), \
            findings

    def test_the_document_files_both_vocabularies_and_their_difference(self,
                                                                      tmp_path):
        doc, _, findings = _render(_numbers_doc(), tmp_path)
        assert not findings, findings
        contract = doc["the_rendering_contract"]
        assert contract["formats_the_numbers_stage_declares"] == list(numbers.FORMATS)
        assert contract["formats_this_stage_can_render"] == \
            sorted(tables.CAPABLE_FORMATS)
        assert contract["formats_declared_but_not_renderable"] == []
        assert contract["formats_declared_but_unused_by_any_table"] == \
            ["optional_ci4", "optional_int"]
        assert set(contract["formats_renderable_but_not_declared"]) == \
            set(tables.CAPABLE_FORMATS) - set(numbers.FORMATS)

    def test_the_specs_the_twelve_tables_actually_use_are_counted(self, tmp_path):
        doc, _, findings = _render(_numbers_doc(), tmp_path)
        assert not findings, findings
        used = doc["the_rendering_contract"]["formats_the_tables_use"]
        assert sum(used.values()) == _cell_count(_numbers_doc())
        assert set(used) <= set(numbers.FORMATS)
        assert len(used) == 14

    def test_a_document_claiming_a_vocabulary_the_code_does_not_hold_is_refused(
            self, tmp_path):
        """The expected side of the comparison is the module, not the artifact."""
        doc, _, findings = _render(_numbers_doc(), tmp_path)
        assert not findings, findings
        doc["the_rendering_contract"]["formats_this_stage_can_render"] = ["str"]
        issues = tables.check_the_renderings(doc)
        assert any("formats_this_stage_can_render" in i for i in issues), issues

    def test_a_document_admitting_an_unrenderable_spec_is_refused(self, tmp_path):
        doc, _, findings = _render(_numbers_doc(), tmp_path)
        assert not findings, findings
        doc["the_rendering_contract"]["formats_declared_but_not_renderable"] = [
            "float9"]
        issues = tables.check_the_renderings(doc)
        assert any("would be rendered by guessing" in i for i in issues), issues


# ---------------------------------------------------------------------------
# Nothing is silently dropped
# ---------------------------------------------------------------------------

class TestNothingIsSilentlyDropped:
    def test_the_companion_keys_a_table_files_but_does_not_print_are_named(
            self, tmp_path):
        """Provenance the numbers stage filed beside a row.

        The paper does not print it, so the renderer has to say it dropped it: a
        reviewer comparing the artifact with the table would otherwise find fields
        in one and not the other with nothing explaining the difference.
        """
        doc, _, findings = _render(_numbers_doc(), tmp_path)
        assert not findings, findings
        filed = {name: block["companion_keys_filed_but_not_printed"]
                 for name, block in doc["tables"].items()}
        assert filed["blinding"] == ["arm_key", "declared_model_id", "n_records"]
        assert filed["panel_exclusions"] == ["family_dropped_with_it", "family_id",
                                             "variant"]
        assert filed["truncation"] == ["arm_key", "replay_outputs_sha256"]
        assert filed["vision_ablation_shift"] == []
        for name in CLAIM_TABLES:
            assert filed[name] == [], (
                f"{name}: the row-level 'format' key is a rendering spec and is "
                f"filed as one, not as a dropped companion")

    def test_every_declared_column_is_printed_in_its_filed_order(self, tmp_path):
        doc = _numbers_doc()
        _, files, findings = _render(doc, tmp_path)
        assert not findings, findings
        for name in sorted(doc["tables"]):
            table = doc["tables"][name]
            headers = [c["header"] for c in table["columns"]]
            head = _latex_header_row(files[f"tex/{name}.tex"])
            assert tables._split_latex_row(head) == headers, name
            md_rows = [line for line in files[f"markdown/{name}.md"].splitlines()
                       if line.startswith("| ")]
            assert tables._split_markdown_row(md_rows[0]) == headers, name

    def test_a_declared_column_a_row_has_no_value_for_is_a_finding(self, tmp_path):
        table = _table("tab:x", "A caption",
                       [_column("a", "A", "str"), _column("b", "B", "str")],
                       [{"a": "x"}])
        _, _, findings = _render_one(table, tmp_path)
        assert any("files no value for it" in f for f in findings), findings

    def test_a_table_with_no_rows_is_a_finding(self, tmp_path):
        table = _table("tab:x", "A caption", [_column("a", "A", "str")], [])
        _, _, findings = _render_one(table, tmp_path)
        assert any("filed with no rows" in f for f in findings), findings

    def test_a_table_with_no_columns_is_a_finding(self, tmp_path):
        table = _table("tab:x", "A caption", [], [{"a": 1}])
        _, _, findings = _render_one(table, tmp_path)
        assert any("filed with no columns" in f for f in findings), findings

    def test_a_table_missing_a_field_it_needs_is_reported_and_skipped(self, tmp_path):
        table = {"label": "tab:x", "caption": "A caption", "rows": []}
        doc = {"kind": numbers.KIND, "tables": {"broken": table}}
        _, files, findings = _render(doc, tmp_path)
        assert any("files no 'columns'" in f for f in findings), findings
        assert "tex/broken.tex" not in files, (
            "a table that cannot be rendered was rendered anyway")

    def test_a_malformed_table_is_a_finding_and_not_a_crash_in_any_of_the_censuses(
            self, tmp_path):
        """One definition of printable, used by the renderer, the agreement check
        and the census. A census that raises on a table the renderer already
        reported turns a finding into a traceback naming no table."""
        for broken in ({"label": "tab:x", "caption": "c", "columns": []},
                       {"label": "tab:x", "caption": "c", "rows": []},
                       {"caption": "c", "columns": [], "rows": []},
                       {"label": "tab:x", "caption": "c", "columns": ["not a dict"],
                        "rows": []},
                       {"label": "tab:x", "caption": "c",
                        "columns": [{"key": "a", "header": "A"}], "rows": []},
                       "not even a dict"):
            good = _all_specs_table()
            doc = {"kind": numbers.KIND, "tables": {"good": good, "broken": broken}}
            _, files, findings = _render(doc, tmp_path)
            assert findings, f"{broken!r} was accepted"
            assert any(f.startswith("broken:") for f in findings), (broken, findings)
            assert "tex/broken.tex" not in files
            assert "tex/good.tex" in files, (
                f"{broken!r} stopped the table beside it being rendered")

    def test_printable_is_one_predicate_and_not_three_copies(self):
        """A table that has all four fields but no rows IS printable: it renders,
        and the emptiness is a finding from the grid rather than a reason to skip
        it, so that a table nobody filled is reported as empty and not dropped."""
        assert tables.is_printable(_all_specs_table())
        assert tables.is_printable({"label": "tab:x", "caption": "c",
                                    "columns": [], "rows": []})
        assert not tables.is_printable({"label": "tab:x"})
        assert tables.unprintable_reasons("n", {"label": "tab:x"}) == [
            "n: files no 'caption', which a table has to have to be printable",
            "n: files no 'columns', which a table has to have to be printable",
            "n: files no 'rows', which a table has to have to be printable"]

    def test_no_cell_is_ever_left_empty(self, tmp_path):
        """An empty cell is indistinguishable from one the renderer dropped, which
        is the whole reason an absent value gets a mark."""
        doc = _numbers_doc()
        _, files, findings = _render(doc, tmp_path)
        assert not findings, findings
        for name in sorted(doc["tables"]):
            for index, row in enumerate(tables._latex_body_rows(
                    files[f"tex/{name}.tex"])):
                for position, cell in enumerate(tables._split_latex_row(row)):
                    assert cell != "", (name, index, position)


# ---------------------------------------------------------------------------
# The document re-derives and the files on disk match it
# ---------------------------------------------------------------------------

def _writable(tmp_path, doc, monkeypatch, upstream=0):
    """File a numbers document and stub the upstream gate.

    The gate asks the numbers stage to re-derive itself, which a synthesised
    document cannot survive; stubbing it is what lets these tests exercise writing
    without twenty-two real artifacts behind them.
    """
    source = tmp_path / "numbers.json"
    source.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    conclusion = "reproduced_exactly" if upstream == 0 else "differs"
    upstream_issues = [] if upstream == 0 else ["an upstream disagreement"]
    monkeypatch.setattr(tables, "numbers_re_derive",
                        lambda path: (upstream, conclusion, upstream_issues))
    return source, tmp_path / "out"


class TestTheFilesOnDiskMatchWhatTheNumbersFileProduces:
    def test_writing_then_verifying_reproduces_exactly(self, tmp_path, monkeypatch,
                                                       capsys):
        source, out = _writable(tmp_path, _numbers_doc(), monkeypatch)
        assert tables.main(["--write", "--numbers", str(source),
                            "--out-dir", str(out)]) == 0
        written = sorted(str(p.relative_to(out)) for p in out.rglob("*")
                         if p.is_file())
        assert len(written) == 26, written
        assert tables.RENDERINGS_NAME in written
        code, conclusion, issues = tables.verify(numbers_path=source, out_dir=out)
        assert code == 0, (conclusion, issues)
        assert conclusion == "reproduced_exactly"
        capsys.readouterr()
        assert tables.main(["--verify", "--numbers", str(source),
                            "--out-dir", str(out)]) == 0
        assert "PAPER TABLES: VERIFIED" in capsys.readouterr().out

    def test_a_table_edited_by_hand_after_rendering_is_caught(self, tmp_path,
                                                             monkeypatch):
        """The check a hash of the document alone would not make.

        The renderings file deliberately does not carry the outputs' hashes, so what
        catches a hand-edit is comparing the bytes on disk with a fresh rendering --
        a stronger comparison than two hashes of one file.
        """
        source, out = _writable(tmp_path, _numbers_doc(), monkeypatch)
        assert tables.main(["--write", "--numbers", str(source),
                            "--out-dir", str(out)]) == 0
        target = out / "tex" / "sign_transport.tex"
        original = target.read_text(encoding="utf-8")
        target.write_text(original.replace("0.0002", "0.9"), encoding="utf-8")
        code, conclusion, issues = tables.verify(numbers_path=source, out_dir=out)
        assert code == 1 and conclusion == "differs", (code, conclusion)
        assert any("sign_transport.tex" in issue
                   and "edited after it was rendered" in issue
                   for issue in issues), issues

    def test_a_table_the_numbers_file_no_longer_has_is_an_orphan(self, tmp_path,
                                                                 monkeypatch):
        source, out = _writable(tmp_path, _numbers_doc(), monkeypatch)
        assert tables.main(["--write", "--numbers", str(source),
                            "--out-dir", str(out)]) == 0
        (out / "tex" / "a_dropped_table.tex").write_text("% stale\n",
                                                         encoding="utf-8")
        code, _, issues = tables.verify(numbers_path=source, out_dir=out)
        assert code == 1
        assert any("a_dropped_table" in issue for issue in issues), issues

    def test_a_rendered_file_that_is_missing_is_caught(self, tmp_path, monkeypatch):
        source, out = _writable(tmp_path, _numbers_doc(), monkeypatch)
        assert tables.main(["--write", "--numbers", str(source),
                            "--out-dir", str(out)]) == 0
        (out / "markdown" / "blinding.md").unlink()
        code, _, issues = tables.verify(numbers_path=source, out_dir=out)
        assert code == 1
        assert any("missing a table the renderings describe" in i for i in issues), \
            issues

    def test_a_renderings_file_that_disagrees_with_a_fresh_one_is_caught(self,
                                                                        tmp_path,
                                                                        monkeypatch):
        source, out = _writable(tmp_path, _numbers_doc(), monkeypatch)
        assert tables.main(["--write", "--numbers", str(source),
                            "--out-dir", str(out)]) == 0
        path = out / tables.RENDERINGS_NAME
        filed = json.loads(path.read_text(encoding="utf-8"))
        filed["totals"]["n_tables"] = 11
        path.write_text(json.dumps(filed, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
        code, _, issues = tables.verify(numbers_path=source, out_dir=out)
        assert code == 1
        assert any(issue.startswith("totals:") for issue in issues), issues

    def test_a_number_that_moves_upstream_moves_the_print(self, tmp_path, monkeypatch):
        """The whole point of the stage, end to end."""
        doc = _numbers_doc()
        source, out = _writable(tmp_path, doc, monkeypatch)
        assert tables.main(["--write", "--numbers", str(source),
                            "--out-dir", str(out)]) == 0
        moved = json.loads(json.dumps(doc))
        moved["tables"]["sign_transport"]["rows"][1]["raw_p"] = 0.0136
        source.write_text(json.dumps(moved, indent=2, ensure_ascii=False) + "\n",
                          encoding="utf-8")
        code, _, issues = tables.verify(numbers_path=source, out_dir=out)
        assert code == 1, "the numbers moved and the print still verified"
        assert any("sign_transport.tex" in issue for issue in issues), issues

    def test_verifying_with_nothing_filed_exits_two(self, tmp_path):
        code, conclusion, issues = tables.verify(
            numbers_path=tables.NUMBERS_PATH, out_dir=tmp_path / "absent")
        assert code == 2 and conclusion == "not_filed"
        assert "--write" in issues[0]

    def test_verifying_with_no_numbers_to_render_from_exits_two(self, tmp_path):
        code, conclusion, issues = tables.verify(
            numbers_path=tmp_path / "absent.json", out_dir=tmp_path)
        assert code == 2 and conclusion == "no_numbers", (code, conclusion, issues)

    def test_an_unreadable_renderings_file_exits_one(self, tmp_path, monkeypatch):
        source, out = _writable(tmp_path, _numbers_doc(), monkeypatch)
        out.mkdir(parents=True, exist_ok=True)
        (out / tables.RENDERINGS_NAME).write_text("{not json", encoding="utf-8")
        code, conclusion, issues = tables.verify(numbers_path=source, out_dir=out)
        assert code == 1 and conclusion == "unreadable", (code, conclusion, issues)

    def test_writing_refuses_numbers_that_do_not_re_derive(self, tmp_path,
                                                           monkeypatch):
        """Writing is the moment a number enters the paper, so it is gated."""
        source, out = _writable(tmp_path, _numbers_doc(), monkeypatch, upstream=1)
        assert tables.main(["--write", "--numbers", str(source),
                            "--out-dir", str(out)]) == 1
        assert not out.exists(), "a table was printed from numbers nobody vouched for"

    def test_writing_refuses_a_rendering_with_findings(self, tmp_path, monkeypatch):
        doc = _numbers_doc()
        doc["tables"]["sign_transport"]["rows"][0]["mean"] = "not a number"
        source, out = _writable(tmp_path, doc, monkeypatch)
        assert tables.main(["--write", "--numbers", str(source),
                            "--out-dir", str(out)]) == 1
        assert not out.exists()

    def test_verify_and_write_are_mutually_exclusive(self):
        with pytest.raises(SystemExit) as exc:
            tables.main(["--verify", "--write"])
        assert exc.value.code == 2

    def test_verifying_is_the_default_when_neither_flag_is_given(self, tmp_path,
                                                                 monkeypatch):
        source, out = _writable(tmp_path, _numbers_doc(), monkeypatch)
        assert tables.main(["--numbers", str(source), "--out-dir", str(out)]) == 2

    def test_a_missing_numbers_file_exits_two_from_the_loader(self, tmp_path):
        with pytest.raises(SystemExit) as exc:
            tables.load_numbers(tmp_path / "absent.json")
        assert exc.value.code == 2

    def test_an_unreadable_numbers_file_exits_one_from_the_loader(self, tmp_path):
        broken = tmp_path / "numbers.json"
        broken.write_text("{not json", encoding="utf-8")
        with pytest.raises(SystemExit) as exc:
            tables.load_numbers(broken)
        assert exc.value.code == 1

    def test_a_numbers_file_with_no_tables_exits_one(self, tmp_path):
        source = tmp_path / "numbers.json"
        source.write_text(json.dumps({"tables": {}}), encoding="utf-8")
        with pytest.raises(SystemExit) as exc:
            tables.render_everything(source, tmp_path / "out")
        assert exc.value.code == 1

    def test_fatal_exits_with_the_code_it_is_given(self):
        for code in (1, 2, 3):
            with pytest.raises(SystemExit) as exc:
                tables.fatal("a failure", code)
            assert exc.value.code == code

    def test_three_is_documented_as_unreachable_here(self):
        """Its one input is a committed JSON file, so no checkout that can run this
        lacks a section of it -- and there is no media section that could be
        missing, which is what makes 3 reachable for the upstream stages."""
        assert "3  not reachable for this stage" in tables.__doc__
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", _relative(FILED_NUMBERS)],
            cwd=ROOT, capture_output=True, text=True)
        assert tracked.returncode == 0, _relative(FILED_NUMBERS)

    def test_the_documented_codes_are_the_ones_the_script_uses(self):
        source = (ROOT / "scripts" / "iter11_paper_tables.py").read_text(
            encoding="utf-8")
        for code in ("0", "1", "2", "3"):
            assert f"\n    {code}  " in source, (
                f"exit {code} is not documented in the module docstring")

    def test_it_reads_nothing_but_the_numbers_file(self, monkeypatch):
        """No artifact of the analysis, no media, no credentials, no network.

        The numbers stage already read twenty-two artifacts; a renderer that went
        back to them would be a second place the paper's numbers come from, and the
        two could differ.
        """
        opened = []
        original = Path.read_text

        def watched(self, *args, **kwargs):
            opened.append(str(self))
            return original(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", watched)
        doc, files, findings = tables.render_everything()
        assert not findings, findings
        assert doc and files
        assert opened == [str(tables.NUMBERS_PATH)], opened


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# The filed renderings
# ---------------------------------------------------------------------------

class TestTheFiledRenderings:
    def test_the_committed_renderings_verify(self):
        code, conclusion, issues = tables.verify()
        assert code == 0, (conclusion, issues)

    def test_the_committed_renderings_agree_with_themselves(self):
        assert tables.check_the_renderings(_filed_renderings()) == []

    def test_every_table_the_numbers_file_has_is_rendered(self):
        doc = _filed_renderings()
        numbers_doc = _numbers_doc()
        assert sorted(doc["tables"]) == sorted(numbers_doc["tables"])
        assert doc["inputs"]["sha256"] == numbers.sha256_file(FILED_NUMBERS)
        assert doc["inputs"]["n_tables"] == len(numbers_doc["tables"]) == 12

    def test_the_labels_are_the_numbers_files_and_not_ones_invented_here(self):
        doc = _filed_renderings()
        numbers_doc = _numbers_doc()
        for name in sorted(doc["tables"]):
            assert doc["tables"][name]["label"] == numbers_doc["tables"][name]["label"]
            assert doc["tables"][name]["source"] == \
                numbers_doc["tables"][name]["source"]

    def test_the_totals_are_the_sum_of_the_tables(self):
        doc = _filed_renderings()
        totals = doc["totals"]
        blocks = doc["tables"].values()
        assert totals["n_columns"] == sum(b["n_columns"] for b in blocks) == 94
        assert totals["n_rows"] == sum(b["n_rows"] for b in blocks) == 110
        assert totals["n_cells"] == sum(b["n_cells"] for b in blocks) == 643
        assert totals["n_rendered_files"] == 25
        assert totals["n_tex_files"] == 13
        assert totals["n_markdown_files"] == 12
        assert len(totals["output_paths"]) == 25

    def test_every_file_it_names_is_on_disk(self):
        doc = _filed_renderings()
        for relative in doc["totals"]["output_paths"]:
            assert (tables.OUT_DIR / relative).is_file(), relative

    def test_the_all_tables_file_inputs_every_table_exactly_once(self):
        text = (tables.OUT_DIR / "tex" / tables.ALL_TEX_NAME).read_text(
            encoding="utf-8")
        inputs = [line for line in text.splitlines() if line.startswith("\\input{")]
        assert len(inputs) == 12
        for name in sorted(_numbers_doc()["tables"]):
            assert f"\\input{{tables/tex/{name}}}" in inputs, name

    def test_the_input_paths_are_relative_to_where_main_tex_sits(self):
        r"""LaTeX resolves \input against the main document's directory, so a path
        without the ``tables/`` prefix would send it looking for a file this stage
        never wrote."""
        doc = _filed_renderings()
        assert "main.tex" in doc["the_rendering_contract"][
            "what_the_input_paths_are_relative_to"]
        text = (tables.OUT_DIR / "tex" / tables.ALL_TEX_NAME).read_text(
            encoding="utf-8")
        assert "\\input{tex/" not in text

    def test_the_committed_renderings_are_pure(self):
        banned = {"generated_at", "timestamp", "git_commit", "git_dirty",
                  "code_commit", "python", "host"}
        for key in _walk_keys(_filed_renderings()):
            assert key not in banned, key


def _walk_keys(node):
    if isinstance(node, dict):
        for key, value in node.items():
            yield key
            yield from _walk_keys(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk_keys(value)
