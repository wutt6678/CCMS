<!--
  Rendered by scripts/iter11_paper_tables.py -- do not edit.
  Table 'evidence_binding'.
  Every cell is derived from paper/numbers/iteration_11_paper_numbers.json
    sha256 2aaf05f47f4a2257611f471b989b5730a3ada466a5b1dcece225c03170a2964f
  whose values are read out of the filed artifact
  'media_manifest' by scripts/iter11_paper_numbers.py.
  Re-render with: python scripts/iter11_paper_tables.py --write
-->

### What is bound by hash: the media, which the repository does not hold, and the commands that re-derive the rest

| Claim | Value | Filed in |
| :--- | :--- | :--- |
| media files bound by hash | 3034 | media_manifest |
| media bytes bound by hash | 3698064267 | media_manifest |
| media roll-up sha256 | 49637e1c | media_manifest |
| scope of that binding | whole_media_tree | media_manifest |
| panel images the media manifest references | 100 | media_manifest |
| panel images referenced but absent from the manifest | 0 | media_manifest |
| media root | data/media | media_manifest |
| what re-derives the closeout manifest | python scripts/iter11_closeout_evidence_manifest.py --verify | procedure |
| what executes every gate the closeout points at | python scripts/iter11_closeout_evidence_manifest.py --deep | procedure |
| what re-derives every number in this table | python scripts/iter11_paper_numbers.py --verify | procedure |
| why this table quotes no count from the closeout manifest | outputs/iteration_11/closeout/iteration_11_evidence_manifest.json binds this file, so this file cannot bind it: two documents carrying each other's hash have no fixed point, and neither can be re-filed without invalidating the other. The closeout is cited here by the command that re-derives it instead of by a count of what it bound | this stage's own input list |

*Note.* The media are not in this repository: data/media is gitignored apart from twenty individually
negated source images, so no manifest committed here can bind their bytes directly and the media
manifest binds them by hash instead. That is why a checkout without the images can still verify
everything committed, and why the verifiers report the media section as not verifiable here rather
than as a failure. The anonymous package ships the media manifest and says so; it does not ship 3.7
GB of images. The committed evidence is bound by the closeout manifest, which is cited here as a
command rather than as a hash for the reason the last row of this table gives.

`tab:evidence_binding` -- source `media_manifest`, rendered from `paper/numbers/iteration_11_paper_numbers.json`.
