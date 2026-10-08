# Provenance and reuse

This repository was prepared from the user-supplied
`Machine_Learning_from_First_Principles.docx`, identified in that document as
the first edition, 2026. The original document is not bundled.

`book_code/companion_classical.py`, `companion_deep.py`, `companion_rl.py` and
`companion_investigation.py` preserve the complete Appendix A1–A4 Python
listings. `book_code/chapter03_array_example.py` preserves the complete Python
array fragment in Chapter 3. Only document formatting was removed; indentation,
comments and code text were retained. `book_code/source_manifest.json` records
the extraction method and SHA-256 checksums. Shell setup commands printed in
the appendix are represented in this repository's setup instructions.

The `mlfirst/` teaching modules, CLI, tests, guides, notebooks and project briefs
were added for this companion. Where a teaching function reuses original code,
its import makes that relationship explicit. Lesson solutions are for the
repository's exercises; they are not a transcription of the book's complete
problem and solution sections. The guides summarize and extend concepts rather
than reproducing the book text or artwork.

One numerical refinement is explicit: the teaching tree/stump implementation
handles adjacent representable feature values and very large opposite-sign
values with a safe split threshold. The original appendix midpoint calculation
is preserved as printed and can produce an invalid split at those extremes.
The teaching implementation rejects missing group identifiers before splitting.

The supplied document did not establish a redistribution license for the book
listings. This package therefore does not assign a blanket open-source license
to those listings or assert third-party rights. A rights holder can add the
appropriate license before public redistribution. NumPy and optional notebook
dependencies retain their own licenses and are not vendored in this ZIP.
