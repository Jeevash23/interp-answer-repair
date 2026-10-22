# Errata to the Bridge 010 (Study 5) reports

Added 9 October 2026. The study's frozen files, results and the reports below are unchanged; this
note corrects two statements in `README.md`.

1. **README.md, "Answer" paragraph.** "No pooled ratio is below 1" is incorrect. Four point
   estimates are below 1, all for the mismatched control: the F ratio 0.997 under both
   references and the A ratios 0.992 (zero) and 0.998 (fit mean), as the table directly below
   that paragraph shows. The correct statement is that **no adjusted ratio interval lies below 1**:
   the mismatched and natural F intervals span 1, and the random F intervals lie above 1. The
   conclusion is unchanged: wording directions are not discounted on average relative to these
   same-geometry controls. `RESULTS.md`, `paper/review/STUDY5-RESULTS.md` and the manuscript
   (Section 5.8, Tables 4 and 18) state this correctly.
2. **README.md, "Runtime".** "after the controlling [redacted] session stopped" means that the
   interactive session that started the run ended during the twelfth condition. The inference
   process itself ran to completion unchanged, and no condition was restarted or re-measured.

`README.md` is left as written because it is part of the retained study record, hashed in
`provenance/release-manifest.json`.
