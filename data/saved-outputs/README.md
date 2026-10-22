# Saved outputs

The studies' results, packed as deterministic tar archives compressed with xz. Member paths are relative to the release root, so `make unpack` (or `python tools/unpack.py`) puts every file where the study code reads it and checks each SHA-256 against `data/PACKED-FILES.json`. Text records inside the archives had personal strings replaced, as listed in `RELEASE-CHANGES.md`.

| Archive | Files | Packed (MB) | Unpacked (MB) | Contents |
|---|---|---|---|---|
| `bridge-001.tar.xz` | 24 | 1.5 | 4.2 | Bridge 001 (the first joint experiment): analyses, reports, verification (raw fit and test arrays not included) |
| `bridge-002.tar.xz` | 34 | 0.9 | 3.6 | Bridge 002: analyses, audit reproductions, verification |
| `bridge-003.tar.xz` | 32 | 1.2 | 2.1 | Bridge 003: analyses, audit reproductions, verification |
| `bridge-004.tar.xz` | 71 | 2.9 | 3.9 | Bridge 004: analyses, qualification and verification records |
| `bridge-005.tar.xz` | 23 | 12.4 | 13.2 | Study 1 (Bridge 005): analyses, verification, the four test arrays the post hoc baseline reads |
| `bridge-006.tar.xz` | 20 | 1.2 | 1.3 | F4/F7 follow-up (Bridge 006): analyses and records |
| `bridge-007.tar.xz` | 848 | 3.0 | 4.5 | Study 2 (Bridge 007): primary and secondary analyses, independent verification, status records |
| `bridge-008.tar.xz` | 11 | 2.9 | 3.3 | Study 3 (Bridge 008): analyses and verification |
| `bridge-009.tar.xz` | 19 | 0.0 | 0.0 | Study 4 (Bridge 009): screen records and verification |
| `bridge-010.tar.xz` | 6 | 0.0 | 0.1 | Study 5 (Bridge 010): analyses and verification |
| `bridge-011.tar.xz` | 11 | 2.1 | 2.6 | Study 6 (Bridge 011): analyses, development summary and verification |
| `bridge-012.tar.xz` | 15 | 0.4 | 0.6 | Study 7 (Bridge 012): every record pulled back from the cloud run |
| `routing.tar.xz` | 46 | 0.2 | 0.7 | routing experiments R1-R2: summaries |
| `transport.tar.xz` | 118 | 0.4 | 6.7 | transport experiments T1-T5: summaries |

Total packed size: 29.2 MB. Rendered figures are not included; `make figures` draws them from these outputs.
