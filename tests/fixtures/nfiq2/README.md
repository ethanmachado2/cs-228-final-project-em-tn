# NFIQ 2 2.3.0 captured outputs (Task 1 spike)

Captured in the `fpv` container (ubuntu:24.04 arm64, official rpos12 arm64 package) with `stdin=/dev/null`.
Each case has `<case>.cmd` (argv), `<case>.stdout`, `<case>.exitcode` and, when non-empty, `<case>.stderr`.

| Case | Input | Notes |
|---|---|---|
| `score_plain` | SOCOFing Real 100 L index, 2x bicubic, 500 PPI PNG | bare score `34` |
| `score_actionable` | same, `-a` | CSV header + row with actionable feedback |
| `score_actionable_verbose` | same, `-a -v` | adds native quality measures |
| `score_actionable_two_files` | scored image + blank | one row per file, mixed success/failure |
| `score_force_nodpi` | 2x PNG without PPI metadata, `-F -a` | `-F` assumes 500 PPI, `Resampled=0`, same score |
| `prompt_nodpi_no_force` | same without `-F` | interactive PPI prompt (why `-F` is mandatory) |
| `fail_native_socofing` | native 96x103 SOCOFing PNG | `QualityScore=NA`, FDA error, exit 0 |
| `fail_blank` / `fail_plain_blank` | all-white image | `NA`, "All image rows appear to be blank", exit 0 |
| `fail_plain_native` | native SOCOFing, no `-a` | `Error: ...` line, exit 0 |
| `fail_missing_file` | nonexistent path | `NA`, "ObjectDoesNotExist", exit 0 |
| `help`, `no_args` | `-h` / no args | banner incl. `NFIQ 2: 2.3.0`; `-h` exits 1 |
