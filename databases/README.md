# mc_ni_v2.034.tdb

Open MatCalc thermodynamic database for the Ni-rich corner, used unchanged.

- **Provider**: MatCalc Engineering GmbH, <https://matcalc.at>
- **Version**: 2.034
- **Licence**: Open Database License (ODbL) 1.0
- **SHA-256**: `4bbb41531b58e3def2ea958802ed9a2917f15e24aa961c33e679c8f94c027ac4`
- **Size**: 390 014 bytes
- **Elements used here**: Ni, Co, Cr, Al, Mo, Ti, C, Va (the database itself
  contains 22 elements and 98 phases)

The file is redistributed unmodified under the ODbL. Any derived database or
publicly shared modification must be distributed under the same licence and
must retain attribution. The calculations in this repository do not modify the
database; the phase list, temperature grid and point density are set in
`configs/base.yaml`.

To verify the file:

```bash
sha256sum databases/mc_ni_v2.034.tdb
```

**Note.** The database file itself is not redistributed in this repository for
licence reasons. Download `mc_ni_v2.034.tdb` from MatCalc
(<https://matcalc.at>, Open Database License) and place it at this location
before running the pipeline; verify it against the SHA-256 above.
