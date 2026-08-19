# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project, at least loosely, adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This file contains the unreleased changes to the codebase. See CHANGELOG.md for
the released changes.

## Unreleased
### Changed
- Document Apple Silicon via native ``linux/arm64`` containers (`nanograv/ng20`) instead of claiming PINT cannot run there; Rosetta/osx-64 remains an alternative
- MJD string formatting uses enough fractional digits for the platform's `numpy.longdouble` precision (needed for IEEE binary128 on Linux aarch64)
- FDJUMPDM sign convention: a positive FDJUMPDM now adds a positive DM (and delay) on selected TOAs, matching Tempo2.
### Added
- Time-domain solar wind GP noise components: ridge, squared-exponential, Matérn, and quasi-periodic kernels
- Documentation page explaining the time-domain solar wind noise model, its interpolation basis, and how it differs from the Fourier-basis noise models
### Fixed
- Precision tests / MJD string length on platforms with true `float128` `longdouble` (e.g. Linux aarch64): stop truncating via a fixed `U30` dtype and emit enough digits for `str(longdouble)`
- Binary-convert roundtrip tests and START/FINISH MJD checks: stop requiring bit-identical `longdouble`/`Time` equality on binary128 (TASC↔T0 goes through float64; MJDParameter stores via jd1/jd2)
- Jodrell Bank Mark II sites (``jbmk2`` / ``jbmk2roach`` / ``jbmk2dfb``) now follow TEMPO2's clock routing (equivalent to ``jbafb`` / ``jbroach`` / ``jbdfb``) instead of the default empty TEMPO clock path.
### Removed
