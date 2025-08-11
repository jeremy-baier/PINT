# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project, at least loosely, adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This file contains the unreleased changes to the codebase. See CHANGELOG.md for
the released changes.

## Unreleased
### Changed
- Simulation functions no longer subtract the residual mean by default.
- `cmwavex_setup` now takes the chromatic index as an optional argument.
### Added
- `WidebandTOAResiduals.calc_wideband_resids()` and `TimingModel.scaled_wideband_uncertainty()` methods
- New abstract base classes `WhiteNoiseComponent` and `CorrelatedNoiseComponent` (changed the type hierarchy of `NoiseComponent`s)
- `get_dm_noise_basis()` and `get_wideband_noise_basis()` methods in `CorrelatedNoiseComponent`
- `noise_model_dm_designmatrix()`, `noise_model_wideband_designmatrix()`, `full_wideband_designmatrix()`, `dm_designmatrix()`, `wideband_designmatrix()` methods in `TimingModel`
- Type hints in `pint.fitter`
- PLSWNoise: a Fourier basis stochastic solar wind model. See Hazboun et al. 2022 for details.
- Explicitly specify the Tspan for power-law GP noise parameters (TN*TSPAN)
- Parallel execution and work stealing in CI tests
- Make `WidebandTOAResiduals.noise_resids` available as a `@property`
### Fixed
- `TimingModel.total_dm()` now returns zero when no DM component is present in the model.
- Made `TimingModel.toa_covariance_matrix()` not explicitly dependent on `ScaleToaError`
- Simulate DM noise in wideband TOAs correctly (Moved the functionality of `update_fake_dms()` to `make_fake_toas()`)
- TCB <-> TDB conversion for power-law GP noise parameters.
- TN*C parameter are now `intParameters`
- Bug in `Fitter.plot()`
- `np.NaN` -> `np.nan`
- When converting between binary models, some parameters may not be visible
- When converting from ELL1H, if only H3 is set should not include M2,SINI in output
- `find_empty_masks` can now handle SWX models
- `photonphase` crash when launched with `--absphase --polycos`
### Removed
