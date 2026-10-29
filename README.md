# tracula-braincharts

Local, containerised application of the `BLR_fa_lifespan_24K_19sites` white-matter fractional anisotropy
(FA) normative model ([Cirstian et al., bioRxiv 2024](https://www.biorxiv.org/content/10.1101/2024.12.11.627897v1.full);
weights from [braincharts](https://github.com/predictive-clinical-neuroscience/braincharts)) to diffusion
data processed with [Tracula](https://surfer.nmr.mgh.harvard.edu/fswiki/Tracula).

The goal is to get the same results you would get uploading the data to [PCNportal](https://pcnportal.dccn.nl/)

Mean FA is computed on the [TBSS](https://fsl.fmrib.ox.ac.uk/fsl/docs/diffusion/tbss.html) skeleton across the 48 JHU ICBM-DTI-81 white-matter labels, with site adaptation using local healthy controls.

The tool has four subcommands:

- `extract`: per subject, compute FA features plus provenance and QA metrics
- `collate`: builds the adaptation and test CSVs and the cohort QA tables
- `score`: normative z-scores and centiles for the test subjects
- `x-val`: cross-validated z-scores and adaptation QA for the controls

Status: under development. Usage and outputs will be documented here as each subcommand is implemented.

## License

BSD 2-Clause (see `LICENSE`). The model files are not part of this repository; they are downloaded from
braincharts (GPL-3.0) when the image is built.
