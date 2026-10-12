# Reproducible visual examples

`run_example.py` constructs exactly one `N=2,d=7` basis. It selects the
interior-owner basis row with the widest nonzero-curl tetrahedral support,
breaking ties by the smallest row ID. A nonzero curl coefficient selects a
horizontal slice through its tetrahedron's barycenter. The default slice
has 17 points per axis; no randomized input or fitted measurement is used.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python numerical_case/run_example.py --output numerical_case/results/example-1
python numerical_case/plot_results.py \
  --input numerical_case/results/example-1 \
  --reference numerical_case/reference \
  --output numerical_case/results/figures-1
```

Both output directories must be new. A repeated command refuses to overwrite
the previous run. The plotter verifies the run's content hashes and the fixed
reference hashes before rendering. Each run retains its parameters, source
hash, package/certificate hashes, versions, thread settings and output hashes.
The plotting manifest binds the plots to that run and those references.

`reference/` holds the compact, previously computed paper-case summaries and
complete sparsity histograms, with immutable public source links and hashes.
They supply the degree-six storage growth and degree-six/seven sparsity CDF;
their runtime observations are not used. New geometry is computed by this
software. No costly grid/degree sweep is hidden in the plot command.

`figures/` contains selected SVG and PDF exports for the README and LaTeX.
`summaries/` retains the selected example's compact summary, geometry, slice
and plotting manifest. Full bases, normal forms and execution logs stay in
ignored `results/`. The scripts never alter the distributed figures or
summaries. To update selected outputs, first compute and inspect a fresh run,
then explicitly select its compact files for version control.

To redraw the distributed compact example without constructing its basis:

```bash
python numerical_case/plot_results.py \
  --input numerical_case/summaries \
  --reference numerical_case/reference \
  --output numerical_case/results/selected-figures-1
```
