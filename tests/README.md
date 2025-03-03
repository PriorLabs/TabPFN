# TabPFN Tests

This directory contains test files for the TabPFN project.

## Test Files

- `test_classifier_interface.py`: Tests for the TabPFNClassifier interface
- `test_regressor_interface.py`: Tests for the TabPFNRegressor interface
- `test_utils.py`: Tests for utility functions
- `test_consistency.py`: Tests to ensure prediction consistency across code changes

## Model Consistency Testing

The `test_consistency.py` file contains tests that verify TabPFN models produce consistent predictions across code changes. These tests help ensure that:

1. Changes to the codebase don't unexpectedly alter model behavior
2. Core algorithms remain stable and reproducible 
3. Any intentional changes to model behavior are explicitly acknowledged and verified

### How It Works

The tests are organized into two classes:

1. `TestModelConsistency`: Verifies that TabPFN models produce consistent predictions
   - Uses standardized datasets (Iris, Breast Cancer, Boston Housing, Diabetes)
   - Creates models with fixed random seeds
   - Computes statistical summaries of model predictions
   - Compares statistics against known reference values

2. `TestStatsRobustness`: Verifies that our statistical approach correctly detects changes
   - Confirms that statistics change when input data changes significantly
   - Confirms that statistics change when model configuration changes
   - Tests both classifiers and regressors

### Platform-Specific Consistency

TabPFN models can produce slightly different predictions across platforms due to:
- Different CPU architectures (x86 vs ARM)
- Different operating systems (Linux, macOS, Windows)
- Different underlying BLAS/LAPACK implementations
- Compiler-specific floating-point optimizations

Instead of relying on exact hash values, we use a statistical approach that:
1. **Computes distribution statistics** - min, max, mean, median, standard deviation
2. **Captures distribution shape** - percentiles at p10, p25, p75, p90
3. **Verifies output shape** - ensures the dimensionality remains consistent

The test suite is designed with these features:
1. **Platform-specific reference values** - Reference stats are generated on a specific
   platform and Python version defined at the top of the test file
2. **Platform-specific test execution** - Tests only run on matching platforms using
   pytest's skipif decorator
3. **Double precision stability** - Uses scikit-learn's 16 decimal precision mode
   (`USE_SKLEARN_16_DECIMAL_PRECISION = True`)
4. **Reasonable tolerance** - Allows small variations (1-3% relative tolerance) in statistics
   to account for unavoidable precision differences
5. **Fixed random seeds** - Ensures reproducibility within the same platform

This approach:
- Catches real regressions or changes in model behavior
- Avoids false positives from platform-specific variations
- Acknowledges the reality that exact numerical equivalence across all platforms is not practical

### When a Consistency Test Fails

When a consistency test fails, it means the model's behavior has changed. The failure message includes:

1. The expected statistic that differs from the reference
2. The actual statistic from your current code
3. The difference between them
4. Instructions for how to proceed

### Managing Intentional Model Changes

If you're making intentional changes to the model that should improve performance:

1. **Verify the improvement:**
   - Benchmark the new model against the old one on standard datasets
   - Document the performance improvements with metrics
   - Consider running on the established AutoML benchmark suite

2. **Check your platform:**
   - Reference stats in the test file are for a specific platform (see `REFERENCE_PLATFORM` 
     and `REFERENCE_PYTHON_VERSION` at the top of the test file)
   - Either update the stats on the same platform or change the platform settings

3. **Update the reference statistics:**
   ```python
   # Run this on the reference platform defined in the test file
   python -c "from tests.test_consistency import update_reference_stats; update_reference_stats()"
   ```

4. **Document the changes in your PR:**
   - Explain what changes were made to the model
   - Provide benchmark results showing the improvement
   - Explain why this change is beneficial
   - Note which platform the reference values were generated on

5. **Update the REFERENCE_STATS dictionary:**
   - Replace the values in `test_consistency.py` with the newly generated statistics
   - If using a different platform, update the `REFERENCE_PLATFORM` and `REFERENCE_PYTHON_VERSION` variables
   - Include these changes in your PR

### Rules for Model Changes

Model changes should:

1. Be intentional and well-understood (not accidental side effects)
2. Improve overall performance on standard benchmarks
3. Not break backward compatibility for users without good reason
4. Be thoroughly documented with evidence of improvement

By following these guidelines, we ensure that TabPFN evolves in a controlled and beneficial way, providing users with a reliable and continually improving foundation model.