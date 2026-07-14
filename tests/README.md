# TabPFN Tests

This directory contains tests for the TabPFN project.

## Test Files

- `test_classifier_interface.py`: Tests for the TabPFNClassifier interface
- `test_regressor_interface.py`: Tests for the TabPFNRegressor interface 
- `test_utils.py`: Tests for utility functions
- `test_consistency.py`: Tests to ensure prediction consistency across code changes

## Model Consistency Testing

The consistency tests verify TabPFN models produce consistent predictions across code changes, ensuring:

1. Changes don't unexpectedly alter model behavior
2. Core algorithms remain stable and reproducible 
3. Intentional behavior changes are explicitly acknowledged

### How It Works

Tests use small, fixed datasets with reproducible random seeds to ensure consistency:

1. Creates a TabPFN model with fixed settings
2. Fits it to a reproducible dataset
3. Gets predictions using a standardized process
4. Compares predictions to previously saved reference values

### Platform Compatibility

The tests run inference in float64, which we expect to produce consistent
predictions across CPU architectures (x86 vs ARM) and operating systems (Linux,
macOS, Windows). All supported platforms therefore share a single reference set
(stored in `reference_predictions/`). Tests are skipped on platforms not listed in
`ENABLED_PLATFORMS`.

### CI Compatibility

We test against specific CI platform configurations:
- Linux, Windows, and macOS
- Python 3.10 and 3.14

### Commands

Check if your platform is CI-compatible:
```bash
python tests/test_consistency.py --print-platform
```

> **Important:** If creating reference values on a non-compatible platform, you must manually edit the platform metadata to match the closest CI platform. Otherwise, tests will fail in CI environments.

Run tests on a different platform:
```bash
FORCE_CONSISTENCY_TESTS=1 pytest tests/test_consistency.py
```

### Guidelines for Model Changes

Model changes should:
1. Be intentional and well-understood
2. Improve performance on standard benchmarks
3. Maintain backward compatibility when possible
4. Be clearly documented with evidence of improvement