#!/usr/bin/env python3
"""Invoke a TabPFN 2.5 model via a SageMaker Runtime endpoint.

This module demonstrates how to:

* Connect to an existing SageMaker endpoint that serves a TabPFN 2.5 model.
* Serialize tabular training and test data from NumPy arrays.
* Send an inference request and parse the JSON response.

Typical usage:

    1. Make sure your SageMaker endpoint is running and note its name.
    2. Configure environment variables:

        export SAGEMAKER_ENDPOINT_NAME="your-endpoint-name"
        export AWS_DEFAULT_REGION="us-east-1"  # or your region

       Optionally, configure AWS credentials via environment variables
       (or rely on the default AWS credential chain):

        export AWS_ACCESS_KEY_ID="..."
        export AWS_SECRET_ACCESS_KEY="..."

    3. Call `invoke_tabpfn` from your own code, or run this script
       directly to execute a toy end-to-end example:

        python example.py

The `main()` function uses scikit-learn's breast cancer dataset as a
sanity check that the endpoint is correctly configured. For production
usage, import `invoke_tabpfn` and pass your own NumPy arrays.
"""

import json
import os
from typing import Any, Literal, Optional

import boto3
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Name of the SageMaker endpoint that is hosting the TabPFN model.
# In most real setups you will want to set this explicitly:
#   export SAGEMAKER_ENDPOINT_NAME="your-endpoint-name"
SAGEMAKER_ENDPOINT_NAME = os.environ.get("SAGEMAKER_ENDPOINT_NAME", "tabpfn")
if not SAGEMAKER_ENDPOINT_NAME:
    raise ValueError("SAGEMAKER_ENDPOINT_NAME environment variable must be set")


# ---------------------------------------------------------------------------
# AWS credentials
# ---------------------------------------------------------------------------
# Credentials are resolved as follows:
#
#   1. Explicit environment variables below (if both are set).
#   2. Otherwise, the standard AWS credential chain is used
#      (e.g. ~/.aws/credentials, environment, IAM role, etc.).
#
# If you are running this from a SageMaker Notebook Instance, SageMaker Studio,
# or an environment with an attached IAM role, you can typically omit the
# explicit environment variables.

# Initialize the SageMaker Runtime client used to invoke the endpoint.
sagemaker_runtime = boto3.client("sagemaker-runtime")


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def prepare_tabpfn_request(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    task: Literal["classification", "regression"],
    model_params: dict[str, Any],
    predict_params: dict[str, Any],
) -> tuple[str, str]:
    """Serialize TabPFN input data and parameters into a JSON request body.

    This utility converts NumPy arrays and configuration dictionaries into a
    JSON-encoded payload suitable for invoking a TabPFN 2.5 model hosted on
    a SageMaker endpoint. It does not alter or validate data — it simply
    prepares the transport format expected by the server.

    For larger datasets, we reconnect preparing and sending the request data
    using the multipart/form-data content type, where datasets are compressed
    and serialized using e.g. PyArrow and Parquet.

    Args:
        x_train:
            Training features of shape `(n_train_samples, n_features)`.
            These samples are used by TabPFN to condition its posterior
            predictor.
        y_train:
            Training targets of shape `(n_train_samples,)`. For
            classification, these are label indices or class IDs.
        x_test:
            Test features of shape `(n_test_samples, n_features)` for
            which predictions should be produced.
        task:
            Task type, e.g. `"classification"` or `"regression"`. This
            is forwarded to the endpoint as-is.
        model_params:
            Dictionary of model initialization parameters forwarded to
            TabPFN on the server side.
        predict_params:
            Dictionary of prediction-time parameters forwarded to the
            predictor on the server side.

    Returns:
        A tuple `(json_body, content_type)` where:

        * `json_body`:
            A JSON string containing the full request body with the
            following structure:

                {
                    "task": "...",
                    "data": {
                        "encoding": "json",
                        "x_train": [...],
                        "y_train": [...],
                        "x_test": [...]
                    },
                    "model_params": { ... },
                    "predict_params": { ... }
                }

        * `content_type`:
            The HTTP Content-Type header string (always
            `"application/json"`).

    Notes:
        This function is intentionally serialization-only. It does not
        perform validation or type-checking beyond the basic `tolist()`
        conversion of the input arrays. Any interpretation or validation
        happens entirely on the endpoint side.
    """
    data_payload = {
        "encoding": "json",
        "x_train": x_train.tolist(),
        "y_train": y_train.tolist(),
        "x_test": x_test.tolist(),
    }

    request_body = {
        "task": task,
        "data": data_payload,
        "model_params": model_params,
        "predict_params": predict_params,
    }

    return json.dumps(request_body), "application/json"


def invoke_tabpfn(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    task: Literal["classification", "regression"],
    model_params: Optional[dict[str, Any]] = None,
    predict_params: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Invoke the TabPFN SageMaker endpoint with tabular data.

    This helper prepares the payload for a TabPFN 2.5 model, sends it to
    the configured SageMaker endpoint, and parses the JSON response.

    Args:
        x_train:
            Training features of shape `(n_train_samples, n_features)`.
        y_train:
            Training targets of shape `(n_train_samples,)`. For
            classification, this is typically a 1D array of class labels.
        x_test:
            Test features of shape `(n_test_samples, n_features)` for
            which predictions should be obtained.
        task:
            Task type for TabPFN. Supported values are typically
            `"classification"` or `"regression"`.
        model_params:
            Optional TabPFN model initialization parameters. If `None`,
            the following default configuration is used:

                {
                    "n_estimators": 8,
                    "random_state": 42,
                    "fit_mode": "fit_preprocessors",
                    "inference_precision": "auto",
                    ... # Additional model parameters
                }

            Any values provided here are forwarded to the model on the
            server side (subject to the deployment's configuration).

            The optional `model_path` parameter can be used to specify a
            pre-trained checkpoint. Available checkpoint files include:

            **Classification checkpoints:**
            - `tabpfn-v2.5-classifier-v2.5_default.ckpt`
            - `tabpfn-v2.5-classifier-v2.5_default-2.ckpt`
            - `tabpfn-v2.5-classifier-v2.5_large-features-L.ckpt`
            - `tabpfn-v2.5-classifier-v2.5_large-features-XL.ckpt`
            - `tabpfn-v2.5-classifier-v2.5_large-samples.ckpt`
            - `tabpfn-v2.5-classifier-v2.5_real-large-features.ckpt`
            - `tabpfn-v2.5-classifier-v2.5_real-large-samples-and-features.ckpt`
            - `tabpfn-v2.5-classifier-v2.5_real.ckpt`
            - `tabpfn-v2.5-classifier-v2.5_variant.ckpt`

            **Regression checkpoints:**
            - `tabpfn-v2.5-regressor-v2.5_default.ckpt`
            - `tabpfn-v2.5-regressor-v2.5_low-skew.ckpt`
            - `tabpfn-v2.5-regressor-v2.5_quantiles.ckpt`
            - `tabpfn-v2.5-regressor-v2.5_real-variant.ckpt`
            - `tabpfn-v2.5-regressor-v2.5_real.ckpt`
            - `tabpfn-v2.5-regressor-v2.5_small-samples.ckpt`
            - `tabpfn-v2.5-regressor-v2.5_variant.ckpt`

            Example usage:

                model_params = {
                    "model_path": "tabpfn-v2.5-classifier-v2.5_real.ckpt",
                    "n_estimators": 8,
                }
        predict_params:
            Optional dictionary of prediction-time parameters (for
            example, settings that influence prediction behavior). If
            `None`, an empty dictionary is sent and endpoint defaults
            are used.

    Returns:
        A dictionary containing the parsed JSON response from the
        SageMaker endpoint.

        The exact schema may depend on the deployed model and task, but
        the endpoint typically returns at least the following keys:

        * `prediction`:
            A list of float values representing the predictions for the
            request. For a regression deployment, each element is the
            predicted continuous value for the corresponding test sample,
            for example:

                {
                    "prediction": [
                        0.9990007281303406,
                        -0.000359066209057346,
                        -0.0010218550451099873,
                        0.9995852708816528,
                        0.9994749426841736,
                        ...
                    ],
                    "params": { ... }
                }

            The length of this list is generally aligned with
            `x_test.shape[0]`, but may vary depending on the task and
            endpoint-specific behavior.

            For classification deployments, the structure of this field
            may differ (for example, class probabilities or scores per
            class). Consult your endpoint's documentation for details.
        * `params`:
            A dictionary of the effective model and inference parameters
            used for this call. This typically mirrors (or extends) the
            configuration of the TabPFN model on the server side. For
            example (truncated):

                "params": {
                    "fit_mode": "fit_preprocessors",
                    "n_estimators": 4,
                    "random_state": 42,
                    ...
                }

            This is useful for debugging and for verifying which
            hyperparameters were applied during inference.

    Raises:
        json.JSONDecodeError:
            If the endpoint response cannot be parsed as valid JSON. In
            this case, the raw response body is printed to stdout before
            the exception is re-raised.

    Examples:
        Basic usage with scikit-learn's breast cancer dataset:

            from sklearn.datasets import load_breast_cancer
            from sklearn.model_selection import train_test_split

            X, y = load_breast_cancer(return_X_y=True)
            x_train, x_test, y_train, y_test = train_test_split(
                X, y, test_size=0.5, random_state=42
            )

            result = invoke_tabpfn(x_train, y_train, x_test)
            preds = result["prediction"]
            assert len(preds) == x_test.shape[0]
    """
    if model_params is None:
        model_params = {
            "n_estimators": 8,
            "random_state": 42,
            "fit_mode": "fit_preprocessors",
            "inference_precision": "auto",
        }

    if predict_params is None:
        predict_params = {}

    # Prepare JSON request body.
    json_body, content_type = prepare_tabpfn_request(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        task=task,
        model_params=model_params,
        predict_params=predict_params,
    )

    # Invoke the endpoint.
    print(f"Invoking SageMaker endpoint: {SAGEMAKER_ENDPOINT_NAME}")
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=SAGEMAKER_ENDPOINT_NAME,
        ContentType=content_type,
        Body=json_body.encode("utf-8"),
    )

    # Read and parse the response.
    response_body = response["Body"].read()
    status_code = response["ResponseMetadata"]["HTTPStatusCode"]
    print(f"Response status code: {status_code}")

    try:
        result: dict[str, Any] = json.loads(response_body.decode("utf-8"))
        return result
    except json.JSONDecodeError as e:
        # For debugging, print the raw response if JSON parsing fails.
        print(f"Error parsing JSON response: {e}")
        print(f"Raw response: {response_body.decode('utf-8')}")
        raise


# ---------------------------------------------------------------------------
# Example script entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run a minimal end-to-end example against the TabPFN endpoint.

    This function is intended as a quick sanity check that:

    * The SageMaker endpoint is active and reachable.
    * The request/response format is correctly wired up.
    * Basic predictions can be obtained without writing additional code.

    The steps performed are:

    1. Load the breast cancer dataset from scikit-learn.
    2. Split it into training and test sets (50/50 split).
    3. Call `invoke_tabpfn` with `task="classification"` and a small
       default model configuration.
    4. Print the raw JSON response to stdout.

    Notes:
        For real-world use cases, you will typically not call `main()`.
        Instead, import `invoke_tabpfn` and pass your own domain-specific
        `x_train`, `y_train`, and `x_test` arrays, optionally customizing
        `model_params` and `predict_params` as needed.
    """
    print("Loading breast cancer dataset...")
    X, y = load_breast_cancer(return_X_y=True)
    x_train, x_test, y_train, _y_test = train_test_split(
        X, y, test_size=0.5, random_state=42
    )

    print(f"Training set: {x_train.shape[0]} samples, {x_train.shape[1]} features")
    print(f"Test set: {x_test.shape[0]} samples")

    result = invoke_tabpfn(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        task="classification",
        model_params={
            "n_estimators": 8,
            "random_state": 42,
            "fit_mode": "fit_preprocessors",  # This is the default value
        },
        predict_params={},
    )

    # Optionally, you may compare its performance in terms of classification
    # accuracy or other metrics using y_test and y_pred from the result.

    print("\n=== Prediction Results ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
