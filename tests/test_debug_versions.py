from __future__ import annotations

import io
from contextlib import redirect_stdout

from tabpfn.misc.debug_versions import display_debug_info


def test_display_debug_info():
    """Test that display_debug_info runs without errors and produces output."""
    # Capture the output
    captured_output = io.StringIO()

    # Run the function and capture stdout
    with redirect_stdout(captured_output):
        display_debug_info()

    # Get the output
    output = captured_output.getvalue()

    # Check that we got some output
    assert output, "display_debug_info() should produce output"

    # Check for expected sections in the output
    assert "Collecting system and dependency information" in output
    assert "Dependency Versions:" in output
    assert "-" * 20 in output  # The separator line

    # Check that some key dependencies are mentioned
    # (tabpfn should always be present)
    assert "tabpfn:" in output

    # Make sure it doesn't crash and produces reasonable output
    assert len(output) > 100  # Should have substantial output
