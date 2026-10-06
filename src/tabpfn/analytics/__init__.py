#  Copyright (c) Prior Labs GmbH 2026.

"""Usage analytics for TabPFN estimators.

Estimator methods decorated with `log_usage` log a usage event each time they
are called: which method ran, whether it succeeded, how long it took and the
size of its data. Only the shape of the data is read, never its values.

Only the outermost logged call is logged. The calls a logged method makes
internally, such as the holdout fits of `tuning_config` or `predict` running
`forward`, belong to the call that made them and are not logged on their own.

Usage events are only sent for accounts that opted in to usage analytics: see
`collector`, which receives them until `set_sink` installs another sink.

- `decorator`: the `log_usage` decorator, and where its usage events go.
- `events`: the usage events a call is logged as.
- `parameters`: which of an estimator's parameters are logged.
- `collector`: queues usage events and delivers them, for accounts that opted in.
"""

from tabpfn.analytics import collector
from tabpfn.analytics.decorator import log_usage, set_sink

set_sink(collector.collect)

__all__ = ["log_usage", "set_sink"]
