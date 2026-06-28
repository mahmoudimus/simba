"""The daemon raises its soft file-descriptor limit at startup.

A full LanceDB scan (e.g. GET /list) opens many fragment/version files at once;
on macOS the default soft RLIMIT_NOFILE (256) is easily exhausted on a large
table -> `Too many open files (os error 24)`. The daemon bumps the soft limit
toward a configured target (capped at the hard limit) before serving.
"""

from __future__ import annotations

import resource
import unittest.mock

import simba.memory.server as server


class TestRaiseFdLimit:
    def test_zero_target_is_noop(self) -> None:
        with unittest.mock.patch.object(resource, "setrlimit") as setr:
            assert server._raise_fd_limit(0) is None
        setr.assert_not_called()

    def test_raises_toward_target_under_infinite_hard(self) -> None:
        with (
            unittest.mock.patch.object(
                resource, "getrlimit", return_value=(256, resource.RLIM_INFINITY)
            ),
            unittest.mock.patch.object(resource, "setrlimit") as setr,
        ):
            assert server._raise_fd_limit(65536) == 65536
        setr.assert_called_once_with(
            resource.RLIMIT_NOFILE, (65536, resource.RLIM_INFINITY)
        )

    def test_caps_at_finite_hard_limit(self) -> None:
        with (
            unittest.mock.patch.object(
                resource, "getrlimit", return_value=(256, 10240)
            ),
            unittest.mock.patch.object(resource, "setrlimit") as setr,
        ):
            assert server._raise_fd_limit(65536) == 10240
        setr.assert_called_once_with(resource.RLIMIT_NOFILE, (10240, 10240))

    def test_noop_when_soft_already_high_enough(self) -> None:
        with (
            unittest.mock.patch.object(
                resource, "getrlimit", return_value=(100000, resource.RLIM_INFINITY)
            ),
            unittest.mock.patch.object(resource, "setrlimit") as setr,
        ):
            assert server._raise_fd_limit(65536) == 100000
        setr.assert_not_called()

    def test_fail_soft_on_oserror(self) -> None:
        with (
            unittest.mock.patch.object(
                resource, "getrlimit", return_value=(256, resource.RLIM_INFINITY)
            ),
            unittest.mock.patch.object(
                resource, "setrlimit", side_effect=OSError("denied")
            ),
        ):
            assert server._raise_fd_limit(65536) is None
