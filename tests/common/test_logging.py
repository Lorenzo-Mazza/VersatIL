"""Tests for versatil.common.logging module."""

import logging

import pytest

from versatil.common.logging import LOG_FORMAT, override_log_format


@pytest.mark.unit
class TestOverrideLogFormat:
    def test_replaces_formatter_on_existing_handlers(self):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logging.root.addHandler(handler)

        try:
            override_log_format()

            assert handler.formatter._fmt == LOG_FORMAT
        finally:
            logging.root.removeHandler(handler)

    def test_replaces_formatter_on_all_handlers(self):
        handler_a = logging.StreamHandler()
        handler_b = logging.StreamHandler()
        handler_a.setFormatter(logging.Formatter("%(message)s [a]"))
        handler_b.setFormatter(logging.Formatter("%(message)s [b]"))
        logging.root.addHandler(handler_a)
        logging.root.addHandler(handler_b)

        try:
            override_log_format()

            assert handler_a.formatter._fmt == LOG_FORMAT
            assert handler_b.formatter._fmt == LOG_FORMAT
        finally:
            logging.root.removeHandler(handler_a)
            logging.root.removeHandler(handler_b)

    def test_no_handlers_does_not_raise(self):
        original_handlers = logging.root.handlers[:]
        logging.root.handlers.clear()

        try:
            override_log_format()
        finally:
            logging.root.handlers = original_handlers
