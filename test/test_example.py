#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# run with 'pytest -ra' from projcect root dir
# import pytest


def test_stuff():
    """
    Check that stuff produces correct result
    """
    a = 2 + 2
    b = 4
    assert a == b
