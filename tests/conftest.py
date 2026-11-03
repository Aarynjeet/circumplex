"""Shared fixtures: small synthetic data so unit tests never need the network."""

from __future__ import annotations

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _seed() -> None:
    np.random.seed(0)
    torch.manual_seed(0)
