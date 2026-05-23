# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import math

import numpy as np
import pytest
import torch
import torch.utils.data
from pytest_mock import MockFixture

from anemoi.training.data.multidataset import MultiDataset


# ---------------------------------------------------------------------------
# Module-level helper for the DataLoader regression test.
# Must be at module level so it is picklable by DataLoader's worker processes.
# ---------------------------------------------------------------------------


class _StampedIterableDataset(torch.utils.data.IterableDataset):
    """Minimal picklable dataset that mimics MultiDataset's output contract.

    Each sample yields a dict with a ``data`` payload whose value is the sample
    index (so we can verify which sample we got) and a ``__sample_time_ns__``
    key that is the nanosecond timestamp of the corresponding date.  The two
    values are correlated: ``data[0] == i`` ↔ ``timestamp == dates[i]``.
    This lets the test assert that timestamps remain attached to their samples
    rather than being reconstructed from batch position.
    """

    def __init__(self, dates: np.ndarray) -> None:
        super().__init__()
        # Store as plain list so we stay picklable without numpy sharing concerns.
        self._dates_ns: list[int] = [int(d.astype(np.int64)) for d in dates]
        self._n = len(dates)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            start, end = 0, self._n
        else:
            per_worker = math.ceil(self._n / worker_info.num_workers)
            start = worker_info.id * per_worker
            end = min(start + per_worker, self._n)

        for idx in range(start, end):
            yield {
                # data encodes the sample index so the test can identify each sample.
                "data": torch.tensor([float(idx)]),
                # Real timestamp embedded in the sample – NOT derived from iteration order.
                "__sample_time_ns__": torch.tensor([self._dates_ns[idx]], dtype=torch.int64),
            }



class TestMultiDataset:
    """Test MultiDataset instantiation and properties."""

    @pytest.fixture
    def dataset_config(self) -> dict:
        """Fixture to provide dataset configuration."""
        return {
            "timestep": "6h",
            "relative_date_indices": [0, 1, 3],  # e.g. f([t, t-6h]) = t+12h
            "shuffle": True,
        }

    @pytest.fixture
    def multi_dataset(self, mocker: MockFixture, dataset_config: dict) -> MultiDataset:
        """Fixture to provide a MultiDataset instance with mocked datasets."""
        data_readers = {"dataset_a": None, "dataset_b": None}
        grid_indices = {"dataset_a": None, "dataset_b": None}

        # Mock create_dataset to return mock datasets
        mock_dataset_a = mocker.MagicMock()
        mock_dataset_a.missing = set()
        mock_dataset_a.dates = list(range(30))  # 15 reference dates
        mock_dataset_a.has_trajectories = False
        mock_dataset_a.frequency = "3h"

        mock_dataset_b = mocker.MagicMock()
        mock_dataset_b.missing = {7, 8, 9, 10}
        mock_dataset_b.dates = list(range(30))  # 15 reference dates
        mock_dataset_b.has_trajectories = False
        mock_dataset_b.frequency = "3h"

        mocker.patch(
            "anemoi.training.data.multidataset.create_dataset",
            side_effect=[mock_dataset_a, mock_dataset_b],
        )

        return MultiDataset(data_readers=data_readers, grid_indices=grid_indices, **dataset_config)

    def test_timeincrement(self, multi_dataset: MultiDataset) -> None:
        """Test that timeincrement is correctly computed from timestep."""
        expected_timeincrement = 2  # 6H (timestep) in 3h steps (frequency)
        assert multi_dataset.timeincrement == expected_timeincrement

    def test_valid_date_indices(self, multi_dataset: MultiDataset) -> None:
        """Test that valid_date_indices returns the intersection of indices from all datasets."""
        # relative_date_indices: [0, 1, 3] (for 6H timestep)
        # data (3h) -> data_relative_time_indices: [0, 2, 6]
        # dataset_a|b has dates [0, 1, 2, ..., 29]
        # dataset_a has indices [0, 1, 2, 3, 4, ..., 22, 23], where 23 = 29 - max(data_relative_time_indices)
        # dataset_b has missing indices {7, 8, 9, 10}
        # dataset_b has missing indices {7, 8, 9, 10}
        # dataset_b has indices [0, 11, 12, 13, ..., 22, 23]
        # intersection should be [0, 11, 12, 13, ..., 22, 23]

        # Test valid_date_indices property
        valid_indices = multi_dataset.valid_date_indices

        # Should return intersection [0, 11, 12, 13, ..., 22, 23]
        expected_indices = np.array([0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])
        assert np.array_equal(valid_indices, expected_indices)

    def test_valid_date_indices_empty_dataset(self, multi_dataset: MultiDataset, mocker: MockFixture) -> None:
        """Test that MultiDataset raises ValueError when a dataset has no valid indices."""
        # Clear the cached property if it was already computed
        if "valid_date_indices" in multi_dataset.__dict__:
            del multi_dataset.__dict__["valid_date_indices"]

        # Mock get_usable_indices: dataset_a has valid indices, dataset_b has none
        mocker.patch(
            "anemoi.training.data.multidataset.get_usable_indices",
            side_effect=[
                np.array([0, 1, 2, 3, 4, 5]),  # dataset_a
                np.array([]),  # dataset_b - empty!
            ],
        )

        # Accessing valid_date_indices should raise ValueError
        with pytest.raises(ValueError, match="No valid date indices found for dataset 'dataset_b'"):
            _ = multi_dataset.valid_date_indices

    def test_valid_date_indices_empty_intersection(self, multi_dataset: MultiDataset, mocker: MockFixture) -> None:
        """Test that MultiDataset raises ValueError when intersection of valid indices is empty."""
        # Clear the cached property if it was already computed
        if "valid_date_indices" in multi_dataset.__dict__:
            del multi_dataset.__dict__["valid_date_indices"]

        # Mock get_usable_indices: both datasets have valid indices but no overlap
        # dataset_a has indices: [0, 1, 2]
        # dataset_b has indices: [5, 6, 7]
        # intersection should be empty ([])
        mocker.patch(
            "anemoi.training.data.multidataset.get_usable_indices",
            side_effect=[
                np.array([0, 1, 2]),  # dataset_a
                np.array([5, 6, 7]),  # dataset_b
            ],
        )

        # Accessing valid_date_indices should raise ValueError
        with pytest.raises(ValueError, match="No valid date indices found after intersection across all datasets"):
            _ = multi_dataset.valid_date_indices

    def test_get_sample_includes_real_timestamps(self, mocker: MockFixture) -> None:
        """get_sample must embed real timestamps so export callbacks never reconstruct from batch_idx.

        With num_workers > 1, DataLoader worker interleaving makes batch_idx-based time
        reconstruction wrong. The fix is to propagate '__sample_time_ns__' through the batch.
        This test verifies the key is present, is a torch.Tensor of int64, and matches the
        primary dataset's dates — regardless of how many workers are used.
        """
        import torch

        base_dates = np.array(
            ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
            dtype="datetime64[D]",
        ).astype("datetime64[ns]")

        mock_dataset_a = mocker.MagicMock()
        mock_dataset_a.missing = set()
        mock_dataset_a.dates = base_dates
        mock_dataset_a.has_trajectories = False
        mock_dataset_a.frequency = "1D"
        mock_dataset_a.get_sample.return_value = torch.zeros(5, 1, 4)

        mock_grid_indices = mocker.MagicMock()
        mock_grid_indices.get_shard_indices.return_value = slice(None)

        mocker.patch(
            "anemoi.training.data.multidataset.create_dataset",
            return_value=mock_dataset_a,
        )

        ds = MultiDataset(
            data_readers={"data": None},
            grid_indices={"data": mock_grid_indices},
            relative_date_indices=[0, 1, 2],
            timestep="1D",
            shuffle=False,
        )

        sample = ds.get_sample(0)

        assert "__sample_time_ns__" in sample, (
            "get_sample must include '__sample_time_ns__' so export callbacks "
            "never need to reconstruct timestamps from batch_idx."
        )
        ts = sample["__sample_time_ns__"]
        assert isinstance(ts, torch.Tensor), "__sample_time_ns__ must be a torch.Tensor"
        assert ts.dtype == torch.int64, "__sample_time_ns__ must be int64 (nanoseconds since epoch)"

        expected_ns = base_dates[:3].astype(np.int64)
        np.testing.assert_array_equal(
            ts.numpy(),
            expected_ns,
            err_msg="Timestamps must match primary dataset dates, not be synthesised from batch_idx.",
        )


# ---------------------------------------------------------------------------
# End-to-end DataLoader regression test for num_workers > 1.
# ---------------------------------------------------------------------------


def test_dataloader_preserves_sample_timestamps_with_multiple_workers() -> None:
    """Verify ``__sample_time_ns__`` survives DataLoader collation with num_workers=2.

    Regression test for the bug where ``ExportPredictions`` reconstructed timestamps
    from ``batch_idx * batch_size + sample_idx``.  With ``num_workers > 1`` the
    DataLoader interleaves batches from different workers, so that formula produces
    wrong dates.  The fix embeds real timestamps in each sample via
    ``MultiDataset.get_sample``; this test verifies that:

    1. The ``__sample_time_ns__`` key is present in every collated batch.
    2. The timestamps correspond to the *actual* sample dates, not to the batch
       position (which would differ under worker interleaving).
    3. All samples are accounted for, with no duplicates.
    """
    n_samples = 20
    batch_size = 2
    n_workers = 2

    # Build a date range so we have a concrete expected value for every index.
    dates = np.array(
        [f"2024-01-{d+1:02d}" for d in range(n_samples)],
        dtype="datetime64[D]",
    ).astype("datetime64[ns]")

    ds = _StampedIterableDataset(dates)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=n_workers)

    seen_indices: list[int] = []
    for batch in dl:
        # Key must survive collation regardless of which worker produced the batch.
        assert "__sample_time_ns__" in batch, (
            "DataLoader collation must preserve '__sample_time_ns__' from every sample dict. "
            "If this key is missing, export/plot callbacks cannot derive correct timestamps "
            "with num_workers > 1."
        )

        ts = batch["__sample_time_ns__"]  # shape: (batch_size, 1)
        data = batch["data"]  # shape: (batch_size, 1), value == sample index

        assert ts.dtype == torch.int64, "__sample_time_ns__ must remain int64 after collation"

        for i in range(data.shape[0]):
            sample_idx = int(data[i, 0].item())
            expected_ns = int(dates[sample_idx].astype(np.int64))
            actual_ns = ts[i, 0].item()

            assert actual_ns == expected_ns, (
                f"Sample {sample_idx}: timestamp {actual_ns} != expected {expected_ns}. "
                "Timestamps must be attached to their samples, not reconstructed from "
                "the iteration order (which is wrong under DataLoader worker interleaving)."
            )
            seen_indices.append(sample_idx)

    # All samples must be delivered exactly once.
    assert sorted(seen_indices) == list(range(n_samples)), (
        "DataLoader must yield every sample exactly once. "
        f"Expected {list(range(n_samples))}, got {sorted(seen_indices)}."
    )

