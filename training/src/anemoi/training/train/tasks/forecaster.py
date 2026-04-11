# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch.utils.checkpoint import checkpoint

from anemoi.training.train.tasks.rollout import BaseRolloutGraphModule

if TYPE_CHECKING:
    from collections.abc import Generator

    import torch

LOGGER = logging.getLogger(__name__)


class GraphForecaster(BaseRolloutGraphModule):
    """Graph neural network forecaster for PyTorch Lightning."""

    task_type = "forecaster"

    def _rollout_step(
        self,
        batch: dict,
        rollout: int | None = None,
        validation_mode: bool = False,
    ) -> Generator[tuple[torch.Tensor | None, dict, list]]:
        """Rollout step for the forecaster.

        Parameters
        ----------
        batch : dict
            Dictionary batch to use for rollout (assumed to be already preprocessed)
        rollout : Optional[int], optional
            Number of times to rollout for, by default None
            If None, will use self.rollout
        validation_mode : bool, optional
            Whether in validation mode, and to calculate validation metrics, by default False
            If False, metrics will be empty

        Yields
        ------
        Generator[tuple[Union[torch.Tensor, None], dict, list], None, None]
            Loss value, metrics, and predictions (per step)

        """
        # start rollout of preprocessed batch
        rollout_steps = rollout or self.rollout
        required_time_steps = rollout_steps * self.n_step_output + self.n_step_input
        x = {}
        for dataset_name, dataset_batch in batch.items():
            x[dataset_name] = dataset_batch[
                :,
                0 : self.n_step_input,
                ...,
                self.data_indices[dataset_name].data.input.full,
            ]  # (bs, n_step_input, latlon, nvar)
            msg = (
                f"Batch length not sufficient for requested n_step_input length for {dataset_name}!"
                f", {dataset_batch.shape[1]} !>= {required_time_steps}"
            )
            assert dataset_batch.shape[1] >= required_time_steps, msg

        for rollout_step in range(rollout_steps):
            y_pred = self(x)
            y = {}
            for dataset_name, dataset_batch in batch.items():
                start = self.n_step_input + rollout_step * self.n_step_output
                y_time = dataset_batch.narrow(1, start, self.n_step_output)
                var_idx = self.data_indices[dataset_name].data.output.full.to(device=dataset_batch.device)
                y[dataset_name] = y_time.index_select(-1, var_idx)
            # y includes the auxiliary variables, so we must leave those out when computing the loss
            # Compute loss for each dataset and sum them up
            loss, metrics_next, y_pred = checkpoint(
                self.compute_loss_metrics,
                y_pred,
                y,
                step=rollout_step,
                validation_mode=validation_mode,
                use_reentrant=False,
            )

            # Advance input state for each dataset
            x = self._advance_input(x, y_pred, batch, rollout_step=rollout_step)

            yield loss, metrics_next, y_pred


class GraphDeltaForecaster(BaseRolloutGraphModule):
    """Graph neural network forecaster trained on state increments.

    The model predicts deltas in output space. During rollout, prognostic state
    variables are reconstructed by adding the predicted delta to the latest
    available state before feeding the next step.
    """

    task_type = "forecaster"

    def _compute_dataset_delta_target(
        self,
        dataset_batch,
        dataset_name: str,
        rollout_step: int,
    ):
        start = self.n_step_input + rollout_step * self.n_step_output
        y_time = dataset_batch.narrow(1, start, self.n_step_output)
        var_idx = self.data_indices[dataset_name].data.output.full.to(device=dataset_batch.device)
        y_state = y_time.index_select(-1, var_idx)

        prev_states = []
        for step in range(self.n_step_output):
            prev_time_index = start + step - 1
            prev_state = dataset_batch[:, prev_time_index, ...].index_select(-1, var_idx)
            prev_states.append(prev_state.unsqueeze(1))
        y_prev = torch.cat(prev_states, dim=1)

        return y_state - y_prev

    def _reconstruct_prognostic_states(
        self,
        x,
        y_pred,
        dataset_name: str,
    ):
        prognostic_in = self.data_indices[dataset_name].model.input.prognostic
        prognostic_out = self.data_indices[dataset_name].model.output.prognostic

        prev_state = x[:, -1, ..., prognostic_in]
        states = []
        for step in range(y_pred.shape[1]):
            prev_state = prev_state + y_pred[:, step, ..., prognostic_out]
            states.append(prev_state.unsqueeze(1))
        return torch.cat(states, dim=1)

    def _advance_dataset_input(
        self,
        x,
        y_pred,
        batch,
        dataset_name: str,
        rollout_step: int = 0,
    ):
        keep_steps = min(self.n_step_output, self.n_step_input)
        x = x.roll(-keep_steps, dims=1)

        reconstructed_states = self._reconstruct_prognostic_states(x, y_pred, dataset_name)

        for i in range(keep_steps):
            state_step = reconstructed_states[:, -(i + 1)]
            x[:, -(i + 1), ..., self.data_indices[dataset_name].model.input.prognostic] = state_step

            batch_time_index = self.n_step_input + (rollout_step + 1) * self.n_step_output - (i + 1)

            x[:, -(i + 1)] = self.output_mask[dataset_name].rollout_boundary(
                x[:, -(i + 1)],
                batch[:, batch_time_index],
                self.data_indices[dataset_name],
                grid_shard_slice=self.grid_shard_slice[dataset_name],
            )

            x[:, -(i + 1), ..., self.data_indices[dataset_name].model.input.forcing] = batch[
                :,
                batch_time_index,
                ...,
                self.data_indices[dataset_name].data.input.forcing,
            ]

        return x

    def _rollout_step(
        self,
        batch: dict,
        rollout: int | None = None,
        validation_mode: bool = False,
    ):
        rollout_steps = rollout or self.rollout
        required_time_steps = rollout_steps * self.n_step_output + self.n_step_input
        x = {}
        for dataset_name, dataset_batch in batch.items():
            x[dataset_name] = dataset_batch[
                :,
                0 : self.n_step_input,
                ...,
                self.data_indices[dataset_name].data.input.full,
            ]
            msg = (
                f"Batch length not sufficient for requested n_step_input length for {dataset_name}!"
                f", {dataset_batch.shape[1]} !>= {required_time_steps}"
            )
            assert dataset_batch.shape[1] >= required_time_steps, msg

        for rollout_step in range(rollout_steps):
            y_pred = self(x)
            y = {
                dataset_name: self._compute_dataset_delta_target(dataset_batch, dataset_name, rollout_step)
                for dataset_name, dataset_batch in batch.items()
            }

            loss, metrics_next, y_pred = checkpoint(
                self.compute_loss_metrics,
                y_pred,
                y,
                step=rollout_step,
                validation_mode=validation_mode,
                use_reentrant=False,
            )

            x = self._advance_input(x, y_pred, batch, rollout_step=rollout_step)

            yield loss, metrics_next, y_pred
