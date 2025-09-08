# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np


class LearningRateExp:
    def __init__(
        self,
        start_lr,
        stop_lr,
        decay_steps,
        stop_steps,
        decay_rate=None,
        **kwargs,
    ) -> None:
        """
        Construct an exponential-decayed learning rate.

        Parameters
        ----------
        start_lr
            The learning rate at the start of the training.
        stop_lr
            The desired learning rate at the end of the training.
            When decay_rate is explicitly set, this value will serve as
            the minimum learning rate during training. In other words,
            if the learning rate decays below stop_lr, stop_lr will be applied instead.
        decay_steps
            The learning rate is decaying every this number of training steps.
        stop_steps
            The total training steps for learning rate scheduler.
        decay_rate
            The decay rate for the learning rate.
            If provided, the decay rate will be set instead of
            calculating it through interpolation between start_lr and stop_lr.
        """
        self.start_lr = start_lr
        default_ds = 100 if stop_steps // 10 > 100 else stop_steps // 100 + 1
        self.decay_steps = decay_steps
        if self.decay_steps >= stop_steps:
            self.decay_steps = default_ds
        self.decay_rate = np.exp(
            np.log(stop_lr / self.start_lr) / (stop_steps / self.decay_steps)
        )
        if decay_rate is not None:
            self.decay_rate = decay_rate
        self.min_lr = stop_lr

    def value(self, step) -> np.float64:
        """Get the learning rate at the given step."""
        step_lr = self.start_lr * np.power(self.decay_rate, step // self.decay_steps)
        if step_lr < self.min_lr:
            step_lr = self.min_lr
        return step_lr


class LearningRateCosine:
    def __init__(
        self,
        start_lr,
        stop_lr,
        stop_steps,
        **kwargs,
    ) -> None:
        self.start_lr = start_lr
        self.lr_min_factor = stop_lr / start_lr
        self.stop_steps = stop_steps

    def value(self, step) -> np.float64:
        if step >= self.stop_steps:
            return self.start_lr * self.lr_min_factor
        return self.start_lr * (
            self.lr_min_factor
            + 0.5
            * (1 - self.lr_min_factor)
            * (1 + np.cos(np.pi * (step / self.stop_steps)))
        )


class LearningRateWSD:
    def __init__(
        self,
        start_lr,
        stop_lr,
        stop_steps,
        decay_mode="85:10:5",  # stable-decay-stable
        **kwargs,
    ) -> None:
        self.start_lr = start_lr
        self.stop_lr = stop_lr
        self.stop_steps = stop_steps
        self.decay_mode = [float(ii) for ii in decay_mode.split(":")]
        assert len(self.decay_mode) == 3
        self.decay_start_rate = self.decay_mode[0] / sum(self.decay_mode)
        self.decay_end_rate = (self.decay_mode[0] + self.decay_mode[1]) / sum(
            self.decay_mode
        )

    def value(self, step) -> np.float64:
        if step < self.decay_start_rate * self.stop_steps:
            return self.start_lr
        elif step >= self.decay_end_rate * self.stop_steps:
            return self.stop_lr
        else:
            # linear decay
            decay_rate = (self.start_lr - self.stop_lr) / (
                self.decay_end_rate * self.stop_steps
                - self.decay_start_rate * self.stop_steps
            )
            return self.start_lr - decay_rate * (
                step - self.decay_start_rate * self.stop_steps
            )


class LearningRateLinear:
    def __init__(
        self,
        start_lr: float,
        stop_steps: int,
        decay_steps: int,
        start_factor: float = 1.0,
        end_factor: float = 1.0,
        **kwargs,
    ) -> None:
        """
        Piecewise-constant linear LR schedule updated every `decay_steps`.

        The LR factor linearly interpolates from `start_factor` (at step=0)
        to `end_factor` (at and after step >= stop_steps), but the value only
        changes at discrete update boundaries (multiples of `decay_steps`).

        Parameters
        ----------
        start_lr : float
            Base learning rate (multiplied by the factor below).
        stop_steps : int
            Total number of training steps for this scheduler.
        decay_steps : int
            Interval (in steps) between LR updates; e.g., 1k or 10k.
        start_factor : float
            Multiplicative factor at step 0.
        end_factor : float
            Multiplicative factor at and after step >= stop_steps.

        Examples
        --------
        Let k = floor(step / decay_steps).
        Let U = stop_steps / decay_steps  (can be non-integer).
        progress = clamp(k / U, 0, 1).
        factor(step) = start_factor + (end_factor - start_factor) * progress.
        After step >= stop_steps, factor(step) = end_factor.
        - If `decay_steps` >= `stop_steps`, it will be replaced by a reasonable
            default so the schedule still updates multiple times.
        - This mirrors the spirit of torch.optim.lr_scheduler.LinearLR but with
            discrete updates every `decay_steps` steps (akin to treating each
            update as an "epoch").
        """
        self.base_lr = float(start_lr)
        self.start_factor = float(start_factor)
        self.end_factor = float(end_factor)
        self.stop_steps = int(stop_steps)

        # Choose a safe decay_steps (avoid zero/oversized intervals)
        self.decay_steps = int(decay_steps) if int(decay_steps) > 0 else 1
        default_ds = 100 if self.stop_steps // 10 > 100 else self.stop_steps // 100 + 1
        if self.decay_steps >= self.stop_steps:
            self.decay_steps = max(1, int(default_ds))

        # Total number of "update buckets" over the training horizon (float)
        self.total_updates = self.stop_steps / self.decay_steps

    def value(self, step: int) -> np.float64:
        """
        Get the learning rate at the given `step`.

        - Updates occur only at multiples of `decay_steps`.
        - Saturates at `end_factor` when step >= stop_steps.
        - Negative steps are treated as 0.
        """
        if step <= 0:
            factor = self.start_factor
        elif step >= self.stop_steps:
            factor = self.end_factor
        else:
            updates_done = step // self.decay_steps  # integer count of updates so far
            progress = (
                updates_done / self.total_updates
            )  # may be slightly < 1 before stop_steps
            # Clamp numerical drift into [0, 1]
            if progress < 0.0:
                progress = 0.0
            elif progress > 1.0:
                progress = 1.0

            factor = (
                self.start_factor + (self.end_factor - self.start_factor) * progress
            )
            # Monotone clamp to never overshoot end_factor due to rounding
            if self.end_factor < self.start_factor:
                factor = max(factor, self.end_factor)
            else:
                factor = min(factor, self.end_factor)

        return np.float64(self.base_lr * factor)
