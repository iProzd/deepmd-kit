# SPDX-License-Identifier: LGPL-3.0-or-later
import datetime
import logging
import math

log = logging.getLogger(__name__)


def _format_estimated_finish_time(
    eta_seconds: int,
    current_time: datetime.datetime | None = None,
) -> str:
    """Format the estimated local finish time.

    Parameters
    ----------
    eta_seconds : int
        Remaining time in seconds.
    current_time : datetime.datetime | None, optional
        Current local time used to estimate the finish timestamp. If ``None``,
        the current local time is used.

    Returns
    -------
    str
        Estimated local finish time in ``YYYY-MM-DD HH:MM`` format.
    """
    if current_time is None:
        current_time = datetime.datetime.now(datetime.timezone.utc).astimezone()
    elif current_time.tzinfo is not None:
        current_time = current_time.astimezone()
    finish_time = current_time + datetime.timedelta(seconds=eta_seconds)
    return finish_time.strftime("%Y-%m-%d %H:%M")


def format_training_message(
    batch: int,
    wall_time: float,
    eta: int | None = None,
    current_time: datetime.datetime | None = None,
) -> str:
    """Format the summary message for one training interval.

    Parameters
    ----------
    batch : int
        The batch index.
    wall_time : float
        Wall-clock time shown in the progress message in seconds.
    eta : int | None, optional
        Remaining time in seconds.
    current_time : datetime.datetime | None, optional
        Current local time used to estimate the finish timestamp. This is only
        used when ``eta`` is provided.

    Returns
    -------
    str
        The formatted training message.
    """
    msg = f"Batch {batch:7d}: total wall time = {wall_time:.2f} s"
    if isinstance(eta, int):
        eta_seconds = int(eta)
        msg += (
            f", eta = {datetime.timedelta(seconds=eta_seconds)!s} at "
            f"{_format_estimated_finish_time(eta_seconds, current_time=current_time)}"
        )
    return msg


def format_training_message_per_task(
    batch: int,
    task_name: str,
    rmse: dict[str, float],
    learning_rate: float | None,
    check_total_rmse_nan: bool = True,
) -> str:
    """Format training messages for a specific task.

    Parameters
    ----------
    batch : int
        The batch index
    task_name : str
        The task name
    rmse : dict[str, float]
        The root-mean-squared errors.
    learning_rate : float | None
        The learning rate
    check_total_rmse_nan : bool
        Whether to throw an error if the total RMSE is NaN

    Returns
    -------
    str
        The formatted training message for the task.
    """
    if task_name:
        task_name += ": "
    if learning_rate is None:
        lr = ""
    else:
        lr = f", lr = {learning_rate:8.2e}"
    # sort rmse
    rmse = dict(sorted(rmse.items()))
    msg = (
        f"Batch {batch:7d}: {task_name}"
        f"{', '.join([f'{kk} = {vv:8.2e}' for kk, vv in rmse.items()])}"
        f"{lr}"
    )
    if check_total_rmse_nan and math.isnan(rmse.get("rmse", 0.0)):
        log.error(msg)
        err_msg = (
            f"NaN detected at batch {batch:7d}: {task_name}. "
            "Something went wrong, and it is meaningless to continue."
        )
        raise RuntimeError(err_msg)
    return msg


def format_grad_norm_message(
    batch: int,
    grad_norm: float | dict[str, float],
) -> str:
    """Format gradient norm message for logging.

    This function formats gradient norm information for display during training.
    It supports both single-task training (single float value) and multitask
    training (dictionary mapping task names to gradient norms).

    Parameters
    ----------
    batch : int
        The batch index.
    grad_norm : float | dict[str, float]
        The gradient norm value(s). For single-task training, this is a float.
        For multitask training, this is a dictionary mapping task names to
        their respective gradient norms.

    Returns
    -------
    str
        The formatted gradient norm message.

    Examples
    --------
    Single-task training:
    >>> format_grad_norm_message(100, 1.5e-3)
    'Batch     100: grad_norm = 1.50e-03'

    Multitask training:
    >>> format_grad_norm_message(100, {"task1": 1.5e-3, "task2": 2.0e-3})
    'Batch     100: grad_norm: task1 = 1.50e-03, task2 = 2.00e-03'
    """
    if isinstance(grad_norm, dict):
        # Multitask: format as "grad_norm: task1 = X, task2 = Y, ..."
        grad_norm = dict(sorted(grad_norm.items()))
        norms_str = ", ".join(
            [f"{task} = {norm:8.2e}" for task, norm in grad_norm.items()]
        )
        msg = f"Batch {batch:7d}: grad_norm: {norms_str}"
    else:
        # Single-task: format as "grad_norm = X"
        msg = f"Batch {batch:7d}: grad_norm = {grad_norm:8.2e}"
    return msg
