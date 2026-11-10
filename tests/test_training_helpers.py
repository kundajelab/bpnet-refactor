"""
    Characterization tests for the early stopping and learning rate
    rules of the trainer (bpnet.train.training)
"""

import pytest

pytest.importorskip("tensorflow")

from bpnet.train.training import early_stopping_check, reduce_lr_on_plateau


# --------------------------------------------------------------------
# early stopping: stop when none of the last `patience` losses is more
# than `min_delta` below the loss just before them
# --------------------------------------------------------------------

def test_early_stopping_needs_more_than_patience_epochs():
    assert early_stopping_check([5.0] * 5, patience=5) is False
    assert early_stopping_check([5.0] * 6, patience=5) is True


def test_early_stopping_does_not_trigger_while_improving():
    losses = [10.0 - i for i in range(20)]
    for n in range(1, 21):
        assert early_stopping_check(losses[:n], patience=5) is False


def test_early_stopping_on_a_plateau():
    losses = [5.0, 4.0, 3.0, 3.0, 3.0, 3.0, 3.0]
    assert early_stopping_check(losses, patience=5) is False
    assert early_stopping_check(losses + [3.0], patience=5) is True


def test_early_stopping_min_delta():
    # improvements of 0.0005 per epoch do not count with min_delta 0.001
    # until they add up to more than min_delta against the anchor
    losses = [3.0, 2.9995, 2.9990, 2.9985]
    assert early_stopping_check(losses, patience=2, min_delta=1e-3) is True
    assert early_stopping_check(losses, patience=3, min_delta=1e-3) is False
    assert early_stopping_check(losses, patience=2, min_delta=1e-4) is False


def test_early_stopping_anchor_is_the_loss_before_the_window():
    # the comparison is with the loss `patience` epochs ago, not with
    # the best loss so far: after a spike, any recovery counts
    losses = [3.0, 9.0, 8.0, 8.0]
    assert early_stopping_check(losses, patience=2) is False
    losses = [3.0, 9.0, 9.5, 9.5]
    assert early_stopping_check(losses, patience=2) is True


# --------------------------------------------------------------------
# learning rate: multiply by `factor` when none of the last `patience`
# losses is below the loss just before them; never go below `min_lr`
# --------------------------------------------------------------------

def test_lr_unchanged_until_enough_epochs():
    assert reduce_lr_on_plateau([5.0, 5.0], 0.001, patience=2) == 0.001


def test_lr_unchanged_when_any_recent_loss_improved():
    assert reduce_lr_on_plateau([5.0, 6.0, 4.9999], 0.001, patience=2) == 0.001
    assert reduce_lr_on_plateau([5.0, 4.0, 6.0], 0.001, patience=2) == 0.001


def test_lr_anchor_is_the_loss_before_the_window():
    # as for early stopping, the comparison is with the loss `patience`
    # epochs ago, not with the best loss so far: recovering from a
    # spike counts as improvement
    assert reduce_lr_on_plateau([3.0, 9.0, 8.0, 8.0], 0.001,
                                patience=2) == 0.001
    assert reduce_lr_on_plateau([3.0, 9.0, 9.0, 9.5], 0.001, factor=0.5,
                                patience=2) == 0.0005


def test_lr_reduced_on_a_plateau():
    assert reduce_lr_on_plateau([5.0, 5.0, 5.0], 0.001, factor=0.5,
                                patience=2) == 0.0005
    assert reduce_lr_on_plateau([5.0, 6.0, 5.5], 0.001, factor=0.1,
                                patience=2) == pytest.approx(0.0001)


def test_lr_is_kept_when_the_reduction_would_cross_min_lr():
    # 0.00015 * 0.5 < 1e-4: the rate stays at 0.00015, it is not set to
    # min_lr
    assert reduce_lr_on_plateau([5.0, 5.0, 5.0], 0.00015, factor=0.5,
                                patience=2, min_lr=1e-4) == 0.00015
    assert reduce_lr_on_plateau([5.0, 5.0, 5.0], 0.0002, factor=0.5,
                                patience=2, min_lr=1e-4) == 0.0001


def test_released_schedule_reaches_three_rates():
    """
        With the released settings (1e-3, factor 0.5, floor 1e-4) the
        rate can only take the values 1e-3, 5e-4, 2.5e-4 and 1.25e-4
    """

    lr, seen = 0.001, [0.001]
    for _ in range(10):
        lr = reduce_lr_on_plateau([5.0] * 6, lr, factor=0.5, patience=5,
                                  min_lr=1e-4)
        seen.append(lr)
    assert sorted(set(seen), reverse=True) == \
        pytest.approx([0.001, 0.0005, 0.00025, 0.000125])
