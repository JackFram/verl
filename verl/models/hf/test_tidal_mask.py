import torch
import pytest
from utlls import _get_tidal_mask


inf_neg = torch.finfo(torch.float32).min   # sentinel used by the model

@pytest.mark.parametrize(
    "attn_weights, attn_mask, window_budget, sink_budget, tidal_budget,"
    " expected_average, expected_union",
    [
        # single batch, two heads
        (
            torch.tensor(
                [[
                    [[0.10, 0.80, 0.20, 0.30],
                     [0.30, 0.40, 0.70, 0.10]],
                    [[0.05, 0.60, 0.40, 0.20],
                     [0.20, 0.50, 0.60, 0.05]],
                ]],
                dtype=torch.float32,
            ),
            torch.zeros((1, 1, 2, 4), dtype=torch.float32),
            1, 1, 1,
            torch.tensor(
                [[[[True,  True,  True,  False],
                   [True,  False, True,  True ]]]],
                dtype=torch.bool,
            ),
            torch.tensor(
                [[[[True,  True,  True,  False],
                   [True,  False, True,  True ]]]],
                dtype=torch.bool,
            ),
        ),

        # single batch, diff focus
        (
            torch.tensor(
                [[
                    [[0.00, 0.00, 0.10, 0.90]],   # head 0
                    [[0.00, 0.00, 0.80, 0.10]],   # head 1
                ]],
                dtype=torch.float32,
            ),
            torch.zeros((1, 1, 1, 4), dtype=torch.float32),
            0, 0, 1,
            torch.tensor([[[[False, False, False, True]]]],  dtype=torch.bool),
            torch.tensor([[[[False, False,  True, False]]]], dtype=torch.bool),
        ),

        # two batches
        (
            torch.tensor(
                [
                    [   # batch 0
                        [[0.10, 0.20, 0.90, 0.10]],
                        [[0.10, 0.80, 0.20, 0.10]],
                    ],
                    [   # batch 1
                        [[0.10, 0.10, 0.30, 0.90]],
                        [[0.80, 0.10, 0.05, 0.05]],
                    ],
                ],
                dtype=torch.float32,
            ),
            torch.zeros((2, 1, 1, 4), dtype=torch.float32),
            0, 0, 1,
            torch.tensor(
                [
                    [[[False, False, True,  False]]],
                    [[[False, False, False, True ]]],
                ],
                dtype=torch.bool,
            ),
            torch.tensor(
                [
                    [[[False, True,  False, False]]],
                    [[[True,  False, False, False]]],
                ],
                dtype=torch.bool,
            ),
        ),

        # left‑padding present
        (
            # 1 × 2 × 1 × 6   (kv_len = 6, qo_len = 1)
            torch.tensor(
                [[
                    [[0.00, 0.00, 0.05, 0.05, 0.50, 0.60]],   # head 0
                    [[0.00, 0.00, 1.00, 0.10, 0.05, 0.05]],   # head 1
                ]],
                dtype=torch.float32,
            ),
            torch.tensor(
                [[[[inf_neg, inf_neg, 0.0, 0.0, 0.0, 0.0]]]],
                dtype=torch.float32,
            ),
            0, 0, 1, 
            torch.tensor([[[[False, False, True, False, False, False]]]], dtype=torch.bool),
            torch.tensor([[[[False, False, False, False, False, True]]]], dtype=torch.bool),
        ),
    ],
)
def test_get_tidal_mask(
    attn_weights,
    attn_mask,
    window_budget,
    sink_budget,
    tidal_budget,
    expected_average,
    expected_union,
):
    mask_average = _get_tidal_mask(
        attn_mask.clone(), attn_weights.clone(),
        window_budget, sink_budget, tidal_budget,
        merge_type="average",
    )
    mask_union = _get_tidal_mask(
        attn_mask.clone(), attn_weights.clone(),
        window_budget, sink_budget, tidal_budget,
        merge_type="union",
    )

    assert torch.equal(mask_average, expected_average)
    assert torch.equal(mask_union,   expected_union)


if __name__ == "__main__":
    pytest.main([__file__])
