import numpy as np

from src.promptgen.common.instances import extract_lesion_instances_xyz
from src.promptgen.common.metrics import match_prompts_to_gt


def test_hnm_mining_logic_toy():
    """Toy GT + proposals -> hard negatives filtered correctly.

    Inputs: none.
    Outputs: none (asserts hard negatives indices).
    Operation: uses match rules to separate FP proposals.
    """
    label = np.zeros((6, 6, 6), dtype=np.uint8)
    label[1, 1, 1] = 1
    lesions = extract_lesion_instances_xyz(label)

    proposals = [
        ((0, 0, 0, 2, 2, 2), 0.9),  # covers center -> matched
        ((3, 3, 3, 5, 5, 5), 0.8),  # FP
        ((1, 1, 1, 2, 2, 2), 0.7),  # matched
    ]

    match_info = match_prompts_to_gt(proposals, lesions)
    matched = match_info["prompt_to_gt"]

    hard_neg_indices = [i for i in range(len(proposals)) if i not in matched]
    assert hard_neg_indices == [1, 2]
