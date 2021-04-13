import pytest
from pgen import models, esm_msa_sampler

####### Fixtures #######

@pytest.fixture(scope="session")
def esm_msa():
    return models.ESM_MSA1()

@pytest.fixture(scope="session")
def msa_sampler(esm_msa):
    sampler = esm_msa_sampler.ESM_MSA_sampler(esm_msa, device="cpu")
    return sampler

###### Tests #######

def test_untokenize_batch(msa_sampler):
    input_batch = [[
        [0, 5, 5, 5, 32, 32],
        [0, 5, 25, 25, 32, 32]
    ], [
        [0, 5, 25, 23, 13, 32]
    ]]
    result = msa_sampler.untokenize_batch(input_batch)
    expected_result = ["AAA<mask><mask>", "ABB<mask><mask>", "ABCD<mask>"]
    assert result == expected_result

def test_get_init_msa(msa_sampler):
    seed = ["AAA", "ABB", "ABCD"]
    batch_size = 2
    max_len = 5
    result = msa_sampler.get_init_msa(seed, 5, 2)
    assert result.shape[0] == batch_size
    assert result.shape[1] == len(seed)
    assert result.shape[2] == max_len + 1

    assert result[0][0].tolist() == [0, 5, 5, 5, 32, 32]
    assert result[0][1].tolist() == [0, 5, 25, 25, 32, 32]
    assert result[0][2].tolist() == [0, 5, 25, 23, 13, 32]


def test_generate_batch_equals_seqs(msa_sampler):
    out = msa_sampler.generate(4, ["AAA", "AAB"], batch_size=4, max_len=3)

    assert (len(out) == 4)
    for s in out:
        assert (len(s) == 3)


def test_generate_batch_single_iteration(msa_sampler):
    out = msa_sampler.generate(4, ["AAA", "AAB"], num_iters=1,
                               max_len=5, num_positions=1, in_order=True)

    assert (len(out) == 4)

    assert out[0][1:3] == "AA"
    assert out[1][1:3] == "AB"
    assert out[2][1:3] == "AA"
    assert out[3][1:3] == "AB"


def test_generate_batch_randomly(msa_sampler):
    out = msa_sampler.generate(4, ["AAA", "AAB"], num_iters=1,
                               max_len=5, num_positions=1, in_order=False)

    assert (len(out) == 4)


def test_get_target_indexes_in_order(msa_sampler):
    last_i, target_indexes = msa_sampler.get_target_index_in_order(
        batch_size=2, indexes=[0, 1, 2, 3], next_i=1, num_positions=2,
        num_sequences=3)

    assert last_i == 3
    assert target_indexes == [
        [[2, 3], [2, 3], [2, 3]],
        [[2, 3], [2, 3], [2, 3]]
    ]

def test_get_target_indexes_randomly(msa_sampler):
    indexes = [0,1,2,3]
    target_indexes = msa_sampler.get_random_target_index(
        batch_size=2, indexes=indexes, num_positions=2,
        num_sequences=3)

    assert len(target_indexes) == 2
    assert len(target_indexes[0]) == 3
    assert len(target_indexes[0][0]) == 2

    for item in target_indexes[0][0]:
        assert item in indexes

def test_get_target_indexes_all_positions(msa_sampler):
    target_indexes = msa_sampler.get_target_indexes_all_positions(
        batch_size=2, indexes=[0, 1, 2, 3], num_sequences=3)

    assert target_indexes == [
        [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]],
        [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]
    ]

# TODO test masking