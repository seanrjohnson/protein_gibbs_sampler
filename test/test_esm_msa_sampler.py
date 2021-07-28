import pytest
from pgen import models, esm_msa_sampler

####### Fixtures #######
from pgen.esm_msa_sampler import ESM_MSA_ALLOWED_AMINO_ACIDS


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
    seed = ["AAA", "ACC", "ACDE"]
    batch_size = 2
    max_len = 5
    result = msa_sampler.get_init_msa(seed, 5, 2)
    assert result.shape[0] == batch_size
    assert result.shape[1] == len(seed)
    assert result.shape[2] == max_len + 1

    assert result[0][0].tolist() == [0, 5, 5, 5, 32, 32]
    assert result[0][1].tolist() == [0, 5, 23, 23, 32, 32]
    assert result[0][2].tolist() == [0, 5, 23, 13, 9, 32]


def test_get_init_msa_lowercase(msa_sampler):
    seed = ["aaa", "aCC", "aCDE"]
    result = msa_sampler.get_init_msa(seed, 5, 2)

    assert result[0][0].tolist() == [0, 5, 5, 5, 32, 32]
    assert result[0][1].tolist() == [0, 5, 23, 23, 32, 32]
    assert result[0][2].tolist() == [0, 5, 23, 13, 9, 32]


def test_get_init_msa_fails_if_non_standard_supplied(msa_sampler):
    seed = ["X"]
    try:
        msa_sampler.get_init_msa(seed, 2)
        assert False
    except Exception as e:
        assert str(e) == "Invalid input character: X"


def test_generate_batch_equals_seqs(msa_sampler):
    out = msa_sampler.generate(4, ["AAA", "AAC"], batch_size=4, max_len=3)

    assert (len(out) == 4)
    for s in out:
        assert (len(s) == 3)


@pytest.mark.parametrize("batch_size, num_positions,mask,leader_length,in_order",
                         [
                             (3, 1, True, 1, True),
                             (3, 1, False, 1, True),
                             (3, 1, True, 1, False),
                             (3, 1, False, 1, False),
                             (3, 1, True, -1, False),
                             (10, 3, False, 1, False)
                         ])
def test_generate_batch_with_varying_input(msa_sampler, batch_size, num_positions, mask, leader_length, in_order):
    out = msa_sampler.generate(4, ["AAA", "AAC"], batch_size=batch_size, max_len=3, num_iters=2,
                               num_positions=num_positions, mask=mask, leader_length=leader_length, in_order=in_order)

    assert (len(out) == 4)
    for s in out:
        assert (len(s) == 3)


def test_generate_batch_single_iteration(msa_sampler):
    out = msa_sampler.generate(4, ["AAA", "AAC"], num_iters=1,
                               max_len=5, num_positions=1, in_order=True)

    assert (len(out) == 4)

    assert out[0][1:3] == "AA"
    assert out[1][1:3] == "AC"
    assert out[2][1:3] == "AA"
    assert out[3][1:3] == "AC"


def test_generate_batch_randomly(msa_sampler):
    out = msa_sampler.generate(4, ["AAA", "AAC"], num_iters=1,
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
    indexes = [0, 1, 2, 3]
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


def test_mask_indexes(msa_sampler):
    batch = [
        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
    ]
    target_indexes = [
        [[2, 3], [1, 2], [0, 1]],
        [[0, 1], [2, 1], [3, 2]]
    ]
    msa_sampler.mask_target_indexes(batch, target_indexes)

    assert batch == [
        [[1, 1, 32, 32], [1, 32, 32, 1], [32, 32, 1, 1]],
        [[32, 32, 1, 1], [1, 32, 32, 1], [1, 1, 32, 32]]
    ]


def test_calculate_indexes_no_rollover(msa_sampler):
    indexes = None
    leader_length = 1
    max_len = 5
    rollover = False

    out_indexes, last_i = msa_sampler.calculate_indexes(indexes, leader_length, max_len, rollover)

    assert out_indexes == [2, 3, 4, 5]
    assert last_i == 0


def test_calculate_indexes_with_rollover(msa_sampler):
    indexes = None
    leader_length = 1
    max_len = 5
    rollover = True

    out_indexes, last_i = msa_sampler.calculate_indexes(indexes, leader_length, max_len, rollover)

    assert out_indexes == [1, 2, 3, 4, 5]
    assert last_i == -1


def test_calculate_indexes_when_indexes_supplied(msa_sampler):
    indexes = [2, 3, 4, 5]
    leader_length = 1
    max_len = 5
    rollover = False

    out_indexes, last_i = msa_sampler.calculate_indexes(indexes, leader_length, max_len, rollover)

    assert out_indexes == [2, 3, 4, 5]
    assert last_i == -1


def map_aa_idx_to_tok_set(msa_sampler):
    return set(msa_sampler.model.alphabet.get_tok(idx) for idx in msa_sampler.valid_aa_idx)


def test_allowable_amino_acid_locations_only_contain_standard_aa(msa_sampler):
    standard_toks = set(msa_sampler.model.alphabet.standard_toks)
    actual_allowed = map_aa_idx_to_tok_set(msa_sampler)

    assert actual_allowed.issubset(standard_toks)
    assert actual_allowed == set(ESM_MSA_ALLOWED_AMINO_ACIDS)


def test_allowable_amino_acid_locations_do_not_contain_amino_acids_we_cant_create(msa_sampler):
    actual_allowed = map_aa_idx_to_tok_set(msa_sampler)
    non_single_standard = set("XBUXZO.")

    assert actual_allowed.isdisjoint(non_single_standard)


def test_generate_batch_only_includes_allowed_aa(msa_sampler):
    out = msa_sampler.generate(10, ["AAA", "AAC"], num_iters=1, max_len=25)

    allowed = set(ESM_MSA_ALLOWED_AMINO_ACIDS)
    for sequence in out:
        assert len({s for s in sequence}.difference(allowed)) == 0


MSA_BATCH_EXAMPLE = [
    ('4', 'MSSESELALLRDSVDRLDANLVALLAQR'),
    ('3', 'MSDPDPLAAARERIKALDEQLLALLAER'),
    ('2', 'MSQPNDLPSLRERIDALDRRLVALLAER'),
    ('1', 'MSEEENLKTCREKLSEIDDKIIKLLAER'),
    ('0', 'MTSPDELAAARARIDELDARLVALLAER')]


def test_likelihood_without_masking(msa_sampler):
    msa = MSA_BATCH_EXAMPLE
    likelihood = msa_sampler.log_likelihood(msa, target_index=4, with_masking=False, mask_entire_sequence=False)

    assert likelihood == pytest.approx(-2.059587240219116)


def test_likelihood_with_individual_masking(msa_sampler):
    msa = MSA_BATCH_EXAMPLE
    likelihood = msa_sampler.log_likelihood(msa, target_index=4, with_masking=True, mask_entire_sequence=False)

    assert likelihood == pytest.approx(-0.738074004650116)


def test_likelihood_with_masking_entire_sequence(msa_sampler):
    msa = MSA_BATCH_EXAMPLE
    likelihood = msa_sampler.log_likelihood(msa, target_index=4, with_masking=True, mask_entire_sequence=True)

    assert likelihood == pytest.approx(-2.3227384090423584)
