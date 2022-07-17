from statistics import mean
import pytest
import torch
from pgen import models, esm_sampler
from pgen.esm_sampler import generate_step, ESM_ALLOWED_AMINO_ACIDS


####### Fixtures #######

@pytest.fixture
def esm6(scope="session"):
    return models.ESM6()


@pytest.fixture
def esm_sampler_fixture(esm6, scope="session"):
    sampler = esm_sampler.ESM_sampler(esm6, device="cpu")
    return sampler



###### Tests #######

def test_sampler_init_cpu(esm6):

    sampler = esm_sampler.ESM_sampler(esm6, device="cpu")
    #TODO: test that the model is actually on the cpu

@pytest.mark.skipif(not torch.cuda.is_available(),reason="requires a cuda to be available")
def test_sampler_init_gpu(esm6):
    sampler = esm_sampler.ESM_sampler(esm6,  device="gpu")
    #TODO: test that the model is actually on the gpu


def test_sampler_init_gpu_when_not_available(esm6,mock_no_gpu):
    pytest.raises(Exception, esm_sampler.ESM_sampler, esm6, device="gpu")


def test_get_init_seq_empty(esm_sampler_fixture):
    sampler = esm_sampler_fixture
    out = sampler.get_init_seq("", 5, 1)
    expected = [[32,33,33,33,33,33]]
    assert (out.tolist() == expected)


def test_get_init_seq_string_seed(esm_sampler_fixture):
    sampler = esm_sampler_fixture
    out = sampler.get_init_seq("AA", 5, 1)
    expected = [[32, 5, 5, 33, 33, 33]]
    assert (out.tolist() == expected)


def test_get_init_seq_string_seed_lowercase(esm_sampler_fixture):
    sampler = esm_sampler_fixture
    out = sampler.get_init_seq("aa", 5, 1)
    expected = [[32, 5, 5, 33, 33, 33]]
    assert (out.tolist() == expected)


def test_get_init_seq_string_fails_if_non_standard_supplied(esm_sampler_fixture):
    sampler = esm_sampler_fixture
    try:
        sampler.get_init_seq("X", 5, 1)
        assert False
    except Exception as e:
        assert str(e) == "Invalid input character: X"


def test_get_init_seq_array_of_seeds(esm_sampler_fixture):
    sampler = esm_sampler_fixture
    out = sampler.get_init_seq(["Aa"], 5, 1)
    expected = [[32, 5, 5, 33, 33, 33]]
    assert (out.tolist() == expected)


def test_get_init_seq_array_of_seeds_builds_batch_randomly(esm_sampler_fixture):
    sampler = esm_sampler_fixture
    out = sampler.get_init_seq(["AA", "A"], 5, 3)
    option1 = [32, 5, 5, 33, 33, 33]
    option2 = [32, 5, 33, 33, 33, 33]

    assert len(out) == 3
    for item in out.tolist():
        assert item == option1 or item == option2

def test_generate_batch_equals_seqs(esm_sampler_fixture):
    sampler = esm_sampler_fixture
    out = sampler.generate(4, "", batch_size=4, max_len=10)

    assert (len(out) == 4)
    for s in out:
        assert(len(s) == 10)


@pytest.mark.parametrize("batch_size, num_positions,mask,leader_length,in_order",
                         [
                             (3, 1, True, 1, True),
                             (3, 1, False, 1, True),
                             (3, 1, True, 1, False),
                             (3, 1, False, 1, False),
                             (3, 1, True, -1, False),
                             (10, 3, False, 1, False)
                         ])
def test_generate_batch_with_varying_input(esm_sampler_fixture, batch_size, num_positions, mask, leader_length, in_order):
    out = esm_sampler_fixture.generate(4, "AAAAAAAAAA", batch_size=batch_size, max_len=10, num_iters=2,
                                       num_positions=num_positions, mask=mask, leader_length=leader_length,
                                       in_order=in_order)

    assert (len(out) == 4)
    for s in out:
        assert(len(s) == 10)


def test_generate_batch_greater_than_seqs(esm_sampler_fixture):
    sampler = esm_sampler_fixture
    out = sampler.generate(4, "", batch_size=10, max_len=10)

    assert (len(out) == 4)
    for s in out:
        assert(len(s) == 10)

#def test_esm34_gpu(esm34):
#    gpu_model = esm34.model.cuda()


def test_get_target_index_in_order(esm_sampler_fixture):
    sampler = esm_sampler_fixture
    last_i, target_indexes = sampler.get_target_index_in_order(
        batch_size=2, indexes=[0, 1, 2, 3], next_i=1, num_positions=2)

    assert len(target_indexes) == 2
    assert last_i == 3
    assert target_indexes == [[2, 3], [2, 3]]


def test_get_target_index_randomly(esm_sampler_fixture):
    sampler = esm_sampler_fixture
    indexes = [0, 1, 2, 3]
    target_indexes = sampler.get_random_target_index(
        batch_size=2, indexes=indexes, num_positions=3)

    assert len(target_indexes) == 2
    assert len(target_indexes[0]) == 3
    for item in target_indexes[0]:
        assert item in indexes


def test_mask_indexes(esm_sampler_fixture):
    batch = [
        [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]
    ]
    target_indexes = [
        [2, 3], [1, 2], [0, 1]
    ]
    esm_sampler_fixture.mask_target_indexes(batch, target_indexes)

    assert batch == [
        [1, 1, 33, 33], [1, 33, 33, 1], [33, 33, 1, 1]
    ]


def map_aa_idx_to_tok_set(esm_sampler_fixture):
    return set(esm_sampler_fixture.model.alphabet.get_tok(idx) for idx in esm_sampler_fixture.valid_aa_idx)


def test_allowable_amino_acid_locations_only_contain_standard_aa(esm_sampler_fixture):
    standard_toks = set(esm_sampler_fixture.model.alphabet.standard_toks)
    actual_allowed = map_aa_idx_to_tok_set(esm_sampler_fixture)

    assert actual_allowed.issubset(standard_toks)
    assert actual_allowed == set(ESM_ALLOWED_AMINO_ACIDS)


def test_allowable_amino_acid_locations_do_not_contain_amino_acids_we_cant_create(esm_sampler_fixture):
    actual_allowed = map_aa_idx_to_tok_set(esm_sampler_fixture)
    non_single_standard = set("XBUXZO.-")

    assert actual_allowed.isdisjoint(non_single_standard)


def test_generate_step_without_idx_restriction():
    cnts = {idx: 0 for idx in range(6)}
    for _ in range(1000):
        out = torch.tensor([[.1, .1, .1, .1, .1, .1]])
        gen_idx = 0
        idx = generate_step(out, gen_idx)
        cnts[idx.item()] += 1

    print(cnts)
    assert cnts[0] > 100
    assert cnts[1] > 100
    assert cnts[2] > 100
    assert cnts[3] > 100
    assert cnts[4] > 100
    assert cnts[5] > 100


def test_generate_step_with_idx_restriction():
    valid_idx = [1, 3, 5]
    cnts = {idx: 0 for idx in range(6)}
    for _ in range(1000):
        out = torch.tensor([[.1, .1, .1, .1, .1, .1]])
        gen_idx = 0
        idx = generate_step(out, gen_idx, valid_idx=valid_idx)
        cnts[idx.item()] += 1

    print(cnts)
    assert cnts[0] == 0
    assert cnts[1] > 200
    assert cnts[2] == 0
    assert cnts[3] > 200
    assert cnts[4] == 0
    assert cnts[5] > 200


def test_generate_step_with_idx_restriction_and_top_k():
    valid_idx = [1, 3, 5]
    cnts = {idx: 0 for idx in range(6)}
    for _ in range(1000):
        out = torch.tensor([[.4, .2, .4, .2, .1, .1]])
        gen_idx = 0
        idx = generate_step(out, gen_idx, top_k=2, valid_idx=valid_idx)
        cnts[idx.item()] += 1

    print(cnts)
    assert cnts[0] == 0
    assert cnts[1] > 400
    assert cnts[2] == 0
    assert cnts[3] > 400
    assert cnts[4] == 0
    assert cnts[5] == 0  # excluded from top k


def test_generate_step_with_out_of_order_idx_restriction():
    valid_idx = [3, 5, 1]
    cnts = {idx: 0 for idx in range(6)}
    for _ in range(1000):
        out = torch.tensor([[.4, .2, .4, .2, .1, .1]])
        gen_idx = 0
        idx = generate_step(out, gen_idx, top_k=2, valid_idx=valid_idx)
        cnts[idx.item()] += 1

    print(cnts)
    assert cnts[0] == 0
    assert cnts[1] > 400
    assert cnts[2] == 0
    assert cnts[3] > 400
    assert cnts[4] == 0
    assert cnts[5] == 0  # excluded from top k


def test_generate_batch_only_includes_allowed_aa(esm_sampler_fixture):
    out = esm_sampler_fixture.generate(10, "", batch_size=10, max_len=25)

    allowed = set(ESM_ALLOWED_AMINO_ACIDS)
    for sequence in out:
        assert len({s for s in sequence}.difference(allowed)) == 0


# def test_generate_batch_does_not_leave_unmasked_characters(esm_sampler_fixture):
#     out = esm_sampler_fixture.generate(1, "", num_iters=1, max_len=5, num_positions=1)
#
#     assert "<mask>" not in out[0]

def test_log_likelihood_with_mask(esm_sampler_fixture):
    (seq_prob, pos_probs) = esm_sampler_fixture.log_likelihood("MRHGDISSSNDTVGVAVVNYKMPRLHTAAEVLDNAR")
    assert seq_prob == pytest.approx(-2.843970775604248)
    assert seq_prob == pytest.approx(mean(pos_probs))

    (seq_prob, pos_probs) = esm_sampler_fixture.log_likelihood("LTWEEQCKTCKGCRYNFQHE")
    assert seq_prob == pytest.approx(-3.0787816047668457)
    assert seq_prob == pytest.approx(mean(pos_probs))

    (seq_prob, pos_probs) = esm_sampler_fixture.log_likelihood("ACDEFGHIKLMNPQRSTVWY")
    assert seq_prob == pytest.approx(-3.290297269821167)
    assert seq_prob == pytest.approx(mean(pos_probs))

def test_log_likelihood_without_mask(esm_sampler_fixture):
    for s,v in [("MRHGDISSSNDTVGVAVVNYKMPRLHTAAEVLDNAR",-2.1893723011016846),("LTWEEQCKTCKGCRYNFQHE",-2.3772685527801514),("ACDEFGHIKLMNPQRSTVWY",-2.412991762161255)]:
        (seq_prob, pos_probs) = esm_sampler_fixture.log_likelihood(s, with_masking=False)
        assert seq_prob == pytest.approx(v)
        assert seq_prob == pytest.approx(mean(pos_probs))

def test_log_likelihood_batch_with_mask(esm_sampler_fixture):
    input_seq = ["MRHGDISSSNDTVGVAVVNYKMPRLHTAAEVLDNAR", "LTWEEQCKTCKGCRYNFQHE", "ACDEFGHIKLMNPQRSTVWY"]
    mask_results = list(esm_sampler_fixture.log_likelihood_batch(input_seq, with_masking=True))

    assert mask_results[0][0] == pytest.approx(-2.843970775604248)
    assert mask_results[0][0] == pytest.approx(mean(mask_results[0][1]))
    assert mask_results[1][0] == pytest.approx(-3.0787816047668457)
    assert mask_results[1][0] == pytest.approx(mean(mask_results[1][1]))
    assert mask_results[2][0] == pytest.approx(-3.290297269821167)
    assert mask_results[2][0] == pytest.approx(mean(mask_results[2][1]))


def test_log_likelihood_batch_without_mask(esm_sampler_fixture):
    input_seq = ["MRHGDISSSNDTVGVAVVNYKMPRLHTAAEVLDNAR", "LTWEEQCKTCKGCRYNFQHE", "ACDEFGHIKLMNPQRSTVWY"]
    no_mask_results = list(esm_sampler_fixture.log_likelihood_batch(input_seq, with_masking=False))
    assert no_mask_results[0][0] == pytest.approx(-2.1893723011016846)
    assert no_mask_results[0][0] == pytest.approx(mean(no_mask_results[0][1]))
    assert no_mask_results[1][0] == pytest.approx(-2.3772685527801514)
    assert no_mask_results[1][0] == pytest.approx(mean(no_mask_results[1][1]))
    assert no_mask_results[2][0] == pytest.approx(-2.412991762161255)
    assert no_mask_results[2][0] == pytest.approx(mean(no_mask_results[2][1]))


@pytest.mark.parametrize("mask_distance,expected", [(1, (-2.7889750003814697, -3.2179431915283203)),
                                                    (2, (-2.82377028465271, -3.142765522003174)),
                                                    (5, (-2.8046181201934814, -3.0614192485809326)),
                                                    (10, (-2.8243350982666016, -3.0533814430236816)),
                                                    (20, (-2.8372862339019775, -3.0787816047668457)),  # input 2 is 20 chars
                                                    (40, (-2.843970775604248, -3.0787816047668457))  # input 1 is 36 chars
                                                    ])
def test_likelihood_batch_with_individual_masking_distance(esm_sampler_fixture, mask_distance, expected):
    input_seq = ["MRHGDISSSNDTVGVAVVNYKMPRLHTAAEVLDNAR", "LTWEEQCKTCKGCRYNFQHE"]
    actual = list(esm_sampler_fixture.log_likelihood_batch(input_seq, with_masking=True, mask_distance=mask_distance))

    assert actual[0][0] == pytest.approx(expected[0])
    assert actual[0][0] == pytest.approx(mean(actual[0][1]))
    assert actual[1][0] == pytest.approx(expected[1])
    assert actual[1][0] == pytest.approx(mean(actual[1][1]))


@pytest.mark.parametrize("batch_size", [1, 2, 5, 100])
@pytest.mark.parametrize("mask_distance,expected", [(1, (-2.7889750003814697, -3.2179431915283203)),
                                                    (2, (-2.82377028465271, -3.142765522003174)),
                                                    (5, (-2.8046181201934814, -3.0614192485809326)),
                                                    ])
def test_likelihood_batch_handles_batch_sizes(esm_sampler_fixture, batch_size, mask_distance, expected):
    input_seq = ["MRHGDISSSNDTVGVAVVNYKMPRLHTAAEVLDNAR", "LTWEEQCKTCKGCRYNFQHE"]
    actual = list(esm_sampler_fixture.log_likelihood_batch(input_seq, with_masking=True, mask_distance=mask_distance, batch_size=batch_size))

    assert actual[0][0] == pytest.approx(expected[0])
    assert actual[0][0] == pytest.approx(mean(actual[0][1]))
    assert actual[1][0] == pytest.approx(expected[1])
    assert actual[1][0] == pytest.approx(mean(actual[1][1]))
