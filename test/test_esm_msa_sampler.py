import enum
from statistics import mean
import pytest
from pgen import models, esm_msa_sampler
from pgen import likelihood_esm_msa
from io import StringIO
from pgen.esm_msa_sampler import ESM_MSA_ALLOWED_AMINO_ACIDS
import tempfile
import pandas as pd

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


@pytest.fixture()
def msa_batch_example():
    return [[
        'MSSESELALLRDSVDRLDANLVALLAQR',
        'MSDPDPLAAARERIKALDEQLLALLAER',
        'MSQPNDLPSLRERIDALDRRLVALLAER',
        'MSEEENLKTCREKLSEIDDKIIKLLAER',
        'MTSPDELAAARARIDELDARLVALLAER'], [
        'MSSESELALLRDSVDRLDANLVALLAQRLAVARQVGRYKQLHGL',
        'MSDPDPLAAARERIKALDEQLLALLAERVACALEVGRLKATHGL',
        'MSQPNDLPSLRERIDALDRRLVALLAERAQTVHEVGRLKAERGL',
        'MSEEENLKTCREKLSEIDDKIIKLLAERFKIAEAIGKYKAENGL',
        'MTSPDELAAARARIDELDARLVALLAERRAAVESVGRLKAESGL']]


def test_likelihood_without_masking(msa_sampler, msa_batch_example):
    (seq_prob, pos_probs) = msa_sampler.log_likelihood(msa_batch_example[0], target_index=4, with_masking=False)
    assert seq_prob == pytest.approx(-0.06248655170202255)
    assert mean(pos_probs) == pytest.approx(seq_prob)

    (seq_prob, pos_probs) = msa_sampler.log_likelihood(msa_batch_example[1], target_index=4, with_masking=False)
    assert seq_prob == pytest.approx(-0.1334836632013321)
    assert mean(pos_probs) == pytest.approx(seq_prob)


def test_likelihood_with_individual_masking(msa_sampler, msa_batch_example):
    (seq_prob, pos_probs) = msa_sampler.log_likelihood(msa_batch_example[0], target_index=4, with_masking=True)
    assert seq_prob == pytest.approx(-0.738074004650116)
    assert mean(pos_probs) == pytest.approx(seq_prob)
    
    (seq_prob, pos_probs)  = msa_sampler.log_likelihood(msa_batch_example[1], target_index=4, with_masking=True)
    assert seq_prob == pytest.approx(-0.876354455947876)
    assert mean(pos_probs) == pytest.approx(seq_prob)


def test_likelihood_with_masking_entire_sequence(msa_sampler, msa_batch_example):
    (seq_prob, pos_probs) = msa_sampler.log_likelihood(msa_batch_example[0], target_index=4, with_masking=True, mask_distance=1)
    assert seq_prob == pytest.approx(-2.3227384090423584)
    assert mean(pos_probs) == pytest.approx(seq_prob)

    (seq_prob, pos_probs) = msa_sampler.log_likelihood(msa_batch_example[1], target_index=4, with_masking=True, mask_distance=1)
    assert seq_prob == pytest.approx(-2.8657891750335693)
    assert mean(pos_probs) == pytest.approx(seq_prob)


def test_likelihood_with_masking_entire_sequence_skip_gap(msa_sampler, msa_batch_example):
    # The last one contributes -2.1356, of 28 characters.. so (-65.0367 - (-2.1356))/27
    msa_batch_example[0][-1] = "MTSPDELAAARARIDELDARLVALLAE-"
    (seq_prob, pos_probs) = msa_sampler.log_likelihood(msa_batch_example[0], target_index=4, with_masking=True,
                                      mask_distance=1, count_gaps=False)
    assert seq_prob == pytest.approx(-2.32967)
    assert mean(pos_probs) == pytest.approx(seq_prob)

def test_likelihood_batch_without_masking(msa_sampler, msa_batch_example):
    result = list(msa_sampler.log_likelihood_batch(msa_batch_example, target_index=4, with_masking=False))
    assert result[0][0] == pytest.approx(-0.06248655170202255)
    assert mean(result[0][1]) == pytest.approx(result[0][0])
    assert result[1][0] == pytest.approx(-0.1334836632013321)
    assert mean(result[1][1]) == pytest.approx(result[1][0])

def test_likelihood_batch_with_individual_masking(msa_sampler, msa_batch_example):
    result = list(msa_sampler.log_likelihood_batch(msa_batch_example, target_index=4, with_masking=True))
    assert result[0][0] == pytest.approx(-0.738074004650116)
    assert mean(result[0][1]) == pytest.approx(result[0][0])
    assert result[1][0] == pytest.approx(-0.876354455947876)
    assert mean(result[1][1]) == pytest.approx(result[1][0])

def test_likelihood_batch_with_masking_entire_sequence(msa_sampler, msa_batch_example):
    result = list(msa_sampler.log_likelihood_batch(msa_batch_example, target_index=4, with_masking=True,
                                              mask_distance=1))
    assert result[0][0] == pytest.approx(-2.3227384090423584)
    assert mean(result[0][1]) == pytest.approx(result[0][0])
    assert result[1][0] == pytest.approx(-2.8657891750335693)
    assert mean(result[1][1]) == pytest.approx(result[1][0])


@pytest.mark.parametrize("mask_distance,expected", [(1, (-2.3227384090423584, -2.8657891750335693)),
                                                    (2, (-0.809113621711731, -0.9949756264686584)),
                                                    (5, (-0.7348969578742981, -0.883063554763794)),
                                                    (10, (-0.7481335401535034, -0.9041081070899963)),
                                                    (28, (-0.738074004650116, -0.9032351970672607)),  # input 1 is 28 chars
                                                    (50, (-0.738074004650116, -0.876354455947876))  # input 2 is 44 chars
                                                    ])
def test_likelihood_batch_with_individual_masking_distance(msa_sampler, msa_batch_example, mask_distance, expected):
    result = list(msa_sampler.log_likelihood_batch(msa_batch_example, target_index=4, with_masking=True,
                                              mask_distance=mask_distance))
    assert result[0][0] == pytest.approx(expected[0])
    assert mean(result[0][1]) == pytest.approx(result[0][0])
    assert result[1][0] == pytest.approx(expected[1])
    assert mean(result[1][1]) == pytest.approx(result[1][0])


@pytest.mark.parametrize("batch_size", [1, 2, 5, 100])
@pytest.mark.parametrize("mask_distance,expected", [(1, (-2.3227384090423584, -2.8657891750335693)),
                                                    (2, (-0.809113621711731, -0.9949756264686584)),
                                                    (5, (-0.7348969578742981, -0.883063554763794)),
                                                    ])
def test_likelihood_batch_handles_batch_sizes(msa_sampler, msa_batch_example, batch_size, mask_distance, expected):
    result = list(msa_sampler.log_likelihood_batch(msa_batch_example, target_index=4, with_masking=True,
                                              mask_distance=mask_distance, batch_size=batch_size))
    assert result[0][0] == pytest.approx(expected[0])
    assert mean(result[0][1]) == pytest.approx(result[0][0])
    assert result[1][0] == pytest.approx(expected[1])
    assert mean(result[1][1]) == pytest.approx(result[1][0])


@pytest.mark.parametrize("input_index,input_name,expected,mask_off,mask_distance", [
    (0, "0", -0.06248655170202255, True, float("inf")),
    (1, "0", -0.1334836632013321, True, float("inf")),
    (0, "0", -0.738074004650116, False, float("inf")),
    (1, "0", -0.876354455947876, False, float("inf")),
    (0, "0", -2.3227384090423584, False, 1),
    (1, "0", -2.8657891750335693, False, 1),
])
def test_likelihood_executable_no_mask(msa_sampler, msa_batch_example, input_index, input_name, expected, mask_off, mask_distance):
    reference_sequence = f">{input_name}\n{msa_batch_example[input_index][-1]}\n"
    input_handle = StringIO(reference_sequence)
    msa_string = "\n".join([f">{n}\n{s}" for n,s in enumerate(msa_batch_example[input_index][:-1]) ]) + "\n"
    alignment_handle = StringIO(msa_string)
    seqlen = len(msa_batch_example[input_index][-1])
    with tempfile.TemporaryDirectory() as tmpdirname:
        output_handle = StringIO()
        # tmpdirname = "tmp_out"
        positionwise_output_path = tmpdirname + "/positionwise_output.tsv"
        likelihood_esm_msa.main(input_handle, output_handle, masking_off=mask_off, sampler=msa_sampler, mask_distance=mask_distance,
            reference_msa_handle=alignment_handle, delete_insertions=False, batch_size=1, subset_strategy="in_order",alignment_size=4,positionwise=positionwise_output_path)
        output_handle.seek(0)
        name, score = output_handle.readline().split()
        assert name == "id"
        assert score == "esm-msa"
        out_n, out_v = output_handle.readline().split()
        out_v = float(out_v)
        assert out_n == input_name
        assert out_v == pytest.approx(expected)

        poswise_output = open(positionwise_output_path).readlines()
        assert len(poswise_output) == 2
        assert len(poswise_output[1].split()[1].split(";")) == seqlen


# TODO: this is not a great test.
def test_likelihood_executable_realign(msa_sampler):
    input_aln = [
     'AKDKG-LDINSAEKFFEALHSESIKHQINVMEK-',
     'N--EGPLDKESVRTIYELLMSSSHDIQAEQRQRE',
     'GQEQN-LDSNYISQVYHTIIEQSVLSQQEFNNRF',
     'N--PGPLDDSAIISMFNLIMDGSRILEKKQTNQH',
     'GKEKQ-LDPQYVSQIFHTIIEDSVLYQRS-----']

    query_name = "xyz"
    reference_sequence = f">{query_name}\n{input_aln[-1]}\n"
    reference_sequence_unaligned = f">{query_name}\n{input_aln[-1].replace('-','')}\n"
    aligned_input_handle = StringIO(reference_sequence)
    unaligned_input_handle = StringIO(reference_sequence_unaligned)
    msa_string = "\n".join([f">{n}\n{s}" for n,s in enumerate(input_aln[:-1]) ]) + "\n"
    alignment_handle = StringIO(msa_string)

    aligned_output_handle = StringIO()
    likelihood_esm_msa.main(aligned_input_handle, aligned_output_handle, masking_off=True, sampler=msa_sampler,
        reference_msa_handle=alignment_handle, delete_insertions=False, batch_size=1, subset_strategy="in_order",alignment_size=4, unaligned_queries=False)
    aligned_output_handle.seek(0)

    unaligned_output_handle = StringIO()
    alignment_handle.seek(0)
    likelihood_esm_msa.main(unaligned_input_handle, unaligned_output_handle, masking_off=True, sampler=msa_sampler,
        reference_msa_handle=alignment_handle, delete_insertions=False, batch_size=1, subset_strategy="in_order",alignment_size=4, unaligned_queries=True)
    unaligned_output_handle.seek(0)
    aligned_out_n, aligned_out_v = aligned_output_handle.readline().split()
    assert aligned_out_n == "id"
    assert aligned_out_v == "esm-msa"

    aligned_out_n, aligned_out_v = aligned_output_handle.readline().split()
    aligned_out_v = float(aligned_out_v)
    assert aligned_out_n == query_name

    unaligned_out_n, unaligned_out_v = unaligned_output_handle.readline().split()
    assert unaligned_out_n == "id"
    assert unaligned_out_v == "esm-msa"
    unaligned_out_n, unaligned_out_v = unaligned_output_handle.readline().split()
    unaligned_out_v = float(unaligned_out_v)
    assert unaligned_out_n == query_name

    assert unaligned_out_v == pytest.approx(aligned_out_v)


# maybe also not a great test?
def test_log_likelihood_count_gaps(msa_sampler):
    input_aln = [
     'RINVMEK-',
     'IAEQRQRE',
     'GQEFNNRF',
     'FKKQTNQH',
     'Q-------']

    no_gaps = msa_sampler.log_likelihood(input_aln, target_index=4, with_masking=False, count_gaps=False)
    gaps = msa_sampler.log_likelihood(input_aln, target_index=4, with_masking=False, count_gaps=True)

    # adding gaps to the calculation should improve (make closer to zero) the score because they should be pretty predictable from the neighboring gaps.
    assert no_gaps < gaps


def test_log_likelihood_count_gaps_2(msa_sampler):
    input_aln = [
     'RINVMEKH',
     'IAEQRQRE',
     'GQEFNNRF',
     'FKKQTNQH',
     'QQQ-Q-QQ']

    no_gaps = msa_sampler.log_likelihood(input_aln, target_index=4, with_masking=False, count_gaps=False)
    gaps = msa_sampler.log_likelihood(input_aln, target_index=4, with_masking=False, count_gaps=True)

    # removing the gap should improve the score, because the Qs should be predicted from the neighbor Qs, and the gaps will not be as much.
    assert no_gaps > gaps

def test_likelihood_executable_top_hits(msa_sampler):
    input_aln = [
     'AKDKGLDINSAEKFFEALHSESIKHQINVMEK',
     'NEGPLDKESVRTIYELLMSSSHDIQAEQRQRE',
     'GQEQNLDSNYISQVYHTIIEQSVLSQQEFNNRF',
     'NPGPLDDSAIISMFNLIMDGSRILEKKQTNQH',
     'GKEKQLDPQYVSQIFHTIIEDSVLYQRS']

    query_name = "xyz"
    reference_sequence = f">{query_name}\n{input_aln[-1]}\n"
    unaligned_input_handle = StringIO(reference_sequence)
    reference_seqs_string = "\n".join([f">{n}\n{s}" for n,s in enumerate(input_aln[:-1]) ]) + "\n"
    alignment_handle = StringIO(reference_seqs_string)

    output_handle = StringIO()
    likelihood_esm_msa.main(unaligned_input_handle, output_handle, masking_off=True, sampler=msa_sampler,
        reference_msa_handle=alignment_handle, delete_insertions=False, batch_size=1, subset_strategy="top_hits",alignment_size=2,)
    output_handle.seek(0)


    # aligned_out_n, aligned_out_v = aligned_output_handle.readline().split()
    # aligned_out_v = float(aligned_out_v)
    # assert aligned_out_n == query_name

    # unaligned_out_n, unaligned_out_v = unaligned_output_handle.readline().split()
    # unaligned_out_v = float(unaligned_out_v)
    # assert unaligned_out_n == query_name

    # assert unaligned_out_v == pytest.approx(aligned_out_v)

@pytest.mark.parametrize("input_list,num_partitions,expected", [
    ([1,2,3,4,5,6,7,8,9,10], 1, [[1,2,3,4,5,6,7,8,9,10]]),
    ([1,2,3,4,5,6,7,8,9,10], 2, [[1,2,3,4,5],[6,7,8,9,10]]),
    ([1,2,3,4,5,6,7,8,9,10], 3, [[1,2,3,4],[5,6,7],[8,9,10]]),
    ([1,2,3,4,5,6,7,8,9,10], 4, [[1,2,3],[4,5,6],[7,8],[9,10]]),
    ([1,2,3,4,5,6,7,8,9,10], 5, [[1,2],[3,4],[5,6],[7,8],[9,10]]),
    ([1,2,3,4,5,6,7,8,9,10], 6, [[1,2],[3,4],[5,6],[7,8],[9],[10]]),
    ([1,2,3,4,5,6,7,8,9,10], 7, [[1,2],[3,4],[5,6],[7],[8],[9],[10]]),
    ([1,2,3,4,5,6,7,8,9,10], 8, [[1,2],[3,4],[5],[6],[7],[8],[9],[10]]),
    ([1,2,3,4,5,6,7,8,9,10], 9, [[1,2],[3],[4],[5],[6],[7],[8],[9],[10]]),
    ([1,2,3,4,5,6,7,8,9,10], 10, [[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]),
    ([1,2,3,4,5,6,7,8,9,10], 11, [[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]),
    ([1,2,3,4,5,6,7,8,9,10], 600, [[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]),
])
def test_partition_1(input_list,num_partitions,expected):
    partitions = esm_msa_sampler.partition(input_list, num_partitions)

    assert len(partitions) == len(expected)
    for i, part in enumerate(partitions):
        assert part == expected[i]

    

def test_generate_single(msa_sampler):
    out = msa_sampler.generate_single(["AAA", "AAA", "GGG"], steps=1,passes=3,burn_in=0)

    assert len(out) == 3
    assert out == "AAA"

    # assert out[0][1:3] == "AA"
    # assert out[1][1:3] == "AC"
    # assert out[2][1:3] == "AA"
    # assert out[3][1:3] == "AC"