import pytest
import torch
from pgen import models, esm_sampler


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


def test_generate_batch_equals_seqs(esm_sampler_fixture):
    sampler = esm_sampler_fixture
    out = sampler.generate(4, "", batch_size=4, max_len=10)
    
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
        batch_size=2, indexes=[0,1,2,3], next_i=1, num_positions=2)

    assert len(target_indexes) == 2
    assert last_i == 3
    assert target_indexes == [[2, 3], [2, 3]]


def test_get_target_index_randomly(esm_sampler_fixture):
    sampler = esm_sampler_fixture
    indexes = [0,1,2,3]
    target_indexes = sampler.get_random_target_index(
        batch_size=2, indexes=indexes, num_positions=3)

    assert len(target_indexes) == 2
    assert len(target_indexes[0]) == 3
    for item in target_indexes[0]:
        assert item in indexes
