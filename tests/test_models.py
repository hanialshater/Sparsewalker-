import torch
from sparsewalker.models import SASRec, ESASRec, GRU4Rec, FMLPRec, HSTU, SparseWalker

def test_sasrec_beauty_param_count():
    assert sum(p.numel() for p in SASRec(12101,50).parameters())==877824

def test_sequence_models_shapes():
    x=torch.tensor([[1,2,3,0]])
    models=[SASRec(100,4,d=16,layers=1,heads=2,inner=32),ESASRec(100,4,d=16,layers=1,heads=2,inner=32),GRU4Rec(100,4,d=16,layers=1),FMLPRec(100,4,d=16,layers=1,inner=32),SparseWalker(100,4,d=16,layers=1,side=16,h=4,active=4,top_side=2,degree=2)]
    for m in models: assert m.encode(x).shape==(1,4,16)

def test_hstu_shape():
    m=HSTU(max_item_id=100,d_model=16,n=8,layers=1,heads=1,dqk=16,dv=16,dropout=0.0,l2_eps=1e-6)
    ids=torch.tensor([[1,2,3,0,0,0,0,0]]); ts=torch.arange(8).view(1,8)+1; lengths=torch.tensor([3])
    assert m.encode_all(ids,ts,lengths).shape==(1,8,16)
