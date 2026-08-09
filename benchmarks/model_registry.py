from sparsewalker.models import MostPop, SASRec, ESASRec, GRU4Rec, FMLPRec, BERT4Rec, Mamba4Rec, HAS_MAMBA, SparseWalker, SparseWalkerE2E

LEARNED=["GRU4Rec","SASRec","SASRec+SS","eSASRec","BERT4Rec","FMLP-Rec","Mamba4Rec","SparseWalker","SparseWalker-E2E"]

def build_model(name,n_items,max_len,d=64,layers=2,heads=2,inner=256,dropout=.2):
    if name=="SASRec" or name=="SASRec+SS": return SASRec(n_items,max_len,d,layers,heads,inner,dropout)
    if name=="eSASRec": return ESASRec(n_items,max_len,d,layers,heads,inner,dropout)
    if name=="GRU4Rec": return GRU4Rec(n_items,max_len,d,layers,dropout)
    if name=="FMLP-Rec": return FMLPRec(n_items,max_len,d,layers,inner,dropout)
    if name=="BERT4Rec": return BERT4Rec(n_items,max_len,d,layers,heads,inner,dropout)
    if name=="Mamba4Rec":
        if not HAS_MAMBA: raise RuntimeError("mamba_ssm unavailable; portable no-compile Mamba is queued")
        return Mamba4Rec(n_items,max_len,d,1,dropout)
    if name=="SparseWalker": return SparseWalker(n_items,max_len,d,layers,256,16,8,2,4,.25)
    if name=="SparseWalker-E2E": return SparseWalkerE2E(n_items,max_len,d,layers,256,16,8,2,4,.25,128)
    raise KeyError(name)

def loss_mode(name):
    return "ss" if name in ("SASRec+SS","eSASRec") else "full"
