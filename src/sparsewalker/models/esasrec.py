from .sasrec import SASRec

class ESASRec(SASRec):
    """LiGR/SwiGLU SASRec architecture. Pair with sampled softmax for the eSASRec recipe."""
    def __init__(self,*args,**kwargs):
        kwargs["ligr"]=True
        super().__init__(*args,**kwargs)
