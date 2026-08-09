import triton
import triton.language as tl

@triton.jit
def route(ITEM,EMB,QW,LK,RK,FI,FM,Q,D:tl.constexpr,H:tl.constexpr,S:tl.constexpr):
 d=tl.arange(0,64); h=tl.arange(0,16); s=tl.arange(0,256); it=tl.load(ITEM).to(tl.int32)
 x=tl.load(EMB+it*D+d,mask=d<D,other=0.).to(tl.float32)*8.
 wl=tl.load(QW+h[:,None]*D+d[None,:],mask=(h[:,None]<H)&(d[None,:]<D),other=0.).to(tl.float32)
 wr=tl.load(QW+H*D+h[:,None]*D+d[None,:],mask=(h[:,None]<H)&(d[None,:]<D),other=0.).to(tl.float32)
 wc=tl.load(QW+2*H*D+h[:,None]*D+d[None,:],mask=(h[:,None]<H)&(d[None,:]<D),other=0.).to(tl.float32)
 ql=tl.sum(wl*x[None,:],1); qr=tl.sum(wr*x[None,:],1); qc=tl.sum(wc*x[None,:],1)
 ql/=tl.sqrt(tl.sum(ql*ql,0)+1e-8); qr/=tl.sqrt(tl.sum(qr*qr,0)+1e-8); qc/=tl.sqrt(tl.sum(qc*qc,0)+1e-8); tl.store(Q+h,qc,mask=h<H)
 lk=tl.load(LK+s[:,None]*H+h[None,:],mask=(s[:,None]<S)&(h[None,:]<H),other=0.).to(tl.float32)
 rk=tl.load(RK+s[:,None]*H+h[None,:],mask=(s[:,None]<S)&(h[None,:]<H),other=0.).to(tl.float32)
 ls=tl.sum(lk*ql[None,:],1); rs=tl.sum(rk*qr[None,:],1)
 li0=tl.argmax(ls,0); lv0=tl.max(ls,0); li1=tl.argmax(tl.where(s==li0,-1e20,ls),0); lv1=tl.max(tl.where(s==li0,-1e20,ls),0)
 ri0=tl.argmax(rs,0); rv0=tl.max(rs,0); ri1=tl.argmax(tl.where(s==ri0,-1e20,rs),0); rv1=tl.max(tl.where(s==ri0,-1e20,rs),0)
 f=tl.arange(0,4); lid=tl.where(f<2,li0,li1); lv=tl.where(f<2,lv0,lv1); rid=tl.where((f%2)==0,ri0,ri1); rv=tl.where((f%2)==0,rv0,rv1)
 z=lv+rv; p=tl.exp(z-tl.max(z,0)); p/=tl.sum(p,0)+1e-8
 tl.store(FI+f,(lid*S+rid).to(tl.int32)); tl.store(FM+f,p)

@triton.jit
def walk(SID,SM,FI,FM,Q,DEST,EDGE,DK,NI,NM,K:tl.constexpr,DEG:tl.constexpr,H:tl.constexpr):
 k=tl.arange(0,K); ms=tl.arange(0,16); path=tl.arange(0,32); h=tl.arange(0,16)
 ids=tl.load(SID+k).to(tl.int32); mass=tl.load(SM+k).to(tl.float32); f=tl.arange(0,4); fi=tl.load(FI+f).to(tl.int32); fm=tl.load(FM+f).to(tl.float32); q=tl.load(Q+h,mask=h<H,other=0.).to(tl.float32)
 for _ in range(2):
  oi=tl.minimum(ms,K-1); ni=tl.maximum(tl.minimum(ms-K,3),0)
  mid=tl.where(ms<K,tl.gather(ids,oi,0),tl.where(ms<K+4,tl.gather(fi,ni,0),0))
  mm=tl.where(ms<K,.75*tl.gather(mass,oi,0),tl.where(ms<K+4,.25*tl.gather(fm,ni,0),-1e20))
  ci=tl.full((K,),0,tl.int32); cm=tl.zeros((K,),tl.float32); sc=mm
  for j in range(K):
   ix=tl.argmax(sc,0); v=tl.max(sc,0); cid=tl.sum(tl.where(ms==ix,mid,0),0); ci=tl.where(k==j,cid,ci); cm=tl.where(k==j,v,cm); sc=tl.where(ms==ix,-1e20,sc)
  cm/=tl.sum(cm,0)+1e-8; src=path//DEG; es=path%DEG; sid=tl.gather(ci,src,0); gi=sid*DEG+es; dest=tl.load(DEST+gi).to(tl.int32); st=tl.load(EDGE+gi).to(tl.float32)
  key=tl.load(DK+dest[:,None]*H+h[None,:],mask=h[None,:]<H,other=0.).to(tl.float32); ctx=tl.sum(key*q[None,:],1)
  score=tl.reshape(st+ctx,(K,DEG)); rm=tl.max(score,1); prob=tl.exp(score-rm[:,None]); prob/=tl.sum(prob,1)[:,None]+1e-8; pm=tl.reshape(prob*cm[:,None],(32,))
  nids=tl.full((K,),0,tl.int32); nm=tl.zeros((K,),tl.float32); sc=pm
  for j in range(K):
   ix=tl.argmax(sc,0); v=tl.max(sc,0); did=tl.sum(tl.where(path==ix,dest,0),0); nids=tl.where(k==j,did,nids); nm=tl.where(k==j,v,nm); sc=tl.where(path==ix,-1e20,sc)
  ids=nids; mass=nm/(tl.sum(nm,0)+1e-8)
 tl.store(NI+k,ids); tl.store(NM+k,mass)

@triton.jit
def readout(ITEM,SID,SM,CV,MW,NW,NB,EMB,HID,K:tl.constexpr,D:tl.constexpr):
 k=tl.arange(0,K); d=tl.arange(0,64); sid=tl.load(SID+k).to(tl.int32); sm=tl.load(SM+k).to(tl.float32)
 val=tl.load(CV+sid[:,None]*D+d[None,:],mask=d[None,:]<D,other=0.).to(tl.float32); msg=tl.sum(val*sm[:,None],0)
 w=tl.load(MW+d[:,None]*D+d[None,:],mask=(d[:,None]<D)&(d[None,:]<D),other=0.).to(tl.float32); proj=tl.sum(w*msg[None,:],1)
 it=tl.load(ITEM).to(tl.int32); iv=tl.load(EMB+it*D+d,mask=d<D,other=0.).to(tl.float32); r=8.*iv+proj
 mu=tl.sum(r,0)/D; c=r-mu; var=tl.sum(c*c,0)/D; nw=tl.load(NW+d,mask=d<D,other=1.).to(tl.float32); nb=tl.load(NB+d,mask=d<D,other=0.).to(tl.float32)
 tl.store(HID+d,c*tl.rsqrt(var+1e-5)*nw+nb,mask=d<D)

@triton.jit
def term_block(HID,SID,SUP,EMB,BI,BS,DEG:tl.constexpr,D:tl.constexpr,TOTAL:tl.constexpr,KEEP:tl.constexpr):
 bid=tl.program_id(0); lane=tl.arange(0,128); d=tl.arange(0,64); p=bid*128+lane; valid=p<TOTAL; src=p//DEG; es=p%DEG
 sid=tl.load(SID+src,mask=valid,other=0).to(tl.int32); it=tl.load(SUP+sid*DEG+es,mask=valid,other=0).to(tl.int32); h=tl.load(HID+d,mask=d<D,other=0.).to(tl.float32)
 e=tl.load(EMB+it[:,None]*D+d[None,:],mask=valid[:,None]&(d[None,:]<D),other=0.).to(tl.float32); sc=tl.where(valid&(it!=0),tl.sum(e*h[None,:],1),-1e20)
 for j in range(KEEP):
  ix=tl.argmax(sc,0); v=tl.max(sc,0); ci=tl.sum(tl.where(lane==ix,it,0),0); tl.store(BI+bid*KEEP+j,ci); tl.store(BS+bid*KEEP+j,v); sc=tl.where(it==ci,-1e20,sc)

@triton.jit
def term_merge(BI,BS,OUT,COUNT:tl.constexpr,BLOCK:tl.constexpr):
 l=tl.arange(0,BLOCK); valid=l<COUNT; it=tl.load(BI+l,mask=valid,other=0).to(tl.int32); sc=tl.load(BS+l,mask=valid,other=-1e20).to(tl.float32)
 for j in range(10):
  ix=tl.argmax(sc,0); ci=tl.sum(tl.where(l==ix,it,0),0); tl.store(OUT+j,ci); sc=tl.where(it==ci,-1e20,sc)
