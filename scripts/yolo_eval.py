import sys, glob, os
from collections import defaultdict
from ultralytics import YOLO
from caridence.schema import BBox
from caridence.metrics import iou

WEIGHTS = sys.argv[1] if len(sys.argv)>1 and not sys.argv[1].startswith("--") else "runs/detect/outputs/yolo/cardd/weights/best.pt"
AUG = "--aug" in sys.argv
IMGSZ = 1280 if "--1280" in sys.argv else 1024
TEST_IMG="data/kaggle/cardd_yolo/test/images"; TEST_LBL="data/kaggle/cardd_yolo/test/labels"
NAMES=["dent","scratch","crack","glass shatter","lamp broken","tire flat"]

def cb(cx,cy,w,h):
    return BBox(x=min(0.999,max(0,cx-w/2)),y=min(0.999,max(0,cy-h/2)),w=min(max(w,1e-6),1.0),h=min(max(h,1e-6),1.0))
def load_gt(p):
    o=[]
    if os.path.exists(p):
        for ln in open(p):
            q=ln.split()
            if len(q)>=5: o.append((int(q[0]),cb(*map(float,q[1:5]))))
    return o

model=YOLO(WEIGHTS)
imgs=sorted(glob.glob(os.path.join(TEST_IMG,"*.jpg")))
AP=[]; AG=[]
for im in imgs:
    r=model.predict(im,conf=0.001,iou=0.6,imgsz=IMGSZ,augment=AUG,verbose=False)[0]
    pr=[]
    for b in r.boxes:
        cx,cy,w,h=b.xywhn[0].tolist(); pr.append((int(b.cls),float(b.conf),cb(cx,cy,w,h)))
    AP.append(pr); AG.append(load_gt(os.path.join(TEST_LBL,os.path.splitext(os.path.basename(im))[0]+".txt")))

def match(pf,gts,it):
    c=[]
    for pi,(pc,pb) in enumerate(pf):
        for gi,(gc,gb) in enumerate(gts):
            if pc==gc:
                o=iou(pb,gb)
                if o>=it: c.append((o,pi,gi))
    c.sort(reverse=True); up,ug=set(),set()
    for o,pi,gi in c:
        if pi in up or gi in ug: continue
        up.add(pi); ug.add(gi)
    return ug
def rec(ct,it):
    GT=M=0; pg=defaultdict(int); pm=defaultdict(int)
    for pr,gts in zip(AP,AG):
        pf=[(c,bb) for c,cf,bb in pr if cf>=ct]; GT+=len(gts)
        ug=match(pf,gts,it); M+=len(ug)
        for gc,gb in gts: pg[gc]+=1
        for gi in ug: pm[gts[gi][0]]+=1
    return (M/GT if GT else 0),pg,pm
def prec(ct,it):
    TP=PP=0
    for pr,gts in zip(AP,AG):
        pf=[(c,bb) for c,cf,bb in pr if cf>=ct]; PP+=len(pf)
        TP+=len(match(pf,gts,it))
    return TP/PP if PP else 0
def presence(ct):
    pa=h=0
    for pr,gts in zip(AP,AG):
        pc={c for c,cf,bb in pr if cf>=ct}; gc={c for c,bb in gts}
        for c in gc:
            pa+=1; h+=(c in pc)
    return h/pa if pa else 0

print("=== YOLO eval  weights=%s  imgsz=%d  aug=%s ==="%(WEIGHTS,IMGSZ,AUG))
for it in (0.1,0.3,0.5):
    print("instance recall @IoU%.1f: "%it+"  ".join("c%.3f=%.3f"%(ct,rec(ct,it)[0]) for ct in (0.001,0.01,0.05,0.1,0.25)))
print("precision @IoU0.3:       "+"  ".join("c%.3f=%.3f"%(ct,prec(ct,0.3)) for ct in (0.001,0.05,0.1,0.25)))
print("image PRESENCE recall:   "+"  ".join("c%.3f=%.3f"%(ct,presence(ct)) for ct in (0.001,0.05,0.1,0.25)))
print("--- per-class recall @conf0.05 IoU0.3 ---")
_,pg,pm=rec(0.05,0.3)
for i,n in enumerate(NAMES): print("  %-14s gt=%4d recall=%.3f"%(n,pg[i],pm[i]/pg[i] if pg[i] else 0))
print("EVAL_DONE")
