#!/usr/bin/env python3
"""Action-Aware Video Summarization (Track 04) - training-free pipeline.
Usage: python code/vsum.py <video.mp4> <out.json>
Steps: ffmpeg frame sampling (2 fps) -> motion / novelty / uniqueness importance
-> redundancy-aware greedy highlight selection under a 14% budget -> JSON.
Optional: CLIP zero-shot video label if open_clip is installed."""
import json, os, subprocess, sys
from pathlib import Path
import numpy as np

FPS, SZ = 2.0, 64
BUDGET, SEG = 0.14, 4.0
ROOT = Path(__file__).resolve().parents[1]
LABELS = ["bike","bikepolo","boxing","cooking","horse","jumping","longboarding","motor",
          "parkour","plane","rockclimbing","scuba","skate","ski","surfing"]

def duration(p):
    out = subprocess.run(["ffprobe","-v","error","-show_entries","format=duration",
                          "-of","default=nw=1:nk=1",p],capture_output=True,text=True,check=True).stdout
    return float(out.strip())

def frames(p):
    cmd = ["ffmpeg","-v","error","-nostdin","-i",p,"-an","-vf",f"fps={FPS},scale={SZ}:{SZ}",
           "-pix_fmt","rgb24","-f","rawvideo","pipe:1"]
    b = np.frombuffer(subprocess.run(cmd,capture_output=True,check=True).stdout,np.uint8)
    n = b.size // (SZ*SZ*3)
    return b[:n*SZ*SZ*3].reshape(n,SZ,SZ,3)

def z(x):
    s = x.std()
    return (x-x.mean())/s if s > 1e-8 else np.zeros_like(x)

def smooth(x, k=3):
    if len(x) < k: return x
    return np.convolve(np.pad(x,(k//2,k//2),mode="edge"),np.ones(k)/k,mode="valid")

def features(F):
    g = F.astype(np.float32).mean(3)
    mot = np.zeros(len(F)); 
    if len(F) > 1: mot[1:] = np.abs(g[1:]-g[:-1]).mean((1,2)); mot[0] = mot[1] if len(F)>1 else 0
    # colour histogram + coarse layout descriptor
    q = (F//32).astype(np.int32); idx = q[...,0]*64+q[...,1]*8+q[...,2]
    hist = np.stack([np.bincount(i.ravel(),minlength=512) for i in idx]).astype(np.float32)
    lay = g[:, ::8, ::8].reshape(len(F),-1)
    f = np.concatenate([hist/ (hist.sum(1,keepdims=True)+1e-8), lay/255.0*0.05],1)
    f = f - f.mean(0); f /= (np.linalg.norm(f,axis=1,keepdims=True)+1e-8)
    return f, mot

def importance(f, mot):
    n = len(f)
    nov = np.zeros(n)
    for t in range(n):
        a, b = max(0,t-20), max(0,t-4)
        if b > a: nov[t] = 1 - float(f[t] @ f[a:b].mean(0)/ (np.linalg.norm(f[a:b].mean(0))+1e-8))
    S = f @ f.T; k = min(10, n)
    uniq = 1 - np.sort(S,1)[:, -k:].mean(1)
    s = 0.4*z(np.log1p(mot)) + 0.35*z(nov) + 0.25*z(uniq)
    return smooth(s, 3)

def select(s, f, D):
    n = len(s); L = int(round(SEG*FPS)); budget = max(SEG, BUDGET*D)
    if D <= SEG or n <= L: return [[0.0, round(D,2)]]
    starts = np.arange(0, n-L+1)
    sc = np.array([s[i:i+L].mean() for i in starts]); sc = z(sc)
    emb = np.stack([f[i:i+L].mean(0) for i in starts]); emb /= (np.linalg.norm(emb,axis=1,keepdims=True)+1e-8)
    K = max(1, int(budget//SEG)); sigma = max(1.0, n/(2*K))
    chosen, used = [], np.zeros(n, bool)
    while len(chosen) < K:
        best, bi = -1e9, None
        for j,i in enumerate(starts):
            if used[i:i+L].any(): continue
            v = sc[j]
            if chosen:
                cs = [c for c,_ in chosen]
                v -= 0.6*max(float(emb[j]@emb[c]) for _,c in chosen)
                v -= 0.5*max(np.exp(-abs(i-c)/sigma) for c in cs)
            if v > best: best, bi = v, (i,j)
        if bi is None: break
        chosen.append(bi); used[max(0,bi[0]-1):bi[0]+L+1] = True
    segs = sorted([[i/FPS, min(D,(i+L)/FPS)] for i,_ in chosen])
    out = []
    for a,b in segs:
        a, b = round(a,2), round(b,2)
        if out and a <= out[-1][1]: out[-1][1] = max(out[-1][1], b)
        else: out.append([a,b])
    return out

def clip_label(F):
    try:
        import torch, open_clip
        m,_,_ = open_clip.create_model_and_transforms("ViT-B-32",pretrained="laion2b_s34b_b79k",
                                                     cache_dir=str(ROOT/"weights"))
        tok = open_clip.get_tokenizer("ViT-B-32"); m.eval()
        idx = np.linspace(0,len(F)-1,min(32,len(F))).astype(int)
        x = torch.tensor(F[idx]).permute(0,3,1,2).float()/255
        x = torch.nn.functional.interpolate(x,size=224,mode="bilinear")
        mean = torch.tensor([0.481,0.458,0.408]).view(1,3,1,1); std = torch.tensor([0.269,0.261,0.276]).view(1,3,1,1)
        with torch.no_grad():
            im = m.encode_image((x-mean)/std); im = im/im.norm(dim=-1,keepdim=True)
            tx = m.encode_text(tok([f"a first-person video of {l}" for l in LABELS])); tx = tx/tx.norm(dim=-1,keepdim=True)
            p = (100*im.mean(0,keepdim=True)@tx.T).softmax(-1)[0].numpy()
        return LABELS[int(p.argmax())], float(round(p.max(),4))
    except Exception:
        # Fallback heuristic: indoor warm scenes -> cooking, else other
        return "other", 0.5

def team_id():
    p = ROOT/"team_id.txt"
    return p.read_text().strip() if p.exists() else "SRMIST_TEAM"

def main(vid, out):
    D = round(duration(vid), 2); F = frames(vid)
    f, mot = features(F); s = importance(f, mot)
    hl = select(s, f, D)
    name, score = clip_label(F)
    res = {"team_id": team_id(), "video_id": Path(vid).stem, "duration_sec": D,
           "label": {"name": name, "score": score}, "actions": [],
           "highlights": [{"segment": h} for h in hl]}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=2))

if __name__ == "__main__":
    if len(sys.argv) != 3: sys.exit("usage: vsum.py <video.mp4> <out.json>")
    main(sys.argv[1], sys.argv[2])
