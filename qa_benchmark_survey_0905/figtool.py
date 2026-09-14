#!/usr/bin/env python3
"""PDF figure helper.
  python3 figtool.py info <pdf>                      # captions + image bboxes per page (1-based pages)
  python3 figtool.py page <pdf> <page> <out.png> [dpi]
  python3 figtool.py clip <pdf> <page> x0 y0 x1 y1 <out.png> [dpi]
  python3 figtool.py auto <pdf> <page> "<caption prefix e.g. Figure 2>" <out.png> [dpi]
  python3 figtool.py text <pdf> <page>               # dump text of one page
"""
import sys, re, fitz

def captions(page):
    out=[]
    for b in page.get_text("blocks"):
        x0,y0,x1,y1,txt=b[0],b[1],b[2],b[3],b[4].strip()
        if re.match(r'^(Figure|Fig\.|Table)\s*\d+', txt):
            out.append((txt.split("\n")[0][:90], fitz.Rect(x0,y0,x1,y1)))
    return out

def info(pdf):
    doc=fitz.open(pdf); print("pages:",len(doc))
    for i,page in enumerate(doc, start=1):
        caps=captions(page)
        imgs=page.get_image_info()
        if caps or imgs:
            print(f"--- page {i} size={page.rect.width:.0f}x{page.rect.height:.0f}")
            for t,r in caps: print(f"  CAP [{r.x0:.0f},{r.y0:.0f},{r.x1:.0f},{r.y1:.0f}] {t}")
            for im in imgs:
                b=im['bbox']; print(f"  IMG [{b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f}] {im.get('width')}x{im.get('height')}")

def render(pdf,pno,out,dpi=170,clip=None):
    doc=fitz.open(pdf); page=doc[pno-1]
    pix=page.get_pixmap(dpi=dpi, clip=clip, alpha=False)
    pix.save(out); print("saved",out,pix.width,"x",pix.height)

def auto(pdf,pno,prefix,out,dpi=170):
    doc=fitz.open(pdf); page=doc[pno-1]
    cap=None
    for t,r in captions(page):
        if t.startswith(prefix): cap=r; break
    if cap is None:
        print("caption not found; captions on page:",[t for t,_ in captions(page)]); sys.exit(1)
    W=page.rect.width
    fullwidth = (cap.x1-cap.x0) > 0.6*W
    # candidate graphics: images + vector drawings above caption
    rects=[]
    for im in page.get_image_info():
        rects.append(fitz.Rect(im['bbox']))
    for d in page.get_drawings():
        r=d.get('rect')
        if r and r.width>2 and r.height>2 and r.width<W: rects.append(fitz.Rect(r))
    cand=[r for r in rects if r.y1 <= cap.y0+6 and cap.y0 - r.y1 < 420 and (fullwidth or (r.x1 > cap.x0-20 and r.x0 < cap.x1+20))]
    if not cand:
        print("no graphics found above caption; falling back to column band"); 
        band=fitz.Rect(cap.x0-10, max(30,cap.y0-330), cap.x1+10, cap.y0-2)
    else:
        band=fitz.Rect(min(r.x0 for r in cand), min(r.y0 for r in cand), max(r.x1 for r in cand), max(r.y1 for r in cand))
        # exclude text blocks that sit fully above the figure (keep figure only): trim top to lowest text block bottom above figure top? keep simple
        band = fitz.Rect(band.x0-6, band.y0-6, band.x1+6, band.y1+4)
    band = band & page.rect
    print("clip:",[round(v) for v in band])
    render(pdf,pno,out,dpi,clip=band)

if __name__=="__main__":
    a=sys.argv[1:]
    if not a: print(__doc__); sys.exit(0)
    cmd=a[0]
    if cmd=="info": info(a[1])
    elif cmd=="page": render(a[1],int(a[2]),a[3],int(a[4]) if len(a)>4 else 150)
    elif cmd=="clip": render(a[1],int(a[2]),a[7],int(a[8]) if len(a)>8 else 170,clip=fitz.Rect(*map(float,a[3:7])))
    elif cmd=="auto": auto(a[1],int(a[2]),a[3],a[4],int(a[5]) if len(a)>5 else 170)
    elif cmd=="text": print(fitz.open(a[1])[int(a[2])-1].get_text())
    else: print(__doc__)
