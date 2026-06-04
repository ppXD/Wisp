#!/usr/bin/env python3
"""Wisp brand kit generator — natural audio waveform, pixel/LED, sci-fi glow.

Run:  python3 branding/generate.py   → rewrites every SVG in this folder.
Then regenerate platform icons (see README) from wisp-icon-1024.png.
"""
import os

BR = "/Users/mars/Projects/Wisp/branding"

CLAY="#c96442"; CREAM="#f7f4ee"; INK="#1a1915"; SAGE="#5f8c6a"
CLAY_D="#b5573a"; EMBER="#e89766"; DEEP="#9c4631"
# App-icon tile: a warm espresso (dark) — the cool waveform glows against it.
ICON_TOP="#322823"; ICON_BOT="#221b16"
# Iridescent waveform — a cool-only spectrum swept left→right across the bars (a colourful
# equaliser). Cool hues stay clear of the warm bg so nothing blends; ends on violet, never
# warm pink. Edit/extend to retune the "AI" gradient.
ICON_PIX=["#4fe3a6","#3fd2dc","#4f9cff","#8a6bff","#a45cff"]
HEAD='<?xml version="1.0" encoding="UTF-8"?>\n'

# waveform shape (mirrored about the centre axis — audio-editor style)
HA=[1,2,4,3,5,2,4,5,3,4,2,3,1]
S=4.2; GX=2.0; GY=1.3; PLAYHEAD=8

def _f(x): return f"{x:.2f}"
def px(cx,cy,s,col,r=1.1):
    return f'<rect x="{_f(cx-s/2)}" y="{_f(cy-s/2)}" width="{_f(s)}" height="{_f(s)}" rx="{_f(r)}" fill="{col}"/>'

def rg_def():
    return (f'<radialGradient id="rg" gradientUnits="userSpaceOnUse" cx="50" cy="50" r="40">'
            f'<stop offset="0" stop-color="{EMBER}"/><stop offset="0.55" stop-color="{CLAY}"/>'
            f'<stop offset="1" stop-color="{CLAY_D}"/></radialGradient>')
def glow_def():
    return ('<filter id="glow" x="-30%" y="-30%" width="160%" height="160%">'
            '<feGaussianBlur stdDeviation="1.1" result="b"/>'
            '<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>')

def waveform(fill, live, glow=True):
    n=len(HA); pitchx=S+GX; pitchy=S+GY
    x0=50-(n*S+(n-1)*GX)/2 + S/2
    seg=[]
    for i,h in enumerate(HA):
        cx=x0+i*pitchx
        for k in range(h):
            seg.append(px(cx,50-(k+0.5)*pitchy,S,fill))
            seg.append(px(cx,50+(k+0.5)*pitchy,S,fill))
    bars="".join(seg)
    phx=x0+PLAYHEAD*pitchx
    ph=(f'<rect x="{_f(phx-1.2)}" y="20" width="2.4" height="60" rx="1.2" fill="{live}"/>'
        f'<rect x="{_f(phx-2.4)}" y="15.4" width="4.8" height="4.8" rx="1.3" fill="{live}"/>')
    if glow:
        return f'<g filter="url(#glow)">{bars}</g><g filter="url(#glow)">{ph}</g>'
    return bars+ph

def svg(defs, inner, vb="0 0 100 100", extra=""):
    d=f'<defs>{defs}</defs>' if defs else ''
    return HEAD+f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{vb}"{extra}>{d}{inner}</svg>\n'

files={}
files["wisp-mark.svg"]         = svg(rg_def()+glow_def(), waveform("url(#rg)", SAGE))
files["wisp-mark-reverse.svg"] = svg(glow_def(), waveform(CREAM, SAGE))
files["wisp-mark-mono.svg"]    = svg("", waveform("currentColor","currentColor", glow=False),
                                     extra=' fill="currentColor" style="color:#c96442"')

# app icon master @1024 — vivid orange squircle + iridescent waveform + sage playhead
def icon(size=1024):
    m=size*0.094; tile=size-2*m; r=tile*0.235; sc=7.6
    tg=(f'<linearGradient id="tg" gradientUnits="userSpaceOnUse" x1="{_f(size/2)}" y1="{_f(m)}" '
        f'x2="{_f(size/2)}" y2="{_f(size-m)}"><stop offset="0" stop-color="{ICON_TOP}"/>'
        f'<stop offset="1" stop-color="{ICON_BOT}"/></linearGradient>')
    stops="".join(f'<stop offset="{i/(len(ICON_PIX)-1):.3f}" stop-color="{c}"/>' for i,c in enumerate(ICON_PIX))
    rg=(f'<linearGradient id="rg" gradientUnits="userSpaceOnUse" x1="14" y1="50" x2="86" y2="50">{stops}</linearGradient>')
    grp=(f'<g transform="translate({_f(size/2)},{_f(size/2)}) scale({sc}) translate(-50,-50)">'
         f'{waveform("url(#rg)", SAGE)}</g>')
    return svg(tg+rg+glow_def(),
               f'<rect x="{_f(m)}" y="{_f(m)}" width="{_f(tile)}" height="{_f(tile)}" rx="{_f(r)}" fill="url(#tg)"/>{grp}',
               vb=f"0 0 {size} {size}", extra=f' width="{size}" height="{size}"')
files["wisp-icon.svg"]=icon()

# wordmark (mark + "Wisp"); Geist with graceful fallback
def wordmark(textcol):
    inner=(f'<g transform="translate(2,2) scale(0.80)">{waveform("url(#rg)", SAGE)}</g>'
           f'<text x="104" y="60" font-family="Geist, &quot;Geist Variable&quot;, system-ui, -apple-system, '
           f'Helvetica, Arial, sans-serif" font-size="46" font-weight="600" letter-spacing="-1.5" '
           f'fill="{textcol}">Wisp</text>')
    return svg(rg_def()+glow_def(), inner, vb="0 0 320 92", extra=' width="320" height="92"')
files["wisp-wordmark.svg"]         = wordmark(INK)
files["wisp-wordmark-reverse.svg"] = wordmark(CREAM)

for name, content in files.items():
    open(os.path.join(BR, name), "w").write(content)
old=os.path.join(BR,"wisp-icon-light.svg")
if os.path.exists(old): os.remove(old)
print("wrote", len(files), "svgs:", ", ".join(sorted(files)))
