#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.data.read_text())
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    fragment = r'''<div id="nsmppo-vis" class="nsmppo-root">
  <div class="nsmppo-heading">
    <h2>No-shared MuJoCo: momentum variants vs fixed-lr PPO</h2>
    <div class="nsmppo-subtitle">Mean curve with min-max seed band; both fixed-lr PPO baselines n=5, RAT variants n=2; 10M environment steps</div>
  </div>
  <div class="nsmppo-legend" aria-label="Method legend"></div>
  <div class="nsmppo-grid"></div>
</div>
<style>
  #nsmppo-vis { color: light-dark(#18202a,#eef2f6); font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; letter-spacing:0; width:100%; }
  #nsmppo-vis .nsmppo-heading { margin:0 0 12px 0; }
  #nsmppo-vis h2 { margin:0; font-size:18px; line-height:1.3; font-weight:700; letter-spacing:0; }
  #nsmppo-vis .nsmppo-subtitle { margin-top:4px; color:light-dark(#596575,#aeb8c4); font-size:12px; }
  #nsmppo-vis .nsmppo-legend { display:flex; flex-wrap:wrap; gap:7px 13px; margin:0 0 14px 0; font-size:11px; color:light-dark(#344050,#cbd3dc); }
  #nsmppo-vis .legend-item { display:flex; align-items:center; gap:5px; white-space:nowrap; }
  #nsmppo-vis .legend-line { width:20px; height:3px; display:inline-block; }
  #nsmppo-vis .nsmppo-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px 22px; }
  #nsmppo-vis .plot { min-width:0; border-top:1px solid light-dark(#d7dde5,#38414b); padding-top:8px; }
  #nsmppo-vis .plot-title { font-size:13px; font-weight:700; margin:0 0 3px 0; }
  #nsmppo-vis svg { width:100%; height:auto; display:block; overflow:visible; }
  #nsmppo-vis .axis { stroke:light-dark(#8995a4,#798592); stroke-width:0.8; }
  #nsmppo-vis .gridline { stroke:light-dark(#e4e8ed,#303842); stroke-width:0.7; }
  #nsmppo-vis .tick-label { fill:light-dark(#667383,#aab4bf); font-size:9px; }
  #nsmppo-vis .axis-label { fill:light-dark(#4d5968,#b8c1cb); font-size:9px; }
  #nsmppo-vis .band { opacity:0.11; }
  #nsmppo-vis .curve { fill:none; stroke-width:1.9; stroke-linejoin:round; stroke-linecap:round; }
  #nsmppo-vis .curve.ppo { stroke-width:2.6; }
  @media (max-width:620px) { #nsmppo-vis .nsmppo-grid { grid-template-columns:1fr; gap:16px; } }
</style>
<script type="application/json" id="nsmppo-data">__PAYLOAD__</script>
<script>
(() => {
  const root = document.getElementById('nsmppo-vis');
  const data = JSON.parse(document.getElementById('nsmppo-data').textContent);
  const envLabels = {ant:'Ant',halfcheetah:'HalfCheetah',hopper:'Hopper',humanoid:'Humanoid',humanoidstandup:'HumanoidStandup',swimmer:'Swimmer',walker2d:'Walker2d'};
  const styles = {
    'PPO fixed lr=3e-4': {color:'#6b7280',dash:'',label:'PPO fixed lr=3e-4'},
    'PPO fixed lr=2e-4': {color:'#c02655',dash:'',label:'PPO fixed lr=2e-4'},
    'dual255p1 m=0.5': {color:'#00876c',dash:'',label:'255+1, m=0.5'},
    'dual255p1 m=0.9': {color:'#00a98f',dash:'6 4',label:'255+1, m=0.9'},
    'curv256 m=0.5': {color:'#2563eb',dash:'',label:'curv256, m=0.5'},
    'curv256 m=0.9': {color:'#60a5fa',dash:'6 4',label:'curv256, m=0.9'},
    'full m=0.5': {color:'#d45b13',dash:'',label:'full, m=0.5'},
    'full m=0.9': {color:'#f59e0b',dash:'6 4',label:'full, m=0.9'}
  };
  const order = Object.keys(styles);
  const NS='http://www.w3.org/2000/svg';
  const make=(name,attrs={})=>{const el=document.createElementNS(NS,name); for(const [k,v] of Object.entries(attrs)) el.setAttribute(k,v); return el;};
  const legend=root.querySelector('.nsmppo-legend');
  order.forEach(name=>{
    const item=document.createElement('span'); item.className='legend-item';
    const swatch=document.createElement('span'); swatch.className='legend-line';
    swatch.style.background=styles[name].dash ? `repeating-linear-gradient(90deg,${styles[name].color} 0 6px,transparent 6px 10px)` : styles[name].color;
    item.append(swatch,document.createTextNode(styles[name].label)); legend.appendChild(item);
  });
  const interp=(curve,x)=>{
    if(x<curve[0][0] || x>curve[curve.length-1][0]) return null;
    if(x===curve[0][0]) return curve[0][1];
    if(x===curve[curve.length-1][0]) return curve[curve.length-1][1];
    let lo=0,hi=curve.length-1;
    while(hi-lo>1){const mid=(lo+hi)>>1; if(curve[mid][0]<=x) lo=mid; else hi=mid;}
    const [x0,y0]=curve[lo],[x1,y1]=curve[hi]; return y0+(y1-y0)*(x-x0)/(x1-x0);
  };
  const grid=root.querySelector('.nsmppo-grid');
  Object.entries(data.environments).forEach(([env,methods])=>{
    const section=document.createElement('section'); section.className='plot';
    const title=document.createElement('div'); title.className='plot-title'; title.textContent=envLabels[env]; section.appendChild(title);
    const W=520,H=292,m={l:50,r:14,t:8,b:31},pw=W-m.l-m.r,ph=H-m.t-m.b;
    const svg=make('svg',{viewBox:`0 0 ${W} ${H}`,role:'img','aria-label':`${envLabels[env]} reward curves`});
    const xs=Array.from({length:121},(_,i)=>i*10000000/120);
    const stats={}; let ymin=Infinity,ymax=-Infinity;
    order.forEach(name=>{
      const seeds=methods[name];
      stats[name]=xs.map(x=>{const vals=seeds.map(c=>interp(c,x)).filter(v=>v!==null); if(!vals.length) return null; const mean=vals.reduce((a,b)=>a+b,0)/vals.length; const lo=Math.min(...vals),hi=Math.max(...vals); ymin=Math.min(ymin,lo); ymax=Math.max(ymax,hi); return {x,mean,lo,hi};});
    });
    const pad=Math.max((ymax-ymin)*0.06,1); ymin-=pad; ymax+=pad;
    const X=x=>m.l+x/10000000*pw, Y=y=>m.t+(ymax-y)/(ymax-ymin)*ph;
    const ticks=5;
    for(let i=0;i<=ticks;i++){
      const y=ymin+(ymax-ymin)*i/ticks,py=Y(y);
      svg.appendChild(make('line',{x1:m.l,y1:py,x2:W-m.r,y2:py,class:'gridline'}));
      const t=make('text',{x:m.l-7,y:py+3,'text-anchor':'end',class:'tick-label'}); t.textContent=Math.abs(y)>=100000?`${(y/1000).toFixed(0)}k`:Math.abs(y)>=1000?`${(y/1000).toFixed(1)}k`:Math.abs(y)>=100?y.toFixed(0):y.toFixed(1); svg.appendChild(t);
    }
    [0,2,4,6,8,10].forEach(v=>{const px=X(v*1e6); svg.appendChild(make('line',{x1:px,y1:m.t,x2:px,y2:H-m.b,class:'gridline'})); const t=make('text',{x:px,y:H-m.b+15,'text-anchor':'middle',class:'tick-label'}); t.textContent=v; svg.appendChild(t);});
    svg.appendChild(make('line',{x1:m.l,y1:m.t,x2:m.l,y2:H-m.b,class:'axis'}));
    svg.appendChild(make('line',{x1:m.l,y1:H-m.b,x2:W-m.r,y2:H-m.b,class:'axis'}));
    order.forEach(name=>{
      const s=stats[name].filter(Boolean),sty=styles[name];
      const band=[...s.map(p=>`${X(p.x)},${Y(p.hi)}`),...s.slice().reverse().map(p=>`${X(p.x)},${Y(p.lo)}`)].join(' ');
      svg.appendChild(make('polygon',{points:band,fill:sty.color,class:'band'}));
      const points=s.map(p=>`${X(p.x)},${Y(p.mean)}`).join(' ');
      const line=make('polyline',{points,stroke:sty.color,class:`curve ${name.startsWith('PPO')?'ppo':''}`}); if(sty.dash) line.setAttribute('stroke-dasharray',sty.dash); svg.appendChild(line);
    });
    const xlab=make('text',{x:m.l+pw/2,y:H-3,'text-anchor':'middle',class:'axis-label'}); xlab.textContent='Environment steps (millions)'; svg.appendChild(xlab);
    const ylab=make('text',{x:11,y:m.t+ph/2,transform:`rotate(-90 11 ${m.t+ph/2})`,'text-anchor':'middle',class:'axis-label'}); ylab.textContent='Episode return'; svg.appendChild(ylab);
    section.appendChild(svg); grid.appendChild(section);
  });
})();
</script>'''.replace("__PAYLOAD__", payload)
    args.output.write_text(fragment)


if __name__ == "__main__":
    main()
