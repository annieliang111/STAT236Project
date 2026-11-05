"use strict";

(function() {
  const ratioButtons = Array.from(document.querySelectorAll('[data-ratio]'));
  const slider = document.getElementById('visibility-range');
  const visibilityOptions = [25, 50, 60, 75, 100];
  let ratio = '1';
  let visibility = 100;
  const svgNS = 'http://www.w3.org/2000/svg';

  function svgEl(name, attrs = {}, content = null) {
    const el = document.createElementNS(svgNS, name);
    for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, String(value));
    if (content != null) el.textContent = content;
    return el;
  }
  function fmt(value) { return Number(value).toFixed(1) + '%'; }
  function setText(id, text) { document.getElementById(id).textContent = text; }
  function yCoord(value) { return 318 - (value - 25) / 27 * 283; }
  function xCoord(q) { return 70 + (q - 25) / 75 * 720; }

  function drawResultsChart() {
    const svg = document.getElementById('results-chart');
    svg.replaceChildren();
    const series = SBM_RESULTS[ratio];
    const yTicks = [25,30,35,40,45,50];
    yTicks.forEach(y => {
      const yy = yCoord(y);
      svg.appendChild(svgEl('line', {x1:70,y1:yy,x2:790,y2:yy,stroke:'#e5efec','stroke-width':1}));
      svg.appendChild(svgEl('text', {x:56,y:yy+4,'font-size':12,'font-family':'Arial, sans-serif','text-anchor':'end',fill:'#8d9b9f'},String(y)));
    });
    visibilityOptions.forEach(q => {
      const xx = xCoord(q);
      if (q === visibility) {
        svg.appendChild(svgEl('rect',{x:xx-23,y:23,width:46,height:303,rx:4,fill:'#d8efeb',opacity:.44}));
        svg.appendChild(svgEl('line',{x1:xx,y1:23,x2:xx,y2:321,stroke:'#8cc6bd','stroke-dasharray':'4 5','stroke-width':1.4}));
      }
      svg.appendChild(svgEl('text',{x:xx,y:354,'text-anchor':'middle','font-size':13,'font-weight':q===visibility?'800':'500','font-family':'Arial, sans-serif',fill:q===visibility?'#176e6a':'#75868e'},q+'%'));
    });
    const names = [{key:'heuristic',color:'#d98758'},{key:'fitted',color:'#138f87'}];
    names.forEach(item => {
      const points = visibilityOptions.map(q => [xCoord(q),yCoord(series[q][item.key])]);
      const d = points.map(([x,y],i) => (i?'L':'M')+x.toFixed(2)+','+y.toFixed(2)).join(' ');
      svg.appendChild(svgEl('path',{d,stroke:item.color,'stroke-width':3.7,fill:'none','stroke-linecap':'round','stroke-linejoin':'round'}));
      points.forEach(([x,y],i) => {
        const selected = visibilityOptions[i] === visibility;
        if (selected) svg.appendChild(svgEl('circle',{cx:x,cy:y,r:11,fill:item.color,opacity:.13}));
        svg.appendChild(svgEl('circle',{cx:x,cy:y,r:selected?6.5:4.2,fill:selected?'#fff':item.color,stroke:item.color,'stroke-width':selected?3.5:1}));
      });
    });
    const f = series[visibility].fitted, h = series[visibility].heuristic;
    const diff = f-h;
    const fBetter = diff < -0.05;
    const hBetter = diff > 0.05;
    setText('visibility-label', visibility + '%');
    setText('scenario-q', visibility + '%');
    setText('scenario-r', ratio + '×');
    setText('value-fitted', fmt(f));
    setText('value-heuristic', fmt(h));
    const d = document.getElementById('outcome-diff');
    d.textContent = Math.abs(diff).toFixed(1) + ' pp ' + (fBetter?'lower':hBetter?'higher':'difference');
    d.classList.toggle('reverse',hBetter);
    setText('outcome-desc', fBetter ? 'for the fitted-risk optimizer versus the adaptive heuristic in this simulated setting.' : hBetter ? 'for the fitted-risk optimizer; the heuristic has the lower point estimate here.' : 'between the fitted-risk optimizer and the adaptive heuristic in this setting.');
    document.getElementById('bar-fitted').style.width = Math.min(f/52*100,100).toFixed(1) + '%';
    document.getElementById('bar-heuristic').style.width = Math.min(h/52*100,100).toFixed(1) + '%';
    svg.setAttribute('aria-label',`For community-strength ratio ${ratio}, plotting mean percentages infected by Day 50 over five observation levels. At ${visibility} percent of edges observed, fitted optimizer: ${fmt(f)}, adaptive heuristic: ${fmt(h)}. Each point averages 30 paired simulated epidemics.`);
  }

  const graphNodes = [];
  const graphEdges = [];
  function hash01(a,b,seed=1) {
    const n = Math.sin((a+1)*127.1+(b+1)*311.7+seed*77.3)*43758.5453;
    return n-Math.floor(n);
  }
  const centers = [{x:144,y:146},{x:397,y:148},{x:278,y:344}];
  for(let g=0;g<3;g++){
    for(let j=0;j<13;j++){
      const angle=j*Math.PI*2/13 + g*.36;
      const radius=49+hash01(g,j,5)*49;
      graphNodes.push({x:centers[g].x+Math.cos(angle)*radius,y:centers[g].y+Math.sin(angle)*radius,g,j});
    }
  }
  for(let i=0;i<graphNodes.length;i++) {
    for(let j=i+1;j<graphNodes.length;j++) {
      const within=graphNodes[i].g===graphNodes[j].g;
      const p=within?.35:.025;
      if (hash01(i,j,3)<p) graphEdges.push({i,j,h:hash01(i,j,15)});
    }
  }
  const diagnosed = new Set([6,18,31]);
  const selected = new Set([3,13,24,34]);
  function drawNetwork() {
    const links=document.getElementById('network-links');
    const nodes=document.getElementById('network-nodes');
    const halos=document.getElementById('network-halos');
    links.replaceChildren();nodes.replaceChildren();halos.replaceChildren();
    graphEdges.forEach(edge => {
      if(edge.h*100>visibility) return;
      const a=graphNodes[edge.i],b=graphNodes[edge.j];
      links.appendChild(svgEl('line',{x1:a.x,y1:a.y,x2:b.x,y2:b.y,stroke:a.g===b.g?'#6dabb1':'#b2c4bd','stroke-width':a.g===b.g?1.35:1.05,opacity:a.g===b.g?.33:.5}));
    });
    graphNodes.forEach((node,i) => {
      if(selected.has(i)) halos.appendChild(svgEl('circle',{cx:node.x,cy:node.y,r:15,stroke:'#b8f3dd','stroke-width':2.1,fill:'none',opacity:.75}));
      const color=diagnosed.has(i)?'#efa976':selected.has(i)?'#a2f4df':node.g===0?'#699bb5':node.g===1?'#8ca7bf':'#83b7b2';
      nodes.appendChild(svgEl('circle',{cx:node.x,cy:node.y,r:diagnosed.has(i)?6.8:selected.has(i)?7.8:4.6,fill:color,stroke:'#192c3c','stroke-width':1.6}));
    });
    centers.forEach((c,i)=>{
      nodes.appendChild(svgEl('text',{x:c.x,y:c.y-116,'text-anchor':'middle','font-size':10,'letter-spacing':1.5,'font-family':'Arial, sans-serif','font-weight':700,fill:'#839eac'},'COMMUNITY 0'+(i+1)));
    });
  }

  function update() { drawResultsChart(); drawNetwork(); }
  ratioButtons.forEach(button=>button.addEventListener('click',()=>{
    ratio=button.dataset.ratio;
    ratioButtons.forEach(b=>{
      const active=b===button;
      b.classList.toggle('selected',active);
      b.setAttribute('aria-pressed',String(active));
    });
    update();
  }));
  slider.addEventListener('input',()=>{visibility=visibilityOptions[Number(slider.value)];update();});
  update();
})();
