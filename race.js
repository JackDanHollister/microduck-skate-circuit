'use strict';

const element = id => document.getElementById(id);
const ui = Object.fromEntries([
  'workspace', 'message', 'track', 'speed-plot', 'ghost', 'line', 'seek', 'clock',
  'duration', 'rate', 'speed', 'play', 'play-label', 'upload', 'track-title',
  'track-detail', 'challenger-result', 'baseline-result', 'difference', 'chart-end',
  'screen', 'video', 'map-view', 'video-view', 'map-options'
].map(id => [id, element(id)]));
const terrain = document.createElement('canvas');
let race = null;
let raceDuration = 0;
let field = null;
let elapsed = 0;
let playing = false;
const cameraParam = new URLSearchParams(location.search).get('camera');
let preferredAngle = ({map:'map', film:'video'})[cameraParam] || 'scene';
let angle = 'map';
let scenePromise = null;
let spectator = null;
let sceneUnavailable = false;
let sceneTrack = null;
let sceneDrivers = [];
let sceneEligible = false;
let forFilm = false;
let movieClockReady = false;
let previousTick = null;
let project = null;
let trackLength = 0;
let speedMax = 1;
let sourceRequest = 0;
let animationFrame = null;

function lapClock(time, precision = 1) {
  const factor = 10 ** precision;
  const ticks = Math.round(time * factor);
  const minutes = Math.floor(ticks / (60 * factor));
  const seconds = ((ticks - minutes * 60 * factor) / factor).toFixed(precision);
  return minutes + ':' + seconds.padStart(3 + precision, '0');
}

function pointList(points) {
  return Array.isArray(points) && points.length >= 3 && points.length <= 20000 &&
    points.every(p => Array.isArray(p) && p.length >= 2 && p.slice(0, 2).every(Number.isFinite));
}

const poseJoints = ['left_hip_yaw','left_hip_roll','left_hip_pitch','left_knee','left_ankle',
  'neck_pitch','head_pitch','head_yaw','head_roll','right_hip_yaw','right_hip_roll','right_hip_pitch','right_knee','right_ankle'];

function readRun(raw, name, jointOrder=null) {
  if (!raw || !Array.isArray(raw.frames) || raw.frames.length < 2 || raw.frames.length > 100000 ||
      typeof raw.finished !== 'boolean' || !Number.isFinite(raw.time) || raw.time < 0) {
    throw new Error('Invalid ' + name + ' lap.');
  }
  const frames = raw.frames;
  for (let i = 0; i < frames.length; i++) {
    const frame = frames[i];
    if (!Array.isArray(frame) || frame.length < 8 || !frame.slice(0, 8).every(Number.isFinite) ||
        frame[0] < 0 || (i && frame[0] <= frames[i - 1][0])) {
      throw new Error('Invalid recorded position in the ' + name + ' lap.');
    }
  }
  if (Math.abs(frames[0][0])>1e-5 || Math.abs(frames.at(-1)[0]-raw.time)>1e-4) {
    throw new Error('Recorded frames do not match the '+name+' lap time.');
  }
  const speeds = frames.map((frame, i) => {
    const a = frames[Math.max(0, i - 1)];
    const b = frames[Math.min(i + 1, frames.length - 1)];
    return Math.hypot(b[1] - a[1], b[2] - a[2]) / (b[0] - a[0]);
  });
  const joints = jointOrder || poseJoints;
  const order = Array.isArray(joints) ? poseJoints.map(j => joints.indexOf(j)) : [];
  const articulated = order.length===14 && order.every(i=>i>=0) && frames.every(frame=>
    frame.length>=22 && Math.hypot(...frame.slice(4,8))>.5 &&
    order.every(i=>Number.isFinite(frame[i+8])));
  return {frames, speeds, order, articulated, time: raw.time, finished: raw.finished, reason: String(raw.reason || "incomplete")};
}

function readTrack(data) {
  if (!data || !data.track || !['centre', 'left', 'right'].every(side => pointList(data.track[side]))) {
    throw new Error('This replay needs a centreline and two track edges.');
  }
  if (data.track.left.length !== data.track.right.length || data.track.centre.length !== data.track.left.length) {
    throw new Error('The track edges must align with the centreline samples.');
  }
  return data.track;
}

function parseRace(data) {
  return {track: readTrack(data), seed: data.seed, baseline: readRun(data.baseline, 'ghost', data.joints), challenger: readRun(data.challenger, 'trained', data.joints)};
}

function frameInterval(frames, time) {
  let low = 0, high = frames.length - 1;
  while (low + 1 < high) {
    const middle = (low + high) >> 1;
    if (frames[middle][0] <= time) low = middle;
    else high = middle;
  }
  const a = frames[low], b = frames[high];
  const blend = Math.max(0, Math.min(1, (time - a[0]) / (b[0] - a[0])));
  return {a,b,low,high,blend};
}

function poseAt(run, time) {
  const {a,b,blend} = frameInterval(run.frames,time);
  const pose = new Array(21);
  for(let i=0;i<3;i++)pose[i] = a[i+1]+blend*(b[i+1]-a[i+1]);
  const sign = a[4]*b[4]+a[5]*b[5]+a[6]*b[6]+a[7]*b[7] < 0 ? -1 : 1;
  for(let i=3;i<7;i++)pose[i] = a[i+1]+blend*(sign*b[i+1]-a[i+1]);
  const norm=Math.hypot(pose[3],pose[4],pose[5],pose[6]);
  for(let i=3;i<7;i++)pose[i]/=norm;
  for(let j=0;j<14;j++){
    const i=run.order[j]+8;
    pose[j+7]=a[i]+blend*(b[i]-a[i]);
  }
  return pose;
}

function frameAt(run, time) {
  const {a,b,low,high,blend}=frameInterval(run.frames,time);
  const mix = (u, v) => u + blend * (v - u);
  const yaw = f => Math.atan2(2 * (f[4] * f[7] + f[5] * f[6]), 1 - 2 * (f[6] ** 2 + f[7] ** 2));
  const yawA = yaw(a), yawDelta = Math.atan2(Math.sin(yaw(b) - yawA), Math.cos(yaw(b) - yawA));
  return {index: low, x: mix(a[1], b[1]), y: mix(a[2], b[2]), yaw: yawA + blend * yawDelta, speed: mix(run.speeds[low], run.speeds[high])};
}

const sceneCanvas = element('scene');
const sceneState = element('scene-state');
const follow = element('follow-driver');
const noFocus = ':course';
const ghostId = ':centreline';
function updateFollow(keep=false){
  const current=keep ? follow.value : null;
  const visible=sceneDrivers.filter(e=>!e.hidden);
  const byTime=[...visible].sort((a,b)=>Number(b.run.finished)-Number(a.run.finished)||a.run.time-b.run.time);
  follow.replaceChildren(...byTime.map(entry=>new Option(entry.label,entry.id)));
  if(ui.ghost.checked && race.baseline)follow.add(new Option('Centreline ghost',ghostId));
  follow.add(new Option('Whole circuit',noFocus));
  follow.value=Array.from(follow.options).some(o=>o.value===current) ? current : byTime[0]?.id || (ui.ghost.checked ? ghostId : noFocus);
}
function setupPerspective(drivers, movie, poster){
  sceneDrivers=drivers;
  sceneEligible=drivers.every(e=>e.run.articulated) && (!race.baseline || race.baseline.articulated);
  updateFollow();
  ui.video.pause();
  ui.video.removeAttribute('src');
  ui.video.removeAttribute('poster');
  if(movie){
    ui.video.src=movie;
    if(poster)ui.video.poster=poster;
  }
  ui.video.load();
  forFilm=Boolean(movie);
  configureAngle(preferredAngle, false);
}
function sceneFallback(){
  sceneUnavailable=true;
  if(angle!=='scene')return;
  configureAngle(forFilm ? 'video':'map', false);
  ui.message.textContent=forFilm
    ? 'Showing the 3D race film because interactive 3D is unavailable on this browser.'
    : 'Interactive 3D is unavailable on this browser. Showing the map.';
}
function ensureScene(){
  if(scenePromise || sceneUnavailable)return;
  scenePromise=import('./race3d.js').then(module=>module.createRace3D(sceneCanvas,{onContextLost:sceneFallback}))
    .then(api=>{
      spectator=api;
      if(angle==='scene'){sceneState.hidden=true; drawCurrent();}
    }).catch(sceneFallback);
}
function drawScene(){
  if(!spectator || !race || !sceneEligible || sceneUnavailable)return;
  if(sceneTrack!==race.track){
    spectator.setTrack(race.track);
    sceneTrack=race.track;
  }
  const drivers=sceneDrivers.filter(e=>!e.hidden).map(e=>({
    id:e.id, colour:e.colour, bib:e.bib, pose:poseAt(e.run,elapsed)
  }));
  if(ui.ghost.checked && race.baseline){
    drivers.push({id:ghostId,colour:'#83d3c9',ghost:true,pose:poseAt(race.baseline,elapsed)});
  }
  spectator.draw(drivers,{focusId:follow.value,camera:follow.value===noFocus?'grandstand':'chase',time:elapsed});
}
function drawCurrent(){
  if(angle==='map')drawMap();
  else if(angle==='scene'){
    try{drawScene();}catch{sceneFallback();}
  }
}
follow.addEventListener('change',drawCurrent);

function setSpan(canvas, width, height) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.max(1, Math.round(width * dpr));
  canvas.height = Math.max(1, Math.round(height * dpr));
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

function poly(points, close = true) {
  const path = new Path2D();
  for (let i = 0; i < points.length; i++) {
    const [x, y] = project(points[i][0], points[i][1]);
    if (i) path.lineTo(x, y); else path.moveTo(x, y);
  }
  if (close) path.closePath();
  return path;
}

function drawTerrain() {
  const {width: w, height: h} = ui.screen.getBoundingClientRect();
  if (w < 1 || h < 1) return;
  const ctx = setSpan(terrain, w, h);
  setSpan(ui.track, w, h);
  const {left, right, centre} = race.track;
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const edge of [left, right]) for (const [x, y] of edge) {
    minX = Math.min(minX, x); maxX = Math.max(maxX, x);
    minY = Math.min(minY, y); maxY = Math.max(maxY, y);
  }
  const scale = Math.min(w * .86 / Math.max(.001, maxX-minX), h * .84 / Math.max(.001, maxY-minY));
  project = (x, y) => [(x - (minX+maxX)/2)*scale + w/2, ((minY+maxY)/2-y)*scale + h/2];
  const grass = ctx.createLinearGradient(0, 0, w, h);
  grass.addColorStop(0, '#1c3530'); grass.addColorStop(1, '#122522');
  ctx.fillStyle = grass; ctx.fillRect(0, 0, w, h);
  const grid = Math.max(15, .5*scale);
  ctx.strokeStyle = '#ffffff0a'; ctx.lineWidth = .7; ctx.beginPath();
  for (let x = w/2 % grid; x < w; x += grid) {ctx.moveTo(x, 0);ctx.lineTo(x, h);}
  for (let y = h/2 % grid; y < h; y += grid) {ctx.moveTo(0, y);ctx.lineTo(w, y);}
  ctx.stroke();
  const edgeA = poly(left), edgeB = poly(right);
  for (const edge of [edgeA, edgeB]) {ctx.strokeStyle='#477367';ctx.lineWidth=8;ctx.stroke(edge);}
  const lane = new Path2D();
  lane.addPath(edgeA); lane.addPath(edgeB);
  ctx.fillStyle = '#323b3b'; ctx.fill(lane,'evenodd');
  for (const edge of [edgeA,edgeB]) {
    ctx.strokeStyle='#c0c7b8';ctx.lineWidth=1.2;ctx.stroke(edge);
  }
  ctx.setLineDash([3,8]);ctx.strokeStyle='#aab2ac58';ctx.lineWidth=1;
  ctx.stroke(poly(centre));ctx.setLineDash([]);

  // Mark the grid across the road at the recorded start, even on imported tracks.
  const start = race.challenger.frames[0];
  let nearest = 0, distance = Infinity;
  for (let i=0;i<centre.length;i++) {
    const ds=(centre[i][0]-start[1])**2+(centre[i][1]-start[2])**2;
    if (ds<distance) {distance=ds;nearest=i;}
  }
  const a=project(...left[nearest]), b=project(...right[nearest]);
  const angle=Math.atan2(b[1]-a[1], b[0]-a[0]);
  const span=Math.hypot(b[1]-a[1],b[0]-a[0]);
  const steps=Math.max(4,Math.round(span/3));
  const square=span/steps;
  ctx.save();ctx.translate(a[0],a[1]);ctx.rotate(angle);
  for(let col=0;col<steps;col++) for(let row=0;row<2;row++) {
    ctx.fillStyle=(row+col)%2?'#e8e8da':'#2b3331';
    ctx.fillRect(col*square,(row-1)*square,square+.1,square+.1);
  }
  ctx.restore();
  ctx.fillStyle='#c9dcd0';ctx.font='600 10px Barlow, sans-serif';
  const labelX = Math.max(6, Math.min(w-52,a[0]+(a[0]-b[0])*.24));
  const labelY = Math.max(12, Math.min(h-7,a[1]+(a[1]-b[1])*.25));
  ctx.fillText('START / FINISH',labelX,labelY);
}

function speedColour(speed) {
  const ratio=Math.max(0,Math.min(1,speed/speedMax));
  return 'rgb('+Math.round(225+30*ratio)+','+Math.round(84+125*ratio)+','+Math.round(69+35*ratio)+')';
}

function trail(ctx,run,current,ghost) {
  const frames=run.frames;
  ctx.lineWidth=ghost?1.45:2.4;ctx.lineCap='round';ctx.lineJoin='round';
  if (ghost) {
    ctx.setLineDash([4,5]);ctx.strokeStyle='rgba(150,224,214,.72)';ctx.beginPath();
    const start=project(frames[0][1],frames[0][2]);ctx.moveTo(...start);
    for(let i=1;i<=current.index;i++)ctx.lineTo(...project(frames[i][1],frames[i][2]));
    ctx.lineTo(...project(current.x,current.y));ctx.stroke();ctx.setLineDash([]);return;
  }
  // Decimate long traces to screen resolution while keeping the current position exact.
  const stride=Math.max(1,Math.floor(frames.length/420));
  let previous=project(frames[0][1],frames[0][2]);
  for(let i=stride;i<=current.index;i+=stride) {
    const next=project(frames[i][1],frames[i][2]);
    ctx.strokeStyle=speedColour(run.speeds[i]);
    ctx.beginPath();ctx.moveTo(...previous);ctx.lineTo(...next);ctx.stroke();previous=next;
  }
  ctx.beginPath();ctx.moveTo(...previous);ctx.lineTo(...project(current.x,current.y));ctx.strokeStyle=speedColour(current.speed);ctx.stroke();
}

function rounded(ctx,x,y,w,h,r,colour) {
  ctx.fillStyle=colour;ctx.beginPath();ctx.roundRect(x,y,w,h,r);ctx.fill();
}

function drawDuck(ctx,sample,ghost,colour=null) {
  const [x,y]=project(sample.x,sample.y);
  ctx.save();ctx.translate(x,y);ctx.rotate(-sample.yaw);
  const size=Math.min(1.2,Math.max(.72,ui.screen.clientWidth/700));
  ctx.scale(size,size);
  if(ghost){
    ctx.globalAlpha=.80;ctx.strokeStyle='#a6e4d8';ctx.lineWidth=1.5;ctx.setLineDash([3,2]);
    ctx.beginPath();ctx.ellipse(0,0,10.5,7.5,0,0,Math.PI*2);ctx.stroke();ctx.setLineDash([]);
    ctx.fillStyle='#8bd6c8';ctx.beginPath();ctx.moveTo(6,-2.6);ctx.lineTo(12,0);ctx.lineTo(6,2.6);ctx.closePath();ctx.fill();
  }else{
    // Top-down schematic: two roller feet, central chassis and the duck's camera head.
    ctx.shadowColor='#051410';ctx.shadowBlur=5;ctx.shadowOffsetY=2;
    rounded(ctx,-8,-6.6,12,3,1.2,'#161c1c');
    rounded(ctx,-8,3.6,12,3,1.2,'#161c1c');
    rounded(ctx,-6.5,-4.8,10,9.6,2.6,'#c8b49a');ctx.shadowBlur=0;ctx.shadowOffsetY=0;
    rounded(ctx,0,-5.7,8.4,11.4,2.3,colour || '#faf5df');
    rounded(ctx,6.4,-4.9,2.6,9.8,1,'#252e31');
    rounded(ctx,8,-2.2,3.7,4.4,1.2,'#ff9955');
    ctx.fillStyle='#a9edf0';ctx.fillRect(7.5,-3.6,1,1.5);ctx.fillRect(7.5,2.1,1,1.5);
    ctx.fillStyle=colour || '#ef8656';ctx.fillRect(-4.7,-6.1,5.5,1.25);ctx.fillRect(-4.7,4.85,5.5,1.25);
  }
  ctx.restore();
}

// Colours identify the same driver on the road, its trail and the results list.
// Labels and bib numbers also distinguish drivers with similar colours.
const fieldColours=['#ff986a','#88cfff','#f7d66b','#c6a6ff','#6bdfae','#ffa4ce','#e6e6e1','#a2bcff'];

function compareConditions(a,b) {
  if(!Number.isInteger(a.seed)||b.seed!==a.seed)return 'different or missing start seed';
  for(const side of ['centre','left','right']){
    const points=a.track[side], compared=b.track[side];
    if(points.length!==compared.length || points.some((p,i)=>
      p[0]!==compared[i][0] || p[1]!==compared[i][1]))return 'different circuit geometry';
  }
  return null;
}

function drawField(ctx) {
  const positions=[];
  if(ui.ghost.checked && race.baseline){
    const marker=frameAt(race.baseline,elapsed);
    if(ui.line.checked)trail(ctx,race.baseline,marker,true);
    drawDuck(ctx,marker,true);
  }
  for(const entry of field){
    if(entry.hidden)continue;
    const marker=frameAt(entry.run,elapsed);
    positions.push({entry,marker});
    if(!ui.line.checked)continue;
    const frames=entry.run.frames;
    // A short tail shows the line through the current corner, without covering
    // the entire road when several drivers share the same path.
    const tail=frameAt(entry.run,Math.min(elapsed,frames.at(-1)[0])-2.8);
    ctx.beginPath();
    ctx.moveTo(...project(tail.x,tail.y));
    const stride=Math.max(1,Math.floor((marker.index-tail.index)/120));
    for(let i=tail.index+1;i<=marker.index;i+=stride)ctx.lineTo(...project(frames[i][1],frames[i][2]));
    ctx.lineTo(...project(marker.x,marker.y));ctx.strokeStyle=entry.colour;
    ctx.globalAlpha=.70;ctx.lineWidth=2;ctx.lineJoin='round';ctx.stroke();ctx.globalAlpha=1;
  }
  for(const {entry,marker} of positions)drawDuck(ctx,marker,false,entry.colour);
  // Move only the numbered bibs when two measured positions overlap; keep the
  // ducks and trails at their original coordinates.
  const occupied=[];
  const width=ui.screen.clientWidth, height=ui.screen.clientHeight;
  ctx.font='bold 10px Barlow, sans-serif';ctx.textAlign='center';ctx.textBaseline='middle';
  for(const {entry,marker} of positions){
    const [x,y]=project(marker.x,marker.y);
    let point=null;
    for(const radius of [17,31,46]){
      for(const [dx,dy] of [[0,-1],[.9,-.5],[-.9,-.5],[1,.2],[-1,.2],[.6,.9],[-.6,.9],[0,1]]){
        const px=Math.max(9,Math.min(width-9,x+dx*radius)),py=Math.max(9,Math.min(height-9,y+dy*radius));
        if(occupied.every(p=>Math.hypot(p[0]-px,p[1]-py)>=16)){
          point=[px,py];break;
        }
      }
      if(point)break;
    }
    if(!point)continue;
    occupied.push(point);
    ctx.strokeStyle=entry.colour;ctx.lineWidth=1;ctx.beginPath();
    ctx.moveTo(x,y);ctx.lineTo(...point);ctx.stroke();ctx.fillStyle='#152321';
    ctx.beginPath();ctx.arc(...point,7.5,0,Math.PI*2);ctx.fill();ctx.stroke();
    ctx.fillStyle=entry.colour;ctx.fillText(entry.bib,...point);
  }
  ctx.textAlign='start';ctx.textBaseline='alphabetic';
}

function driverName(id,local) {
  if(local)return id;
  return id.replace(/^jackdh-/, '').split('-').map(word=>word.charAt(0).toUpperCase()+word.slice(1)).join(' ');
}

function resultsList(missing,training){
  const target=element('field-rows');
  target.replaceChildren();
  const ranked=[...field].sort((a,b)=>Number(b.run.finished)-Number(a.run.finished) ||
    (a.run.finished && b.run.finished ? a.run.time-b.run.time : 0) || a.id.localeCompare(b.id));
  const wins=ranked.filter(e=>e.run.finished && e.run.time===ranked[0].run.time);
  const fastest=wins.length>0?wins[0].run.time:0;
  const winnerLabel=training?'Fastest shown':(wins.length>1?'Joint winners on this start':'Winner on this start');
  element('field-winner').textContent=wins.length===0?'None of the drivers shown finished.':
    winnerLabel+' · '+wins.map(e=>e.label).join(', ')+' · '+lapClock(fastest,2);
  let previousTime=null,place=0;
  for(const [index,entry] of ranked.entries()){
    const row=document.createElement('li');
    row.dataset.runId=entry.id;
    row.style.setProperty('--driver-colour',entry.colour);
    if(wins.includes(entry))row.className='fastest-row';
    const select=document.createElement('label');
    const checkbox=document.createElement('input');
    checkbox.type='checkbox';checkbox.checked=true;checkbox.disabled=angle==='video';
    checkbox.setAttribute('aria-label','Show '+entry.label+' in the replay');
    checkbox.addEventListener('change',()=>{
      entry.hidden=!checkbox.checked;row.classList.toggle('muted-driver',entry.hidden);updateFollow(true);drawCurrent();
    });
    const bib=document.createElement('span');bib.className='field-bib';bib.textContent=entry.bib;
    const name=document.createElement('span');name.className='field-driver';name.textContent=entry.label;
    select.append(checkbox,bib,name);
    const time=document.createElement('span');time.className='field-time';
    if(entry.run.finished){
      if(previousTime!==entry.run.time)place=index+1;
      previousTime=entry.run.time;
      time.textContent=lapClock(entry.run.time,2);
      time.setAttribute('aria-label','Place '+place+' · '+entry.run.time.toFixed(2)+' seconds');
    }else time.textContent='DNF';
    const details=document.createElement('small');details.className='field-gap';
    details.textContent=entry.run.finished?(wins.includes(entry)?'Fastest lap':'+'+(entry.run.time-fastest).toFixed(2)+' s'):entry.run.reason+' · '+entry.run.time.toFixed(2)+' s';
    row.append(select,time,details);target.append(row);
  }
  const unavailable=element('field-unavailable');
  unavailable.hidden=missing.length===0;
  unavailable.textContent=missing.map(item=>item.id+' · '+item.error).join('; ');
}

function loadField(reference,entrants,missing,local,movie=null,hasBaseline=true,training=false) {
  pause();
  element('field-results').querySelector('h2').textContent='Recorded finish order';
  const all=entrants.map((entry,i)=>({...entry,
    label:entry.label || driverName(entry.id,local),
    bib:String(entry.candidate??i+1),
    colour:fieldColours[((entry.candidate??i+1)-1)%fieldColours.length],
    hidden:false}));
  const focus=all.find(entry=>entry.run.finished) || all[0];
  race={track:reference.track,seed:reference.seed,baseline:hasBaseline ? reference.baseline : null,challenger:focus.run};
  element('ghost-control').hidden=!hasBaseline;
  field=all;
  raceDuration=all.reduce((end,e)=>Math.max(end,e.run.time),race.baseline?.time || 0);
  elapsed=0;
  element('single-telemetry').hidden=true;
  element('field-results').hidden=false;
  element('trail-label').textContent='Racing lines';
  document.querySelector('.gradient-legend').hidden=true;
  ui['track-title'].textContent=local?'Local ghost race':'re:Invent 2018';
  ui['track-detail'].textContent=all.length+' drivers · same start, seed '+reference.seed;
  ui.track.setAttribute('aria-label','Overhead race with '+all.length+' numbered drivers sharing one track and clock');
  setupPerspective(all,movie,movie?'./replays/ghost-race.jpg':null);
  ui.workspace.hidden=false;ui.message.textContent='';
  ui.seek.max=raceDuration;ui.duration.textContent=lapClock(raceDuration);
  resultsList(missing,training);
  pause();resize();
}

function drawMap() {
  if(!project || angle!=='map')return;
  const ctx=ui.track.getContext('2d');
  ctx.drawImage(terrain,0,0,ui.screen.clientWidth,ui.screen.clientHeight);
  if (field) {
    drawField(ctx);
    return;
  }
  const duck=frameAt(race.challenger,elapsed);
  if(ui.ghost.checked && race.baseline){
    const ghost=frameAt(race.baseline,elapsed);
    trail(ctx,race.baseline,ghost,true);drawDuck(ctx,ghost,true);
  }
  if(ui.line.checked)trail(ctx,race.challenger,duck,false);
  drawDuck(ctx,duck,false);
}

let plot = null;
function buildPlot(){
  const canvas=ui['speed-plot'];
  const w=canvas.clientWidth,h=canvas.clientHeight;
  if(!w || !h)return;
  setSpan(canvas,w,h);
  plot=document.createElement('canvas');
  const ctx=setSpan(plot,w,h);
  const run=race.challenger;
  const y=s=>h-5-Math.max(0,Math.min(1,s/speedMax))*(h-10);
  ctx.lineWidth=1;ctx.strokeStyle='#d0d2c8';ctx.setLineDash([3,4]);ctx.beginPath();
  for (const yp of [5,h/2,h-5]){ctx.moveTo(0,yp);ctx.lineTo(w,yp);}ctx.stroke();ctx.setLineDash([]);
  ctx.beginPath();
  for(let i=0;i<run.frames.length;i++){
    const x=run.frames[i][0]/raceDuration*w;
    if(!i)ctx.moveTo(x,y(run.speeds[i]));else ctx.lineTo(x,y(run.speeds[i]));
  }
  ctx.strokeStyle='#bd502c';ctx.lineWidth=1.7;ctx.stroke();
}
function drawPlot(){
  if(!plot || field)return;
  const c=ui['speed-plot'],ctx=c.getContext('2d'),w=c.clientWidth,h=c.clientHeight;
  ctx.clearRect(0,0,w,h);ctx.drawImage(plot,0,0,w,h);
  const x=Math.min(w-1,Math.max(1,elapsed/raceDuration*w));ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,h);ctx.strokeStyle='#475b51';ctx.lineWidth=1.4;ctx.stroke();
}

function update(){
  if(!race)return;
  ui.seek.value=elapsed;
  ui.seek.setAttribute('aria-valuetext',lapClock(elapsed));
  ui.clock.textContent=lapClock(elapsed);
  const done=elapsed>race.challenger.frames.at(-1)[0]+.001;
  ui.speed.textContent=done?'—':frameAt(race.challenger,elapsed).speed.toFixed(2);
  drawCurrent();drawPlot();
}

function resize(){
  if(!race || ui.workspace.hidden)return;
  if(angle==='map')drawTerrain();
  if(!field)buildPlot();
  update();
}

function pause(){
  playing=false;previousTick=null;
  if(animationFrame!==null){cancelAnimationFrame(animationFrame);animationFrame=null;}
  ui.video.pause();ui.play.classList.remove('playing');
  const atEnd=race && elapsed>=raceDuration-.001;
  ui['play-label'].textContent=atEnd?'Replay':'Play';
  ui.play.setAttribute('aria-label',atEnd?'Replay from start':'Play replay');
}

function syncVideo(force=false){
  const v=ui.video;
  if(angle!=='video' || !v.src || v.readyState<1)return;
  const target=Math.min(elapsed,Number.isFinite(v.duration)?v.duration:elapsed);
  if(force || Math.abs(v.currentTime-target)>.08)v.currentTime=target;
  movieClockReady=true;
  v.playbackRate=Number(ui.rate.value);
}

function tick(timestamp){
  if(!playing)return;
  if(previousTick!==null){
    const delta=(timestamp-previousTick)/1000*Number(ui.rate.value);
    if(angle==='video' && !ui.video.ended && movieClockReady && ui.video.readyState>=1){
      elapsed=Math.min(raceDuration,ui.video.currentTime);
    }else{
      elapsed=Math.min(raceDuration,elapsed+delta);
    }
  }
  previousTick=timestamp;update();
  if(elapsed>=raceDuration-.001){elapsed=raceDuration;update();pause();return;}
  animationFrame=requestAnimationFrame(tick);
}

function play(){
  if(!race || playing)return;
  if(elapsed>=raceDuration-.001)elapsed=0;
  playing=true;previousTick=null;
  ui.play.classList.add('playing');ui['play-label'].textContent='Pause';ui.play.setAttribute('aria-label','Pause replay');
  if(angle==='video'){
    syncVideo();ui.video.play().catch(()=>{if(angle==='video')pause();});
  }
  update();animationFrame=requestAnimationFrame(tick);
}

function configureAngle(requested, remember=true){
  if(remember)preferredAngle=requested;
  let next=requested;
  if(requested==='scene' && (!sceneEligible || sceneUnavailable))next=forFilm?'video':'map';
  if(requested==='video' && !forFilm)next=sceneEligible && !sceneUnavailable?'scene':'map';
  const wasPlaying=playing;pause();
  angle=next;movieClockReady=false;
  sceneCanvas.hidden=next!=='scene';ui.track.hidden=next!=='map';ui.video.hidden=next!=='video';
  sceneState.hidden=next!=='scene' || Boolean(spectator);
  element('scene-view').hidden=!sceneEligible || sceneUnavailable;
  ui['video-view'].hidden=!forFilm;
  for(const [id,key] of [['scene-view','scene'],['map-view','map'],['video-view','video']]){
    element(id).setAttribute('aria-pressed',String(next===key));
  }
  ui['map-options'].hidden=next==='video';
  element('follow-label').hidden=next!=='scene';
  element('trail-control').hidden=next!=='map';
  document.querySelector('.gradient-legend').hidden=next!=='map' || Boolean(field);
  for(const cb of element('field-rows').querySelectorAll('input'))cb.disabled=next==='video';
  if(next==='scene')ensureScene();
  if(next==='video')syncVideo();
  resize();
  if(wasPlaying)play();
}
function setAngle(next){
  configureAngle(next);
  const url=new URL(location.href);
  url.searchParams.set('camera',next==='video'?'film':next);
  history.replaceState(null,'',url);
}

function resultText(run){
  return run.finished ? lapClock(run.time,2) : 'DNF';
}

function loadRace(raw,custom=false,archived=false){
  const validated=parseRace(raw);
  element('ghost-control').hidden=false;
  field=null;
  element('field-results').hidden=true;
  element('single-telemetry').hidden=false;
  element('trail-label').textContent='Speed trail';
  document.querySelector('.gradient-legend').hidden=false;
  ui.track.setAttribute('aria-label','Overhead map of the skating duck and the centreline ghost');
  pause();
  race=validated;
  raceDuration=Math.max(race.challenger.frames.at(-1)[0],race.baseline.frames.at(-1)[0],race.challenger.time,race.baseline.time);
  if(raceDuration<=0)throw new Error('Replay duration must be positive.');
  speedMax=race.challenger.speeds.reduce((max,x)=>Math.max(max,x),.1);
  elapsed=0;
  ui['track-title'].textContent=custom?(typeof raw.track.name==='string'?raw.track.name:'Imported circuit'):'re:Invent 2018';
  trackLength=race.track.centre.reduce((length,point,i,points)=>{
    const next=points[(i+1)%points.length];return length+Math.hypot(next[0]-point[0],next[1]-point[1]);
  },0);
  ui['track-detail'].textContent=(custom?'Local replay':archived?'Archived race':'AWS DeepRacer circuit')+' · '+trackLength.toFixed(1)+' m lap';
  ui.workspace.hidden=false;ui.message.textContent='';
  ui['challenger-result'].textContent=resultText(race.challenger);
  ui['baseline-result'].textContent=resultText(race.baseline);
  const gap=race.challenger.time-race.baseline.time;
  ui.difference.textContent=race.challenger.finished && race.baseline.finished
    ? (Math.abs(gap)<.005?'Same recorded lap time':Math.abs(gap).toFixed(2)+' s '+(gap<0?'faster':'slower')+' than the ghost')
    : 'DNF means the recorded run ended before a valid full lap.';
  ui.seek.max=raceDuration;ui.duration.textContent=lapClock(raceDuration);
  ui['chart-end'].textContent=race.challenger.finished?'Finish':'Run ended';
  const movie=!custom && !archived && raw.video==='replays/trackside.mp4';
  setupPerspective([{id:':challenger',run:race.challenger,label:'Trained duck',colour:'#ff986a'}],
                   movie?'./replays/trackside.mp4':null,movie?'./replays/trackside.jpg':null);
  pause();resize();
}

ui.play.addEventListener('click',()=>playing?pause():play());
ui.seek.addEventListener('input',()=>{
  const value=Number(ui.seek.value);
  elapsed=raceDuration-value <= Number(ui.seek.step)/2+1e-8 ? raceDuration : value;
  if(elapsed>=raceDuration)pause();
  else if(!playing)pause();
  syncVideo();update();
});
ui.rate.addEventListener('change',()=>{previousTick=null;syncVideo();});
ui.ghost.addEventListener('change',()=>{updateFollow(true);drawCurrent();});
element('scene-view').addEventListener('click',()=>setAngle('scene'));
ui.line.addEventListener('change',drawMap);
ui['map-view'].addEventListener('click',()=>setAngle('map'));
ui['video-view'].addEventListener('click',()=>setAngle('video'));
ui.video.addEventListener('loadedmetadata',()=>{syncVideo(true);});
ui.video.addEventListener('error',()=>{
  if(ui['video-view'].hidden || !ui.video.getAttribute('src'))return;
  forFilm=false;configureAngle(sceneEligible && !sceneUnavailable?'scene':'map',false);
  ui.message.textContent='The trackside film could not load. The overhead replay is available.';
});
ui.upload.addEventListener('change',async()=>{
  const files=Array.from(ui.upload.files);
  if(!files.length)return;
  const request=++sourceRequest;
  try{
    if(files.reduce((sum,f)=>sum+f.size,0)>50e6)throw new Error('Choose replay files totalling under 50 MB.');
    const raws=await Promise.all(files.map(async f=>JSON.parse(await f.text())));
    if(request!==sourceRequest)return;
    if(files.length===1 && Array.isArray(raws[0]?.field))loadSprintField(raws[0]);
    else if(files.length===1)loadRace(raws[0],true);
    else {
      const fieldRuns=raws.map(parseRace);
      const seen=new Set();
      const entrants=fieldRuns.map((r,i)=>{
        const mismatch=compareConditions(fieldRuns[0],r);
        if(mismatch)throw new Error(files[i].name+': '+mismatch+'.');
        const parent=files[i].webkitRelativePath?.split('/').at(-2);
        let label=(parent || files[i].name.replace(/\.json$/i,'')).trim() || 'Local run';
        if(seen.has(label))label+=' ('+(i+1)+')';
        seen.add(label);
        return {id:label,run:r.challenger};
      });
      loadField(fieldRuns[0],entrants,[],true);
    }
    leaveTraining();
    localOption.textContent=files.length>1?'Opened replays':'Opened file';
    localOption.hidden=false;
    picker.value=':local';
    activeRun=':local';
    element('run-parent').hidden=true;
    const url=new URL(location.href);
    url.searchParams.delete('run');
    clearTrainingParams(url);
    url.searchParams.delete('view');
    history.replaceState(null,'',url);
  }catch(error){if(request===sourceRequest)ui.message.textContent='Replay could not open. '+error.message;}
  ui.upload.value='';
});
new ResizeObserver(resize).observe(ui.screen);
new ResizeObserver(resize).observe(ui['speed-plot']);

const picker=element('run-picker');
const runs=new Map();
const localOption=new Option('Opened file',':local');
localOption.disabled=true;
localOption.hidden=true;
picker.add(localOption);
let activeRun='';

function setParent(run){
  const target=element('run-parent');
  target.replaceChildren();
  target.hidden=!run;
  if(!run)return;
  if(!run.parent){target.textContent='First recorded driver';return;}
  const link=document.createElement('a');
  const url=new URL(location.href);
  url.searchParams.set('run',run.parent);
  url.searchParams.delete('view');
  link.href=url.href;
  link.textContent=run.parent;
  target.append('Parent: ',link);
}

const replayPaths = new Map();
async function fetchReplay(id) {
  if (replayPaths.has(id))return replayPaths.get(id);
  const path=id?'./lineage/'+encodeURIComponent(id)+'/race.json':'./replays/race.json';
  const result=fetch(path).then(response=>{
    if(!response.ok)throw new Error('Replay unavailable');
    return response.json();
  });
  replayPaths.set(id,result);
  try{return await result;}
  catch(error){replayPaths.delete(id);throw error;}
}

let trainingArchive=null;
let trainingIntent=false;
const trainingBox=element('training-archive');
function clearTrainingParams(url){
  for(const key of ['strategy','generation','seed','heat','batch'])url.searchParams.delete(key);
}
function leaveTraining(){
  trainingIntent=false;
  trainingArchive?.close();
  trainingBox.hidden=true;
}
function showTrainingGroup(group){
  if(!trainingIntent)return;
  const reference=parseRace(group.reference);
  const entries=group.entries.map(e=>({id:String(e.id),label:e.label,candidate:e.candidate,
    run:readRun(e.run,e.label,group.reference.joints)}));
  loadField(reference,entries,[],true,null,group.hasBaseline!==false,true);
  ui['track-title'].textContent=group.title || 'Training attempts';
  ui['track-detail'].textContent=entries.length+' shown · same start, seed '+reference.seed;
  element('field-results').querySelector('h2').textContent='Finish order';
  picker.value=':training';
}
async function openTraining(request,replaceAddress){
  trainingIntent=true;
  pause();
  trainingBox.hidden=false;
  setParent(null);
  ui.workspace.hidden=false;
  ui.message.textContent='Loading the recorded training attempts…';
  if(replaceAddress){
    const url=new URL(location.href);
    url.searchParams.delete('run');
    url.searchParams.set('view','training');
    history.replaceState(null,'',url);
  }
  if(!trainingArchive){
    const mod=await import('./training3d.js');
    if(request!==sourceRequest)return;
    trainingArchive=mod.createTrainingArchive(trainingBox,{
      onGroup:showTrainingGroup,
      onError:err=>{
        if(!trainingIntent)return;
        ui.message.textContent='Training replay unavailable. '+(err?.message || err)+' Choose another start or switch to Saved drivers.';
        if(!race && defaultTrainingRequest===sourceRequest){
          const fallback=runs.size?':all':'';
          openRun(fallback).then(()=>{
            if(activeRun===fallback)ui.message.textContent='The training archive is unavailable. Showing the saved race.';
          });
        }
      }
    });
  }
  if(request!==sourceRequest)return;
  picker.value=':training';
  activeRun=':training';
  localOption.hidden=true;
  await trainingArchive.open();
}

async function openRun(id, replaceAddress=true){
  const request=++sourceRequest;
  try{
    if(id===':training'){
      await openTraining(request,replaceAddress);
      return;
    }
    leaveTraining();
    if(id===':sprint'){
      const response=await fetch('./replays/sprint-transfer/race.json');
      if(!response.ok)throw new Error('The sprint checkpoint race could not be loaded.');
      const data=await response.json();
      if(request!==sourceRequest)return;
      loadSprintField(data);
      setParent(null);
    }else if(id===':all'){
      const reference=parseRace(await fetchReplay(''));
      const snapshots=await Promise.all([...runs.keys()].map(async key=>{
        try{
          const archived=parseRace(await fetchReplay(key));
          const mismatch=compareConditions(reference,archived);
          if(mismatch)return {id:key,error:mismatch};
          return {id:key,run:archived.challenger};
        }catch {return {id:key,error:'replay unavailable'};}
      }));
      if(request!==sourceRequest)return;
      const entrants=snapshots.filter(item=>item.run);
      const missing=snapshots.filter(item=>item.error);
      if(!entrants.length)throw new Error('No saved laps on the featured track and start seed.');
      // A fork may add, remove or change its archived drivers. Only attach the
      // prerecorded film when these are still the exact five filmed runs.
      let movie=null;
      if(entrants.length===5 && !missing.length){
        try{
          const manifest=await fetch('./replays/ghost-film.json').then(r=>r.json());
          const match=await Promise.all(entrants.map(async entry=>{
            const bytes=new TextEncoder().encode(JSON.stringify(entry.run.frames));
            const hash=await crypto.subtle.digest('SHA-256',bytes);
            const hex=Array.from(new Uint8Array(hash),b=>b.toString(16).padStart(2,'0')).join('');
            return hex===manifest.poses[entry.id];
          }));
          if(match.every(Boolean))movie='./replays/ghost-race.mp4';
        }catch{ /* The film is optional in self-hosted forks. */ }
      }
      if(request!==sourceRequest)return;
      loadField(reference,entrants,missing,false,movie);
      setParent(null);
    }else{
      if(id && !runs.has(id))throw new Error('This run is not in the archive.');
      const raw=await fetchReplay(id);
      if(request!==sourceRequest)return;
      loadRace(raw,false,Boolean(id));
      setParent(runs.get(id));
    }
    picker.value=id;
    localOption.hidden=true;
    activeRun=id;
    if(replaceAddress){
      const url=new URL(location.href);
      clearTrainingParams(url);
      if(id && !id.startsWith(':'))url.searchParams.set('run',id);
      else url.searchParams.delete('run');
      if(id==='')url.searchParams.set('view','featured');
      else if(id===':all')url.searchParams.set('view','all');
      else if(id===':sprint')url.searchParams.set('view','sprint');
      else url.searchParams.delete('view');
      history.replaceState(null,'',url);
    }
  }catch(error){
    if(request===sourceRequest){
      picker.value=activeRun;
      ui.message.textContent=error.message+' Choose another recorded run or open a local replay.';
    }
  }
}
picker.addEventListener('change',()=>openRun(picker.value));
function loadSprintField(data){
  const track=readTrack(data);
  if(!Array.isArray(data.field) || !data.field.length ||
     data.field.some(e=>!e || typeof e.id!=='string' || !e.id || typeof e.label!=='string') ||
     new Set(data.field.map(e=>e.id)).size!==data.field.length)throw new Error('Invalid checkpoint field.');
  const entries=data.field.map(e=>({id:e.id,label:e.label,run:readRun(e.run,e.label,data.joints)}));
  loadField({track,seed:data.seed},entries,[],true,null,false);
  ui['track-title'].textContent='Sprint checkpoints on the circuit';
  ui['track-detail'].textContent='Duck 28 steering · same start, seed '+data.seed;
}
let defaultTrainingRequest=null;
async function startViewer(){
  const pending=sourceRequest;
  try{
    const response=await fetch('./lineage/index.json');
    if(response.ok){
      const index=await response.json();
      for(const run of index){
        if(typeof run.id!=='string' || !/^[a-z0-9][a-z0-9-]*$/.test(run.id))continue;
        runs.set(run.id,run);
        picker.add(new Option(run.id,run.id));
      }
    }
  }catch{ /* The featured replay works in copies without a run archive. */ }
  element('run-archive').hidden=false;
  picker.querySelector('[value=":all"]').hidden=runs.size===0;
  // Keep setup and the run tree attached to the current public GitHub Pages fork.
  const pagesOwner=location.hostname.match(/^([a-z0-9-]+)\.github\.io$/i);
  const pagesRepo=location.pathname.split('/').filter(Boolean)[0];
  if (pagesOwner && pagesRepo && /^[a-z0-9._-]+$/i.test(pagesRepo)){
    const repo='https://github.com/'+pagesOwner[1]+'/'+encodeURIComponent(pagesRepo);
    document.querySelector('.train-link').href=repo+'#try-your-own-duck';
    document.querySelector('.race-credits a').href=repo+'/blob/main/assets/duck-attribution.md';
  }
  if(sourceRequest!==pending)return;
  const params=new URLSearchParams(location.search);
  const view=params.get('view');
  const views={featured:'', all:runs.size?':all':'', sprint:':sprint'};
  const requested=params.get('run') || (Object.hasOwn(views,view)?views[view]:':training');
  if(!params.has('run') && !params.has('view'))defaultTrainingRequest=sourceRequest+1;
  if(requested && ![':all',':training',':sprint'].includes(requested) && !runs.has(requested)){
    await openRun('',false);
    ui.message.textContent='That run is unavailable here. Showing the featured race.';
  }else{
    await openRun(requested,false);
  }
}
startViewer();
