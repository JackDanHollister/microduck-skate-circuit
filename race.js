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
let elapsed = 0;
let playing = false;
let angle = 'map';
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

function readRun(raw, name) {
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
  const speeds = frames.map((frame, i) => {
    const a = frames[Math.max(0, i - 1)];
    const b = frames[Math.min(i + 1, frames.length - 1)];
    return Math.hypot(b[1] - a[1], b[2] - a[2]) / (b[0] - a[0]);
  });
  return {frames, speeds, time: raw.time, finished: raw.finished};
}

function parseRace(data) {
  if (!data || !data.track || !['centre', 'left', 'right'].every(side => pointList(data.track[side]))) {
    throw new Error('This replay needs a centreline and two track edges.');
  }
  if (data.track.left.length !== data.track.right.length || data.track.centre.length !== data.track.left.length) {
    throw new Error('The track edges must align with the centreline samples.');
  }
  return {track: data.track, baseline: readRun(data.baseline, 'ghost'), challenger: readRun(data.challenger, 'trained')};
}

function frameAt(run, time) {
  const frames = run.frames;
  let low = 0, high = frames.length - 1;
  while (low + 1 < high) {
    const middle = (low + high) >> 1;
    if (frames[middle][0] <= time) low = middle;
    else high = middle;
  }
  const a = frames[low], b = frames[high];
  const blend = Math.max(0, Math.min(1, (time - a[0]) / (b[0] - a[0])));
  const mix = (u, v) => u + blend * (v - u);
  const yaw = f => Math.atan2(2 * (f[4] * f[7] + f[5] * f[6]), 1 - 2 * (f[6] ** 2 + f[7] ** 2));
  const yawA = yaw(a), yawDelta = Math.atan2(Math.sin(yaw(b) - yawA), Math.cos(yaw(b) - yawA));
  return {index: low, x: mix(a[1], b[1]), y: mix(a[2], b[2]), yaw: yawA + blend * yawDelta, speed: mix(run.speeds[low], run.speeds[high])};
}

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

function drawDuck(ctx,sample,ghost) {
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
    rounded(ctx,0,-5.7,8.4,11.4,2.3,'#faf5df');
    rounded(ctx,6.4,-4.9,2.6,9.8,1,'#252e31');
    rounded(ctx,8,-2.2,3.7,4.4,1.2,'#ff9955');
    ctx.fillStyle='#a9edf0';ctx.fillRect(7.5,-3.6,1,1.5);ctx.fillRect(7.5,2.1,1,1.5);
    ctx.fillStyle='#ef8656';ctx.fillRect(-4.7,-6.1,5.5,1.25);ctx.fillRect(-4.7,4.85,5.5,1.25);
  }
  ctx.restore();
}

function drawMap() {
  if(!project || angle!=='map')return;
  const ctx=ui.track.getContext('2d');
  ctx.drawImage(terrain,0,0,ui.screen.clientWidth,ui.screen.clientHeight);
  const duck=frameAt(race.challenger,elapsed);
  if(ui.ghost.checked){
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
  if(!plot)return;
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
  drawMap();drawPlot();
}

function resize(){
  if(!race || ui.workspace.hidden)return;
  drawTerrain();buildPlot();update();
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

function setAngle(next){
  if(next===angle)return;
  const wasPlaying=playing;pause();
  angle=next;movieClockReady=false;
  ui.track.hidden=next!=='map';ui.video.hidden=next!=='video';
  ui['map-options'].hidden=next!=='map';
  ui['map-view'].setAttribute('aria-pressed',String(next==='map'));
  ui['video-view'].setAttribute('aria-pressed',String(next==='video'));
  if(next==='video'){
    syncVideo();
  }else resize();
  if(wasPlaying)play();
}

function resultText(run){
  return run.finished ? lapClock(run.time,2) : 'DNF';
}

function loadRace(raw,custom=false,archived=false){
  const validated=parseRace(raw);
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
  ui.video.pause();ui.video.removeAttribute('src');ui.video.load();
  // Only the checked-in film may be loaded. User-supplied replays are data only.
  const hasVideo=!custom && !archived && raw.video==='replays/trackside.mp4';
  ui['video-view'].hidden=!hasVideo;
  if(hasVideo)ui.video.src='./replays/trackside.mp4';
  if(angle!=='map')setAngle('map');
  pause();resize();
}

ui.play.addEventListener('click',()=>playing?pause():play());
ui.seek.addEventListener('input',()=>{
  elapsed=Number(ui.seek.value);
  if(elapsed>=raceDuration)pause();
  else if(!playing)pause();
  syncVideo();update();
});
ui.rate.addEventListener('change',()=>{previousTick=null;syncVideo();});
ui.ghost.addEventListener('change',drawMap);
ui.line.addEventListener('change',drawMap);
ui['map-view'].addEventListener('click',()=>setAngle('map'));
ui['video-view'].addEventListener('click',()=>setAngle('video'));
ui.video.addEventListener('loadedmetadata',()=>{syncVideo(true);});
ui.video.addEventListener('error',()=>{
  if(ui['video-view'].hidden || !ui.video.getAttribute('src'))return;
  ui['video-view'].hidden=true;setAngle('map');
  ui.message.textContent='The trackside film could not load. The overhead replay is available.';
});
ui.upload.addEventListener('change',async()=>{
  const file=ui.upload.files[0];if(!file)return;
  const request=++sourceRequest;
  try{
    if(file.size>50e6)throw new Error('Please use a replay smaller than 50 MB.');
    const raw=JSON.parse(await file.text());
    if(request===sourceRequest){
      loadRace(raw,true);
      picker.value=':local';
      activeRun=':local';
      const option=picker.querySelector('[value=":local"]');
      option.hidden=false;
      element('run-parent').hidden=true;
      const url=new URL(location.href);
      url.searchParams.delete('run');
      history.replaceState(null,'',url);
    }
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
  link.href=url.href;
  link.textContent=run.parent;
  target.append('Parent: ',link);
}

async function openRun(id, replaceAddress=true){
  const request=++sourceRequest;
  try{
    if(id && !runs.has(id))throw new Error('This run is not in the archive.');
    const archived=Boolean(id);
    const path=archived ? './lineage/'+encodeURIComponent(id)+'/race.json' : './replays/race.json';
    const response=await fetch(path);
    if(!response.ok)throw new Error('Could not fetch the replay.');
    const raw=await response.json();
    if(request!==sourceRequest)return;
    loadRace(raw,false,archived);
    picker.value=id;
    localOption.hidden=true;
    activeRun=id;
    setParent(runs.get(id));
    if(replaceAddress){
      const url=new URL(location.href);
      if(id)url.searchParams.set('run',id);
      else url.searchParams.delete('run');
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
  element('run-archive').hidden=runs.size===0;
  // Keep setup and the run tree attached to the current public GitHub Pages fork.
  const pagesOwner=location.hostname.match(/^([a-z0-9-]+)\.github\.io$/i);
  const pagesRepo=location.pathname.split('/').filter(Boolean)[0];
  if (pagesOwner && pagesRepo && /^[a-z0-9._-]+$/i.test(pagesRepo)){
    const repo='https://github.com/'+pagesOwner[1]+'/'+encodeURIComponent(pagesRepo);
    document.querySelector('.train-link').href=repo+'#try-your-own-duck';
  }
  if(sourceRequest!==pending)return;
  const requested=new URLSearchParams(location.search).get('run') || '';
  if(requested && !runs.has(requested)){
    await openRun('',false);
    ui.message.textContent='That run is unavailable here. Showing the featured race.';
  }else{
    await openRun(requested,false);
  }
}
startViewer();
