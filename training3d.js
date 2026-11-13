// Decode the archived MuJoCo body and joint samples. A heat compares one course
// and one initial seed; changing strategy/generation never mixes starting states.
const ARCHIVE = 'training-replays/';
const STRATEGY_NAMES = {pace:'Pace', corner:'Cornering', stability:'Stability'};

function integer(value) { return Number.isSafeInteger(value) && value >= 0; }

async function fetchData(path, signal, binary=false) {
  const response = await fetch(path,{signal});
  if (!response.ok) throw new Error('Could not open the training recording.');
  return binary ? response.arrayBuffer() : response.json();
}

async function checkedBuffer(meta, signal) {
  if (!/^[a-z0-9-]+\.bin$/.test(meta.file) || !/^[0-9a-f]{64}$/i.test(meta.sha256))
    throw new Error('The recording index contains an invalid file reference.');
  const bytes = await fetchData(ARCHIVE + meta.file,signal,true);
  // Browsers omit Web Crypto on plain HTTP LAN hosts. The decoder still checks
  // the byte length, sample clock and joint poses when viewing a local server.
  if (!globalThis.crypto?.subtle) return bytes;
  const hash = await crypto.subtle.digest('SHA-256',bytes);
  const hex = [...new Uint8Array(hash)].map(n => n.toString(16).padStart(2,'0')).join('');
  if (hex !== meta.sha256.toLowerCase()) throw new Error('The recording checksum did not match. Reload the page to try again.');
  return bytes;
}

function checkIndex(raw) {
  if (raw.version !== 1 || !raw.track || !Array.isArray(raw.joints) || raw.joints.length!==14 ||
      !Array.isArray(raw.groups) || !raw.groups.length ||
      !Array.isArray(raw.baselines) || !(raw.frameHz > 0) || !(raw.poseScale > 0))
    throw new Error('The training recording index could not be read.');
  const ids = new Set();
  let count = 0;
  for (const group of raw.groups) {
    if (!Object.hasOwn(STRATEGY_NAMES,group.strategy) || !integer(group.generation) || !integer(group.seed) ||
        typeof group.id !== 'string' || ids.has(group.id) || !Array.isArray(group.attempts) || !group.attempts.length)
      throw new Error('A training heat is missing its starting conditions.');
    ids.add(group.id);
    const numbers = new Set();
    for (const a of group.attempts) {
      if (!integer(a.candidate) || numbers.has(a.candidate) || !integer(a.frames) || a.frames < 2 ||
          typeof a.finished !== 'boolean' || !Number.isFinite(a.time) || a.time <= 0)
        throw new Error('A candidate has incomplete race data.');
      numbers.add(a.candidate);
      count++;
    }
  }
  if (raw.totalAttempts !== count) throw new Error('The archive and its candidate count disagree.');
  // Keep the buttons and the drop-downs in the same order across generations.
  const strategies=Object.keys(STRATEGY_NAMES);
  raw.groups.sort((a,b)=>strategies.indexOf(a.strategy)-strategies.indexOf(b.strategy) ||
    a.generation-b.generation || a.seed-b.seed);
  return raw;
}

function decodeRun(bytes, offset, meta, archive) {
  const stride = 21;
  const size = meta.frames * stride * 2;
  if (!integer(meta.frames) || meta.frames < 2 || offset+size > bytes.byteLength ||
      !Number.isFinite(meta.time) || meta.time <= (meta.frames-2)/archive.frameHz)
    throw new Error('A training trajectory is incomplete or out of order.');
  const view = new DataView(bytes,offset,size);
  const frames = new Array(meta.frames);
  for (let i=0; i<meta.frames; i++) {
    const row = new Array(stride+1);
    row[0] = i===meta.frames-1 ? meta.time : i/archive.frameHz;
    for (let k=0; k<stride; k++) row[k+1] = view.getInt16((i*stride+k)*2,true) / archive.poseScale;
    const q = Math.hypot(...row.slice(4,8));
    if (q<.98 || q>1.02) throw new Error('A recorded body pose is invalid.');
    for (let k=4;k<8;k++) row[k]/=q;
    frames[i]=row;
  }
  return {time:meta.time, finished:meta.finished, reason:meta.reason,
          progress:meta.progress,gates:meta.gates,frames};
}

function options(select, values, preferred) {
  select.replaceChildren(...values.map(([v,label]) => new Option(label,String(v))));
  if (values.some(([v]) => String(v)===String(preferred))) select.value=String(preferred);
}

function selectControl(form,name) {
  const label=document.createElement('label'), select=document.createElement('select');
  const text=document.createElement('span');
  text.textContent=name;
  select.setAttribute('aria-label',name);
  label.append(text,select);
  form.append(label);
  return select;
}

function button(form, title) {
  const control=document.createElement('button');
  control.type='button';
  control.textContent=title;
  form.append(control);
  return control;
}

function candidateId(group, pos) {
  return group.id+'-duck-'+String(pos).padStart(2,'0');
}

// Inputs: head-level container and callbacks. Set the race picker to 'training'
// before open(). close() aborts outstanding requests; open() restores URL state.
export function createTrainingArchive(container,{onGroup,onError}) {
  const form=document.createElement('div');
  form.className='training-controls';
  const strategy=selectControl(form,'Search');
  const generation=selectControl(form,'Generation');
  const seed=selectControl(form,'Start seed');
  const heat=selectControl(form,'Drivers');
  for (const select of [strategy,generation,seed,heat]) select.disabled=true;
  const nav=document.createElement('div');
  nav.className='training-heat-nav';
  const previous=button(nav,'Previous start');
  const next=button(nav,'Next start');
  previous.disabled=true; next.disabled=true;
  const detail=document.createElement('p');
  detail.className='training-detail'; detail.setAttribute('role','status');
  container.replaceChildren(form,nav,detail);
  container.hidden=true;

  let index=null, group=null, active=false, sequence=0;
  let request=null, prepared=null;
  const cache=new Map();
  const groupsAtStrategy=() => index.groups.filter(g=>g.strategy===strategy.value);
  const groupsAtGeneration=() => groupsAtStrategy().filter(g=>g.generation===Number(generation.value));

  function setHeats(preferred) {
    const candidates=group.attempts.slice().sort((a,b)=>a.candidate-b.candidate);
    const pages=[];
    for(let i=0;i<candidates.length;i+=10){
      const first=candidates[i].candidate, last=candidates[Math.min(i+9,candidates.length-1)].candidate;
      pages.push([String(i/10+1),first+'–'+last]);
    }
    if (pages.length>1) pages.push(['all','All '+candidates.length]);
    options(heat,pages,preferred);
  }

  function choose(want={}) {
    for (const select of [strategy,generation,seed,heat]) select.disabled=false;
    const strategies=[...new Set(index.groups.map(g=>g.strategy))];
    options(strategy,strategies.map(s=>[s,STRATEGY_NAMES[s]]),want.strategy);
    const gens=[...new Set(groupsAtStrategy().map(g=>g.generation))].sort((a,b)=>a-b);
    options(generation,gens.map(g=>[g,String(g)]),want.generation);
    const groups=groupsAtGeneration().sort((a,b)=>a.seed-b.seed);
    options(seed,groups.map(g=>[g.seed,String(g.seed)]),want.seed);
    group=groups.find(g=>g.seed===Number(seed.value));
    setHeats(want.heat);
    const i=index.groups.findIndex(g=>g.id===group.id);
    previous.disabled=i<=0; next.disabled=i>=index.groups.length-1;
  }

  function address() {
    const url=new URL(location.href);
    url.searchParams.set('view','training');
    url.searchParams.delete('run');
    for(const [name,value] of [['strategy',strategy.value],['generation',generation.value],['seed',seed.value],['heat',heat.value]])
      url.searchParams.set(name,value);
    history.replaceState(null,'',url);
  }

  async function loadBuffer(meta,signal) {
    let buffer=cache.get(meta.file);
    if(buffer)return buffer;
    buffer=await checkedBuffer(meta,signal);
    if(signal.aborted)throw new DOMException('Stopped','AbortError');
    cache.set(meta.file,buffer);
    if(cache.size>4)cache.delete(cache.keys().next().value);
    return buffer;
  }

  function show() {
    if (!active || prepared?.group!==group.id)return;
    const records=group.attempts;
    let offset=0;
    const indexed=records.map(meta=>{
      const start=offset; offset+=meta.frames*42; return {meta,start};
    }).sort((a,b)=>a.meta.candidate-b.meta.candidate);
    if(offset!==prepared.bytes.byteLength)throw new Error('The training heat has an unexpected length.');
    const first=heat.value==='all'?0:(Number(heat.value)-1)*10;
    const batch=heat.value==='all' ? indexed : indexed.slice(first,first+10);
    const entries=batch.map(({meta,start})=>{
      let run=prepared.decoded.get(meta.candidate);
      if (!run){
        run=decodeRun(prepared.bytes,start,meta,index);
        prepared.decoded.set(meta.candidate,run);
      }
      return {id:candidateId(group,meta.candidate), candidate:meta.candidate,
              label:'Duck '+meta.candidate, run};
    });
    const finished=records.filter(r=>r.finished).sort((a,b)=>a.time-b.time);
    const chosen=entries.filter(e=>e.run.finished).sort((a,b)=>a.run.time-b.run.time)[0] || entries[0];
    const leader=finished[0];
    const head=STRATEGY_NAMES[group.strategy]+' · Generation '+group.generation+' · Seed '+group.seed;
    const visible=entries.length, total=records.length;
    const summary=[index.totalAttempts.toLocaleString()+(index.source==='reconstructed' ? ' reconstructed training laps' : ' training laps'),
                   visible+' of '+total+' candidates',finished.length+' finished'];
    if(leader)summary.push('Fastest of all '+total+': Duck '+leader.candidate+' · '+leader.time.toFixed(2)+' s');
    if(!prepared.baseline)summary.push('Centreline reference unavailable');
    detail.textContent=summary.join(' · ');
    const reference={track:index.track,joints:index.joints,seed:group.seed,
                     baseline:prepared.baseline || chosen.run, challenger:chosen.run};
    onGroup({track:index.track,seed:group.seed,reference,entries,total,visible,focus:chosen.id,
             title:head,group:{id:group.id,strategy:group.strategy,generation:group.generation,seed:group.seed},
             hasBaseline:Boolean(prepared.baseline),totalAttempts:index.totalAttempts,
             finished:finished.length,winner:leader?{candidate:leader.candidate,time:leader.time}:null});
  }

  function fail(err, token) {
    if (!active || token!==sequence || err?.name==='AbortError')return;
    detail.textContent=err?.message || 'Training recording unavailable.';
    onError?.(err);
  }

  async function loadSelected() {
    const selected=group;
    request?.abort();
    request=new AbortController();
    const signal=request.signal, token=++sequence;
    prepared=null;
    detail.textContent='Loading candidates…';
    try {
      const baselineMeta=index.baselines.find(b=>b.seed===selected.seed);
      const [bytes,baseline]=await Promise.all([
        loadBuffer(selected,signal),
        baselineMeta ? loadBuffer(baselineMeta,signal).then(data=>{
            if(data.byteLength!==baselineMeta.frames*42)throw new Error('Reference recording has an unexpected length.');
            return decodeRun(data,0,baselineMeta,index);
          }).catch(error=>{if(signal.aborted)throw error;return null;}) : null
      ]);
      if(!active || token!==sequence)return;
      prepared={group:selected.id,bytes,baseline,decoded:new Map()};
      show();
    }catch(e){fail(e,token);}
  }

  function change(want) {
    choose(want); address(); return loadSelected();
  }
  strategy.addEventListener('change',()=>change({strategy:strategy.value}));
  generation.addEventListener('change',()=>change({strategy:strategy.value,generation:generation.value}));
  seed.addEventListener('change',()=>change({strategy:strategy.value,generation:generation.value,seed:seed.value}));
  heat.addEventListener('change',()=>{
    address(); try{show();}catch(e){fail(e,sequence);}
  });
  function move(direction) {
    const at=index.groups.findIndex(g=>g.id===group.id);
    const target=index.groups[at+direction];
    if(target)change({strategy:target.strategy,generation:target.generation,seed:target.seed});
  }
  previous.addEventListener('click',()=>move(-1)); next.addEventListener('click',()=>move(1));

  return {
    async open() {
      request?.abort();
      request=new AbortController();
      const signal=request.signal, token=++sequence;
      active=true;container.hidden=false;detail.textContent='Opening training archive…';
      try {
        if(!index){
          const raw=await fetchData(ARCHIVE+'index.json',signal);
          if(!active || token!==sequence)return;
          index=checkIndex(raw);
        }
        const params=new URLSearchParams(location.search);
        choose(Object.fromEntries(params));address();
        await loadSelected();
      }catch(e){fail(e,token);}
    },
    close() {
      active=false;sequence++;request?.abort();prepared=null;container.hidden=true;
    }
  };
}
