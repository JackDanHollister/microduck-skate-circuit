/** Z-up spectator. Every skater's root and 14 joint angles come from its recorded lap. */
import * as THREE from './vendor/three/three.module.min.js';

const jointNames = [
  'left_hip_yaw', 'left_hip_roll', 'left_hip_pitch', 'left_knee', 'left_ankle',
  'neck_pitch', 'head_pitch', 'head_yaw', 'head_roll',
  'right_hip_yaw', 'right_hip_roll', 'right_hip_pitch', 'right_knee', 'right_ankle'
];
const clamp = THREE.MathUtils.clamp;
const tau = Math.PI * 2;
const up = new THREE.Vector3(0, 0, 1);
let assetCache;

async function fetchAsset(name) {
  const response = await fetch(new URL('./assets/' + name, import.meta.url));
  if (!response.ok) throw new Error('Could not load the Microduck model.');
  return response;
}

function loadAssets() {
  if (!assetCache) {
    assetCache = Promise.all([
      fetchAsset('duck-3d.json').then(r => r.json()),
      fetchAsset('duck-3d.bin').then(r => r.arrayBuffer()),
      fetchAsset('duck-3d-low.json').then(r => r.json()),
      fetchAsset('duck-3d-low.bin').then(r => r.arrayBuffer()),
      fetchAsset('duck-3d-mid.json').then(r => r.json()),
      fetchAsset('duck-3d-mid.bin').then(r => r.arrayBuffer())
    ]).catch(error => {
      assetCache = null;
      throw error;
    });
  }
  return assetCache;
}

function quat(values) {
  const [w, x, y, z] = values;
  return new THREE.Quaternion(x, y, z, w);
}

/** Merge fixed visual parts per moving body to keep a field of ducks inexpensive to draw. */
function bodyGeometry(body, meshes, buffer) {
  if (!body.geoms.length) return null;
  const positions = [], normals = [], colours = [], indices = [];
  const vertex = new THREE.Vector3(), normal = new THREE.Vector3();
  for (const geom of body.geoms) {
    const mesh = meshes[geom.mesh];
    const pos = new Float32Array(buffer, mesh.positions.offset, mesh.positions.count * 3);
    const norm = new Float32Array(buffer, mesh.normals.offset, mesh.normals.count * 3);
    const ix = new Uint32Array(buffer, mesh.indices.offset, mesh.indices.count);
    const local = new THREE.Vector3(...geom.pos), rotation = quat(geom.quat);
    const colour = new THREE.Color().setRGB(...geom.rgba.slice(0, 3), THREE.SRGBColorSpace);
    const start = positions.length / 3;
    for (let i = 0; i < pos.length; i += 3) {
      vertex.fromArray(pos, i).applyQuaternion(rotation).add(local);
      normal.fromArray(norm, i).applyQuaternion(rotation);
      positions.push(vertex.x, vertex.y, vertex.z);
      normals.push(normal.x, normal.y, normal.z);
      colours.push(colour.r, colour.g, colour.b);
    }
    for (const i of ix) indices.push(start + i);
  }
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geom.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
  geom.setAttribute('color', new THREE.Float32BufferAttribute(colours, 3));
  geom.setIndex(indices);
  geom.computeBoundingSphere();
  return geom;
}

function createBadge(text, colour) {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = 96;
  const ctx = canvas.getContext('2d');
  ctx.beginPath(); ctx.arc(48, 48, 36, 0, tau);
  ctx.fillStyle = '#20292c'; ctx.fill();
  ctx.lineWidth = 6; ctx.strokeStyle = colour; ctx.stroke();
  ctx.fillStyle = '#fff'; ctx.font = 'bold 43px system-ui, sans-serif';
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText(text, 48, 49);
  const map = new THREE.CanvasTexture(canvas);
  map.colorSpace = THREE.SRGBColorSpace;
  const badge = new THREE.Sprite(new THREE.SpriteMaterial({
    map, transparent: true, depthTest: false, depthWrite: false, sizeAttenuation: false
  }));
  badge.renderOrder = 6;
  return badge;
}

function shadowTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = 64;
  const ctx = canvas.getContext('2d');
  const gradient = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
  gradient.addColorStop(0, 'rgba(0,0,0,.45)');
  gradient.addColorStop(.4, 'rgba(0,0,0,.30)');
  gradient.addColorStop(1, 'rgba(0,0,0,0)');
  ctx.fillStyle = gradient; ctx.fillRect(0, 0, 64, 64);
  return new THREE.CanvasTexture(canvas);
}

/** One draw call per body (legs, head, etc.) for the whole field of ghosts. */
function instanceField(scene, geometries, root, capacity, ghost=false) {
  const material = new THREE.MeshStandardMaterial({vertexColors: true, roughness: .74, metalness: .05});
  // Tinted enamel keeps individual arms, servos and skate shells legible even in a tight pack.
  if(ghost){
    material.roughness=.84;
    material.polygonOffset=true;
    material.polygonOffsetFactor=1;
    material.polygonOffsetUnits=2;
  }
  const meshes = geometries.map(g => {
    if (!g) return null;
    const mesh = new THREE.InstancedMesh(g, material, capacity);
    mesh.count = 0;
    mesh.frustumCulled = false;
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    root.add(mesh);
    return mesh;
  });
  scene.add(root);
  return {
    capacity, meshes,
    end(count) {
      for (const mesh of meshes) {
        if (!mesh) continue;
        mesh.count = count;
        mesh.instanceMatrix.needsUpdate = true;
        if(mesh.instanceColor)mesh.instanceColor.needsUpdate=true;
      }
    },
    dispose(){
      for(const mesh of meshes)mesh?.dispose();
      material.dispose();
      root.removeFromParent();
    }
  };
}

function createSkater(model, shadowMap, racer) {
  const root=new THREE.Group(), group=new THREE.Group(), hinges=[], anchors=[];
  const bodies=model.bodies.map(()=>new THREE.Group());
  for(const [i,b] of model.bodies.entries()){
    const node=bodies[i];
    if (b.parent<0) root.add(node);
    else {
      node.position.fromArray(b.pos);node.quaternion.copy(quat(b.quat));bodies[b.parent].add(node);
    }
    const joint=b.joint, slot=joint?jointNames.indexOf(joint.name):-1;
    if(slot>=0){
      const hinge=new THREE.Group(), offset=new THREE.Group();
      hinge.position.fromArray(joint.pivot);offset.position.copy(hinge.position).negate();
      node.add(hinge);hinge.add(offset);bodies[i]=offset;
      hinges.push({hinge,slot,axis:new THREE.Vector3(...joint.axis).normalize(),ref:joint.reference||0});
    }
    anchors.push(bodies[i]);
  }
  const shadow=new THREE.Mesh(new THREE.PlaneGeometry(.24,.19),
    new THREE.MeshBasicMaterial({map:shadowMap,transparent:true,depthWrite:false}));
  shadow.renderOrder=1; group.add(shadow);
  const hex=racer.colour||'#c3dfdf';
  const sourceColour=new THREE.Color(hex);
  // Mixing towards white preserves the black sensors and motor detail of the real duck.
  const tint=sourceColour.clone().lerp(new THREE.Color(0xffffff),racer.ghost?.24:.34);
  const white=new THREE.Color(0xffffff);
  const badge=racer.bib!=null?createBadge(String(racer.bib),hex):null;
  if(badge)group.add(badge);
  return {
    group,badge,
    pose(p,focus,field,index) {
      root.position.fromArray(p);root.quaternion.set(p[4],p[5],p[6],p[3]).normalize();
      for(const j of hinges)j.hinge.quaternion.setFromAxisAngle(j.axis,p[7+j.slot]-j.ref);
      root.updateMatrixWorld(true);
      for(let i=0;i<anchors.length;i++){
        const mesh=field.meshes[i];
        if(!mesh)continue;
        mesh.setMatrixAt(index,anchors[i].matrixWorld);
        mesh.setColorAt(index,focus?white:tint);
      }
      shadow.position.set(p[0],p[1],.004);shadow.rotation.z=Math.atan2(2*(p[3]*p[6]+p[4]*p[5]),1-2*(p[5]**2+p[6]**2));
      shadow.material.opacity=focus?1:.55;
      if(badge)badge.position.set(p[0],p[1],Math.max(.23,p[2]+.17));
    },
    dispose() {
      shadow.geometry.dispose();shadow.material.dispose();
      if(badge){badge.material.map.dispose();badge.material.dispose();}
      group.removeFromParent();
    }
  };
}

function roadMesh(track) {
  const group = new THREE.Group();
  const geometry = new THREE.BufferGeometry(), points = [], colours = [];
  function quad(a,b,c,d,hex,z) {
    const rgb = new THREE.Color(hex);
    for (const point of [a,b,c,a,c,d]) {
      points.push(point[0],point[1],z);
      colours.push(rgb.r,rgb.g,rgb.b);
    }
  }
  function offset(edge, centre, dist) {
    const x=edge[0]-centre[0],y=edge[1]-centre[1],w=Math.hypot(x,y)||1;
    return [edge[0]+dist*x/w,edge[1]+dist*y/w];
  }
  function mix(a,b,f) {return [a[0]+(b[0]-a[0])*f,a[1]+(b[1]-a[1])*f];}
  const c=track.centre,l=track.left,r=track.right;
  for (let i=0; i<c.length; i++) {
    const j=(i+1)%c.length;
    quad(l[i],l[j],r[j],r[i],'#454f56',0);
    for (const e of [l,r]) {
      quad(e[i],e[j],offset(e[j],c[j],.061),offset(e[i],c[i],.061),i%4<2?'#f4ebd8':'#dc7357',.0003);
      quad(e[i],e[j],offset(e[j],c[j],-.018),offset(e[i],c[i],-.018),'#f2eddd',.001);
    }
    if (i%3===0) {
      const a=mix(c[i],c[j],.16), b=mix(c[i],c[j],.65);
      const x=c[j][0]-c[i][0],y=c[j][1]-c[i][1],h=Math.hypot(x,y)||1;
      const nx=-y/h*.005,ny=x/h*.005;
      quad([a[0]+nx,a[1]+ny],[b[0]+nx,b[1]+ny],[b[0]-nx,b[1]-ny],[a[0]-nx,a[1]-ny],'#aeb8b5',.0011);
    }
  }
  const dx=c[1][0]-c.at(-1)[0],dy=c[1][1]-c.at(-1)[1],dist=Math.hypot(dx,dy)||1;
  const step=[.059*dx/dist,.059*dy/dist];
  for (let col=0; col<12; col++) for(let row=0;row<2;row++) {
    const a=mix(l[0],r[0],col/12), b=mix(l[0],r[0],(col+1)/12);
    const k=(p,n)=>[p[0]+step[0]*n,p[1]+step[1]*n];
    quad(k(a,row),k(b,row),k(b,row+1),k(a,row+1),(row+col)%2?'#e8e7d6':'#28312f',.0016);
  }
  geometry.setAttribute('position',new THREE.Float32BufferAttribute(points,3));
  geometry.setAttribute('color',new THREE.Float32BufferAttribute(colours,3));
  const road = new THREE.Mesh(geometry,new THREE.MeshBasicMaterial({vertexColors:true,side:THREE.DoubleSide}));
  group.add(road);
  return group;
}

function trackBounds(track) {
  const bounds = new THREE.Box3();
  const p=new THREE.Vector3();
  for (const edge of [track.left,track.right]) for (const [x,y] of edge) bounds.expandByPoint(p.set(x,y,0));
  return bounds;
}

/** Use nearby course tangents for a smooth camera heading. Root bob and lean stay on the robot. */
function tangentAt(points,x,y,heading) {
  let best=Infinity,index=0;
  for(let i=0;i<points.length;i++){
    const p=points[i],d=(p[0]-x)**2+(p[1]-y)**2;
    if (d<best){best=d;index=i;}
  }
  const previous=points[(index+points.length-1)%points.length],next=points[(index+1)%points.length];
  const a=Math.atan2(next[1]-previous[1],next[0]-previous[0]);
  return Math.cos(a-heading)<0 ? a+Math.PI : a;
}

function disposeObject(root) {
  root.traverse(obj=>{
    obj.geometry?.dispose();
    if(obj.material) obj.material.dispose();
  });
  root.removeFromParent();
}

export async function createRace3D(canvas,{onContextLost}={}) {
  const [model,buffer,lod,lodBuffer,mid,midBuffer] = await loadAssets();
  const renderer = new THREE.WebGLRenderer({canvas,alpha:false,antialias:true,powerPreference:'high-performance'});
  const scene=new THREE.Scene(), camera=new THREE.PerspectiveCamera(40,1,.02,100);
  camera.up.copy(up);
  scene.background=new THREE.Color('#c9dad2'); scene.fog=new THREE.FogExp2('#c9dad2',.036);
  const skyLight=new THREE.HemisphereLight('#f5fbff','#e0d0a8',2.05);
  skyLight.up.copy(up); skyLight.position.copy(up);
  scene.add(skyLight);
  const sun=new THREE.DirectionalLight('#fff0db',2.35);
  sun.position.set(-2,-3,5); scene.add(sun);
  const fill=new THREE.DirectionalLight('#d1e3ff',.72);
  fill.position.set(2,3,1.8); scene.add(fill);
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(140,140),
                           new THREE.MeshBasicMaterial({color:'#689282'}));
  ground.position.z=-.015; ground.material.depthWrite=false; ground.renderOrder=-10; scene.add(ground);
  const geometries=model.bodies.map(b=>bodyGeometry(b,model.meshes,buffer));
  const lowMeshes=lod.bodies.map(b=>bodyGeometry(b,lod.meshes,lodBuffer));
  const mediumMeshes=mid.bodies.map(b=>bodyGeometry(b,mid.meshes,midBuffer));
  const first=instanceField(scene,geometries,new THREE.Group(),1);
  const closePack=instanceField(scene,mediumMeshes,new THREE.Group(),30,true);
  let pack=instanceField(scene,lowMeshes,new THREE.Group(),8,true);
  const skaters=new Map(), shadowMap=shadowTexture();
  let track=null, road=null, bounds=null, last=null, state=null;
  let width=0,height=0, orbit=0, pitchShift=0, zoom=1, drag=null;
  const centre=new THREE.Vector3(),eye=new THREE.Vector3(),projected=new THREE.Vector3();
  const focusPoint = new THREE.Vector3();

  function setTrack(value) {
    if(track===value) return;
    if(road)disposeObject(road);
    track=value;bounds=trackBounds(value);road=roadMesh(value);scene.add(road);
    last=null;
  }
  function draw(racers,options={}) {
    state=[racers,options];
    const w=canvas.clientWidth,h=canvas.clientHeight;
    if(!track||!w||!h)return;
    const dpr=Math.min(window.devicePixelRatio || 1,1.65,Math.sqrt(1450000/(w*h)));
    if(w!==width||h!==height||Math.abs(renderer.getPixelRatio()-dpr)>.015){
      width=w;height=h;renderer.setPixelRatio(dpr);renderer.setSize(w,h,false);
      camera.aspect=w/h;camera.updateProjectionMatrix();
    }
    const visible=new Set(racers.map(r=>r.id));
    for(const [id,duck] of skaters) if(!visible.has(id)){duck.dispose();skaters.delete(id);}
    const target=racers.find(r=>r.id===options.focusId)||racers.find(r=>!r.ghost)||racers[0];
    const extra=racers.length-1;
    if(extra>pack.capacity){
      pack.dispose();
      pack=instanceField(scene,lowMeshes,new THREE.Group(),2**Math.ceil(Math.log2(extra)),true);
    }
    const nearby=new Set();
    if(options.camera!=='grandstand' && target){
      const distance=r=>Math.hypot(r.pose[0]-target.pose[0],r.pose[1]-target.pose[1]);
      const nearest=racers.filter(r=>r!==target && distance(r)<2)
        .sort((a,b)=>distance(a)-distance(b)).slice(0,w>560?30:16);
      for(const r of nearest)nearby.add(r.id);
    }
    let closeSlot=0,farSlot=0;
    for(const racer of racers){
      let duck=skaters.get(racer.id);
      if(!duck) {
        duck=createSkater(model,shadowMap,racer);
        skaters.set(racer.id,duck);scene.add(duck.group);
      }
      const focus=target===racer;
      if(focus)duck.pose(racer.pose,true,first,0);
      else if(nearby.has(racer.id))duck.pose(racer.pose,false,closePack,closeSlot++);
      else duck.pose(racer.pose,false,pack,farSlot++);
    }
    first.end(target?1:0);
    closePack.end(closeSlot);
    pack.end(farSlot);
    const mode=options.camera==='grandstand'?'grandstand':'chase';
    const t=Number(options.time)||0;
    const elapsed=t-(last?.time??t),changed=!last||last.mode!==mode||last.id!==target?.id||elapsed<0||elapsed>.35;
    let heading=0;
    if(target) {
      const p=target.pose;
      const yaw=Math.atan2(2*(p[3]*p[6]+p[4]*p[5]), 1-2*(p[5]**2+p[6]**2));
      const a=tangentAt(track.centre,p[0],p[1],yaw);
      heading=changed?a:last.heading+Math.atan2(Math.sin(a-last.heading),Math.cos(a-last.heading))*(1-Math.exp(-Math.max(elapsed,0)*5));
      focusPoint.set(p[0]+Math.cos(heading)*.085,p[1]+Math.sin(heading)*.085,.114);
    }
    if (mode==='grandstand'||!target) {
      bounds.getCenter(centre);
      const size=bounds.getSize(new THREE.Vector3()), viewAngle=orbit-Math.PI/2;
      const elevation=clamp(.92+pitchShift,.25,1.49);
      const horizontalExtent=Math.abs(size.x*Math.sin(viewAngle))+Math.abs(size.y*Math.cos(viewAngle));
      const depthExtent=Math.abs(size.x*Math.cos(viewAngle))+Math.abs(size.y*Math.sin(viewAngle));
      const distance=zoom*(Math.max(horizontalExtent/camera.aspect,depthExtent*Math.sin(elevation))/(2*Math.tan(THREE.MathUtils.degToRad(camera.fov)/2)) + depthExtent*Math.cos(elevation)/2)*1.13;
      eye.set(Math.cos(viewAngle)*Math.cos(elevation),Math.sin(viewAngle)*Math.cos(elevation),Math.sin(elevation)).multiplyScalar(Math.max(1,distance)).add(centre);
    }else{
      centre.copy(focusPoint);
      const elevation=clamp(.48+pitchShift,.13,1.4);
      const viewAngle=heading+1.13+orbit;
      const distance=1.12*zoom;
      eye.set(Math.cos(viewAngle)*Math.cos(elevation),Math.sin(viewAngle)*Math.cos(elevation),Math.sin(elevation)).multiplyScalar(distance).add(centre);
    }
    camera.position.copy(eye); camera.lookAt(centre); camera.updateMatrixWorld();
    // Bibs can spread out on the screen even when the recorded ducks occupy
    // almost the same space. Articulated meshes always stay on their true paths.
    const diameter=Math.min(mode==='grandstand'?22:29, h*.095);
    const scale=diameter/h * 2*Math.tan(THREE.MathUtils.degToRad(camera.fov)/2) * 1.33;
    const placed=[];
    const cameraDistance=r=>(r.pose[0]-eye.x)**2+(r.pose[1]-eye.y)**2+(r.pose[2]-eye.z)**2;
    const ordered=target?[target,...racers.filter(r=>r!==target).sort((a,b)=>cameraDistance(a)-cameraDistance(b))]:racers;
    for(const racer of ordered){
      const badge=skaters.get(racer.id).badge;
      if(!badge)continue;
      if(placed.length>=Math.max(7,Math.floor(w/90))){badge.visible=false;continue;}
      badge.scale.setScalar(scale);
      projected.copy(badge.position).project(camera);
      const x=(projected.x+1)*w/2, y=(1-projected.y)*h/2, z=projected.z;
      badge.visible=z>0 && z<1 && x>0 && x<w && y>0 && y<h;
      if(!badge.visible)continue;
      let spot=null;
      // Prefer the anchor point; only shift in screen space if another bib covers it.
      for(const [dx,dy] of [[0,0],[0,-1],[-1,0],[1,0],[-.8,-.8],[.8,-.8],[-1.6,-.8],[1.6,-.8],[0,-2]]){
        const px=x+dx*(diameter+2), py=y+dy*(diameter+2);
        if(px<diameter/2||px>w-diameter/2||py<diameter/2||py>h-diameter/2)continue;
        if(placed.every(o=>(o[0]-px)**2+(o[1]-py)**2>(diameter+1)**2)){spot=[px,py];break;}
      }
      if(!spot){badge.visible=false;continue;}
      placed.push(spot);
      badge.position.set(2*spot[0]/w-1,1-2*spot[1]/h,z).unproject(camera);
    }
    renderer.render(scene,camera);
    last={id:target?.id,mode,heading,time:t};
  }
  const redraw=()=>{if(state)draw(...state);};
  const events=new AbortController(), signal=events.signal;
  canvas.addEventListener('pointerdown',e=>{
    if(e.button!==0)return;
    drag={id:e.pointerId,x:e.clientX,y:e.clientY};
    canvas.setPointerCapture(e.pointerId);
  },{signal});
  canvas.addEventListener('pointermove',e=>{
    if(!drag||drag.id!==e.pointerId)return;
    orbit-=(e.clientX-drag.x)*.009;
    pitchShift=clamp(pitchShift+(e.clientY-drag.y)*.006,-.65,.9);
    drag.x=e.clientX;drag.y=e.clientY;redraw();
  },{signal});
  const stop=e=>{if(drag?.id===e.pointerId)drag=null;};
  canvas.addEventListener('pointerup',stop,{signal});
  canvas.addEventListener('pointercancel',stop,{signal});
  canvas.addEventListener('wheel',e=>{
    if(!e.ctrlKey&&!e.shiftKey)return;
    e.preventDefault(); zoom=clamp(zoom*Math.exp(e.deltaY*.0014),.55,2.5);redraw();
  },{passive:false,signal});
  canvas.addEventListener('dblclick',()=>{orbit=0;pitchShift=0;zoom=1;redraw();},{signal});
  canvas.addEventListener('webglcontextlost',e=>{e.preventDefault();onContextLost?.();},{signal});
  const resize=new ResizeObserver(redraw);resize.observe(canvas);
  return {setTrack,draw,dispose(){
    resize.disconnect();events.abort();
    for(const duck of skaters.values())duck.dispose();
    skaters.clear(); if(road)disposeObject(road); disposeObject(ground);shadowMap.dispose();
    first.dispose();closePack.dispose();pack.dispose();
    for(const g of [...geometries,...mediumMeshes,...lowMeshes])g?.dispose();
    renderer.dispose();
    state=null;
  }};
}
