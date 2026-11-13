"""Film recorded lap poses on the re:Invent circuit using Pollen's visual meshes."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
from io import BytesIO
import json
from pathlib import Path
from unittest.mock import patch
from xml.etree import ElementTree as ET

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from arena import course
from arena.constants import JOINT_NAMES
from arena.manifest import load_event


def track_surface(root, assets, track):
    centre, left, right = (np.asarray(track[k]) for k in ('centre', 'left', 'right'))
    # UV coordinates use the same metre-space points as the simulator and the overhead view.
    points = np.vstack([left, right])
    lower, upper = points.min(axis=0) - 1.2, points.max(axis=0) + 1.2
    span = upper - lower
    resolution = 265
    size = (span * resolution).astype(int) + 1
    rng = np.random.default_rng(2018)
    grain = rng.integers(-3, 4, (size[1], size[0], 1), dtype=np.int8)
    im = Image.fromarray((np.array([61, 93, 87], np.int16) + grain).astype(np.uint8))
    brush = ImageDraw.Draw(im)
    def xy(point):
        return (int((point[0] - lower[0]) * (size[0]-1) / span[0]),
                int((upper[1] - point[1]) * (size[1]-1) / span[1]))
    def stripe(points, colour, metres):
        brush.line([xy(p) for p in points], fill=colour, width=max(1,round(metres*resolution)),joint='curve')
    for i in range(len(centre)):
        j = (i+1) % len(centre)
        brush.polygon([xy(p) for p in (left[i],left[j],right[j],right[i])],fill=(65,74,86))
    for edge in (left,right):
        outer = edge + .095*(edge-centre)/np.linalg.norm(edge-centre,axis=1)[:,None]
        for i in range(len(edge)):
            j=(i+1)%len(edge)
            colour=(220,104,83) if i%4<2 else (236,224,196)
            brush.polygon([xy(p) for p in (edge[i],edge[j],outer[j],outer[i])],fill=colour)
        stripe([*edge,edge[0]],(247,238,214),.016)
    for i in range(0,len(centre),3):
        stripe([centre[i],.48*centre[(i+1)%len(centre)]+.52*centre[i]],(171,178,172),.012)
    tangent = centre[1]-centre[-1]
    tangent = tangent/np.linalg.norm(tangent)
    for i in range(12):
        a,b = [left[0]+(right[0]-left[0])*k/12 for k in (i,i+1)]
        for j in (0,1):
            off = tangent*.055*j
            brush.polygon([xy(p+off) for p in (a,b,b+tangent*.055,a+tangent*.055)],
                          fill=(230,228,213) if (i+j)%2 else (31,36,42))
    blob=BytesIO()
    im.save(blob,format='PNG')
    assets['circuit.png']=blob.getvalue()
    asset=root.find('asset')
    ET.SubElement(asset,'texture',name='circuit_texture',type='2d',file='circuit.png')
    ET.SubElement(asset,'material',name='circuit_material',texture='circuit_texture',texuniform='false',
                  specular='0',shininess='0',reflectance='0')
    # Only the top of this thin enclosed mesh is visible. MuJoCo needs a solid mesh.
    xmin,ymin=lower
    xmax,ymax=upper
    corners=[(xmin,ymin),(xmax,ymin),(xmax,ymax),(xmin,ymax)]
    vertices=[(x,y,z) for z in (-.006,-.001) for x,y in corners]
    faces=[(4,5,6),(4,6,7),(2,1,0),(3,2,0),(0,1,5),(0,5,4),(1,2,6),(1,6,5),(2,3,7),(2,7,6),(3,0,4),(3,4,7)]
    def values(array):
        return ' '.join(str(x) for row in array for x in row)
    ET.SubElement(asset,'mesh',name='track_board', vertex=values(vertices),face=values(faces),
                  texcoord='0 1 1 1 1 0 0 0 0 1 1 1 1 0 0 0')
    world=root.find('worldbody')
    ET.SubElement(world,'geom',type='mesh',mesh='track_board',material='circuit_material',
                  contype='0',conaffinity='0',group='2')
    floor=next(g for g in root.iter('geom') if g.get('type')=='plane')
    floor.set('pos','0 0 -.009')
    floor.set('rgba','.14 .23 .21 1')


def visual_model(track):
    with patch.object(course,'_strip_visuals',lambda root:None):
        xml,assets=course.build_course_xml(replace(load_event('got-a-talent'),rig='rollers'))
    root=ET.fromstring(xml)
    world=root.find('worldbody')
    ghost=deepcopy(world.find("body[@name='trunk_base']"))
    for node in ghost.iter():
        if 'name' in node.attrib:
            node.set('name','ghost_'+node.get('name'))
        if node.tag=='geom' and node.get('class')=='visual':
            node.attrib.pop('material',None)
            node.set('rgba','.30 .79 .96 .43')
    world.append(ghost)
    root.remove(root.find('keyframe'))
    ET.SubElement(root.find('asset'),'texture',type='skybox',builtin='gradient',
                  rgb1='.33 .48 .53',rgb2='.11 .17 .22',width='128',height='768')
    ET.SubElement(world,'light',pos='0 0 5',dir='-.35 -.25 -1',directional='true',
                  diffuse='.72 .70 .63',ambient='.20 .20 .20',castshadow='true')
    track_surface(root,assets,track)
    model=course._compiled(course._motorized(ET.tostring(root,encoding='unicode'),assets))
    model.vis.global_.offwidth=960
    model.vis.global_.offheight=540
    model.vis.headlight.ambient[:]=[.42,.42,.42]
    model.vis.headlight.diffuse[:]=[.58,.58,.58]
    return model


def pose_at(frames,t):
    index=np.clip(np.searchsorted(frames[:,0],t,side='right'),1,len(frames)-1)
    a,b=frames[index-1],frames[index].copy()
    if a[4:8] @ b[4:8] < 0:
        b[4:8]*=-1
    mix=np.clip((t-a[0])/max(b[0]-a[0],1e-9),0,1)
    pose=a[1:] + mix*(b[1:]-a[1:])
    pose[3:7]/=np.linalg.norm(pose[3:7])
    return pose


def render(source,output,fps=24):
    race=json.loads(source.read_text())
    entries=[np.array(race[name]['frames']) for name in ('challenger','baseline')]
    model=visual_model(race['track'])
    data=mujoco.MjData(model)
    indices=[]
    for prefix in ('','ghost_'):
        trunk=model.body(prefix+'trunk_base').id
        free=model.jnt_qposadr[model.body_jntadr[trunk]]
        joints=[model.joint(prefix+n).qposadr[0] for n in JOINT_NAMES]
        indices.append((slice(free,free+7),joints))
    cam=mujoco.MjvCamera()
    cam.distance=1.08
    cam.elevation=-23
    cam.azimuth=135
    focus=entries[0][0,1:3].copy()
    output.parent.mkdir(parents=True,exist_ok=True)
    with mujoco.Renderer(model,height=540,width=960) as renderer, imageio.get_writer(
            str(output),fps=fps,codec='libx264',macro_block_size=1,quality=None,
            ffmpeg_params=['-crf','24','-movflags','+faststart']) as video:
        for i in range(1+int(np.ceil(max(r[-1,0] for r in entries)*fps))):
            t=i/fps
            poses=[pose_at(rows,t) for rows in entries]
            for (free,joints),pose in zip(indices,poses):
                data.qpos[free]=pose[:7]
                data.qpos[joints]=pose[7:]
            mujoco.mj_forward(model,data)
            # Follow the racer, then follow the remaining ghost after the racer finishes.
            active=poses[0] if t<=entries[0][-1,0] else poses[1]
            focus+=.23*(active[:2]-focus)
            cam.lookat[:]=[*focus,.105]
            renderer.update_scene(data,camera=cam)
            video.append_data(renderer.render())


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('replay',type=Path)
    ap.add_argument('output',type=Path)
    args=ap.parse_args()
    render(args.replay,args.output)
