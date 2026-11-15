"""Film recorded lap poses on the re:Invent circuit using Pollen's visual meshes."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
from io import BytesIO
from math import atan2, degrees, exp
import json
from pathlib import Path
from unittest.mock import patch
from xml.etree import ElementTree as ET

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

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


# Matches the bib order and colours in the overhead replay.
FIELD_COLOURS = ('ff986a', '88cfff', 'f7d66b', 'c6a6ff', '6bdfae',
                 'ffa4ce', 'e6e6e1', 'a2bcff')
COLOURED_MESHES = {'left_shell', 'right_shell', 'top_head_shell',
                   'upper_leg_rigidity_plate', 'rim', 'ankle_l_v1', 'ankle_r_v1'}


def visual_model(track, colours=None):
    with patch.object(course,'_strip_visuals',lambda root:None):
        xml,assets=course.build_course_xml(replace(load_event('got-a-talent'),rig='rollers'))
    root=ET.fromstring(xml)
    world=root.find('worldbody')
    original = world.find("body[@name='trunk_base']")
    template = deepcopy(original)
    for i, colour in enumerate(colours if colours else (None, 'ghost')):
        duck = original if i == 0 else deepcopy(template)
        prefix = '' if i == 0 else f'ghost{i}_'
        for node in duck.iter():
            if prefix and 'name' in node.attrib:
                node.set('name', prefix + node.get('name'))
            if node.tag != 'geom' or node.get('class') != 'visual' or not colour:
                continue
            if colour == 'ghost':
                rgba = '.30 .79 .96 .43'
            elif node.get('mesh') in COLOURED_MESHES:
                rgb = [int(colour[k:k+2], 16) / 255 for k in (0, 2, 4)]
                rgba = ' '.join(str(channel) for channel in [*rgb, 1.0])
            else:
                continue
            node.attrib.pop('material', None)
            node.set('rgba', rgba)
        if i:
            world.append(duck)
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


def field_laps(index_path):
    archive = index_path.parent
    records = json.loads(index_path.read_text())
    if not records:
        raise ValueError('The run tree contains no racers.')
    runs = [json.loads((archive / item['id'] / 'race.json').read_text()) for item in records]
    first = runs[0]
    if any(run['track'] != first['track'] or run['seed'] != first['seed']
           or run['joints'] != list(JOINT_NAMES) for run in runs):
        raise ValueError('The field requires a shared course, start seed and joint order.')
    laps = [run['challenger'] for run in runs]
    order = sorted(range(len(laps)), key=lambda i: (not laps[i]['finished'], laps[i]['time']))
    colours = [FIELD_COLOURS[i % len(FIELD_COLOURS)] for i in range(len(laps))]
    # The centreline reference shares the field clock but does not enter the standings.
    return (first['track'], [np.array(lap['frames']) for lap in [*laps, first['baseline']]],
            [*order, len(laps)], [*colours, 'ghost'])


def render(source, output, fps=24, field=False, poster=None, solo=False):
    if field and solo:
        raise ValueError('Choose either a solo lap or the archived field.')
    if field:
        track, entries, order, colours = field_laps(source)
    else:
        race = json.loads(source.read_text())
        track = race['track']
        if solo:
            if 'field' in race:
                if len(race['field']) != 1:
                    raise ValueError('A solo field replay must contain exactly one driver.')
                lap = race['field'][0]['run']
            else:
                lap = race['challenger']
            entries = [np.array(lap['frames'])]
            order, colours = [0], [None]
        else:
            entries = [np.array(race[name]['frames']) for name in ('challenger', 'baseline')]
            order = [0, 1]
            colours = None
    model = visual_model(track, colours)
    data = mujoco.MjData(model)
    indices = []
    for i in range(len(entries)):
        prefix = '' if i == 0 else f'ghost{i}_'
        trunk = model.body(prefix + 'trunk_base').id
        free = model.jnt_qposadr[model.body_jntadr[trunk]]
        joints = [model.joint(prefix + n).qposadr[0] for n in JOINT_NAMES]
        indices.append((slice(free, free+7), joints))

    cam = mujoco.MjvCamera()
    cam.distance = 1.08
    cam.elevation = -23
    cam.azimuth = 135
    focus = entries[order[0]][0, 1:3].copy()
    output.parent.mkdir(parents=True, exist_ok=True)
    finish = max(frames[-1, 0] for frames in entries)
    duration = finish + (0.8 if solo else 0)
    with mujoco.Renderer(model, height=540, width=960) as renderer, imageio.get_writer(
            str(output), fps=fps, codec='libx264', macro_block_size=1, quality=None,
            ffmpeg_params=['-crf', '24', '-movflags', '+faststart']) as video:
        for i in range(1 + int(np.ceil(duration * fps))):
            t = i / fps
            poses = [pose_at(rows, t) for rows in entries]
            for (free, joints), pose in zip(indices, poses):
                data.qpos[free] = pose[:7]
                data.qpos[joints] = pose[7:]
            mujoco.mj_forward(model, data)
            active = [k for k in order if entries[k][-1, 0] >= t]
            leader = active[0] if active else order[-1]
            if field:
                # A broadcast camera, independent of robot poses: pan with the winner,
                # and widen just enough for the racing pack to stay visible.
                # Keep finishers on camera at their measured finish poses as
                # the remaining racers and centreline ghost cross the line.
                xy = np.array([pose[:2] for pose in poses])
                target = .46 * poses[leader][:2] + .54 * (xy.min(0) + xy.max(0)) / 2
                focus += (1 - exp(-12 / fps)) * (target - focus)
                where = pose_at(entries[leader], t + .25)[:2] - pose_at(entries[leader], t - .2)[:2]
                if np.linalg.norm(where) > .002:
                    bearing = degrees(atan2(where[1], where[0])) + 118
                    if i == 0:
                        cam.azimuth = bearing
                    turn = (bearing - cam.azimuth + 180) % 360 - 180
                    cam.azimuth += np.clip(turn * (1-exp(-2.2 / fps)), -75 / fps, 75 / fps)
                az = np.radians(cam.azimuth)
                side = np.array([-np.sin(az), np.cos(az)])
                longitudinal = np.array([np.cos(az), np.sin(az)])
                spread = xy - focus
                width = np.max(np.abs(spread @ side))
                depth = np.max(np.abs(spread @ longitudinal))
                distance = max(.87, .55 + 1.55 * width + 1.12 * depth)
                cam.distance += (1-exp(-5/fps)) * (distance-cam.distance)
                cam.elevation = -25
            else:
                focus += .23 * (poses[leader][:2] - focus)
            cam.lookat[:] = [*focus, .105]
            renderer.update_scene(data, camera=cam)
            frame = renderer.render()
            if solo:
                # The film holds the finish pose briefly; its clock stops at the
                # recorded crossing time rather than including that final hold.
                overlay = Image.fromarray(frame)
                draw = ImageDraw.Draw(overlay)
                draw.rounded_rectangle((18, 18, 330, 100), radius=9, fill=(24, 34, 39))
                draw.text((32, 29), 'Microduck | re:Invent 2018', fill=(245, 242, 230),
                          font=ImageFont.load_default(size=18))
                clock = f'{min(t, finish):.2f} s'
                draw.text((32, 60), clock + ('  FINISH' if t >= finish else ''), fill=(255, 206, 88),
                          font=ImageFont.load_default(size=24))
                frame = np.asarray(overlay)
            video.append_data(frame)
            if poster and i == round(min(duration * .7, 19.0) * fps):
                Image.fromarray(frame).save(poster, quality=88, optimize=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('replay', type=Path, help='Replay JSON, or lineage/index.json for --field')
    ap.add_argument('output', type=Path, help='Output MP4')
    ap.add_argument('--field', action='store_true', help='Film all compatible archived runs together')
    ap.add_argument('--solo', action='store_true', help='Film only the trained duck, with a lap clock')
    ap.add_argument('--poster', type=Path, help='Save a representative JPEG frame')
    args = ap.parse_args()
    render(args.replay, args.output, field=args.field, poster=args.poster, solo=args.solo)
