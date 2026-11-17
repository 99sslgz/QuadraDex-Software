"""Run synchronized full-arm + finger tracking and the LEAP V1 motion simulator.

python -m teleoperation.run              # camera picker + tracking + simulation
python -m teleoperation.run --demo       # synthetic motion, no camera required
"""
from __future__ import annotations
import argparse
from collections import deque
import json
from pathlib import Path
import socket
import sys
import time
import uuid

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/"cv_hand_tracking"))
from track_hand import LatestCamera, Output, draw, BUTTONS
from camera_picker import choose_camera, discover_cameras, CameraChoice
from calibration_ui import CalibrationCapture, finger_ranges
from hand_geometry import StreamGate, make_calibration
from teleoperation.arm_tracking import ArmAnchor
from teleoperation.robot_model import RobotModel
from teleoperation.retarget import Retargeter, SignalGenerator
from teleoperation.simulator import Simulation
from teleoperation.vision import WholeArmVision, ensure_models

WINDOW="QuadraDex | Full arm + LEAP control"


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--camera',type=int)
    parser.add_argument('--list-cameras',action='store_true')
    parser.add_argument('--backend',choices=('auto','msmf','dshow'),default='auto')
    parser.add_argument('--side',choices=('right','left'),default='right')
    parser.add_argument('--swap-handedness',action='store_true')
    parser.add_argument('--width',type=int,default=1280)
    parser.add_argument('--height',type=int,default=720)
    parser.add_argument('--camera-fps',type=float,default=30)
    parser.add_argument('--min-visibility',type=float,default=.6)
    parser.add_argument('--max-age-ms',type=float,default=120)
    parser.add_argument('--output-hz',type=float,default=25)
    parser.add_argument('--arm-config',type=Path,default=ROOT/'demo_arm.json')
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--duration',type=float)
    parser.add_argument('--demo',action='store_true',help='Synthetic arm/finger motion, clearly marked in exported packets')
    parser.add_argument('--snapshot',type=Path,help='Save the simulated robot view on exit; no camera image is saved')
    parser.add_argument('--download-models',action='store_true')
    parser.add_argument('--log',type=Path,help='New JSONL log, never overwrites an existing file')
    parser.add_argument('--udp',help='Optional same-host JSON output, e.g. 127.0.0.1:5006')
    args=parser.parse_args(argv)
    for key in ('width','height','camera_fps','max_age_ms','output_hz'):
        if not np.isfinite(getattr(args,key)) or getattr(args,key)<=0:
            parser.error(f'{key} must be positive and finite')
    if args.camera is not None and args.camera<0:
        parser.error('Camera index must be nonnegative')
    if not 0<args.min_visibility<=1:
        parser.error('Visibility threshold must be in (0,1]')
    if args.duration is not None and (not np.isfinite(args.duration) or args.duration<=0):
        parser.error('Duration must be positive and finite')
    if args.headless and args.duration is None and not (args.download_models or args.list_cameras):
        parser.error('Use --duration with --headless')
    if args.udp:
        try:
            host,port=args.udp.rsplit(':',1)
            if socket.gethostbyname(host)!='127.0.0.1' or not 1<=int(port)<=65535:
                raise ValueError()
            args.udp=('127.0.0.1',int(port))
        except (OSError,ValueError):
            parser.error('This prototype exports to 127.0.0.1:PORT on the same computer')
    return args


def envelope(session,sequence,state,reason,signal=None,observation=None,metrics=None,synthetic=False):
    return {'schema':'quadradex.teleop.v1','session_id':session,'sequence':sequence,
            'published_monotonic_ns':time.perf_counter_ns(),'clock':'host_perf_counter_ns',
            'state':state,'reason':reason,'synthetic_demo':synthetic,'hardware_enabled':False,
            'arm_accuracy':'not_independently_validated','control_signal':signal,
            'source_frame_id':observation.frame.index if observation else None,
            'arm_observation':observation.arm if observation else None,
            'hand_measurement':observation.hand if observation else None,'fit_metrics':metrics}


def run_demo(args,robot,simulation,output):
    generator=SignalGenerator(robot,args.max_age_ms)
    session=str(uuid.uuid4())
    start=previous=time.perf_counter()
    sequence=0
    print('SYNTHETIC DEMO: simulated arm + LEAP V1. No camera or physical actuators are used.',flush=True)
    while simulation.is_running():
        now=time.perf_counter()
        elapsed=now-start
        if args.duration is not None and elapsed>=args.duration:
            break
        desired=robot.home.copy()
        desired[0]=.30*np.sin(elapsed*.65)
        desired[1]+=.12*np.sin(elapsed*.45)
        desired[3]+=.15*np.sin(elapsed*.6)
        curl=.5*(1-np.cos(elapsed*1.2))
        for offset in (0,4,8):
            desired[7+offset:7+offset+4]=[.1*np.sin(elapsed),.85*curl,1.0*curl,.75*curl]
        desired[19:23]=[.6*curl,.5*curl,.6*curl,.45*curl]
        source=time.perf_counter_ns()
        signal=generator.generate(desired,source)
        simulation.apply(signal)
        simulation.step(min(.05,now-previous))
        previous=now
        sequence+=1
        output.emit(envelope(session,sequence,'SYNTHETIC_DEMO','Generated demo motion; not a measured operator pose',signal,synthetic=True))
        if output.log_error:
            raise RuntimeError(output.log_error)
        time.sleep(max(0,1/args.output_hz-(time.perf_counter()-now)))
    output.emit(envelope(session,sequence+1,'STOPPED','Demo finished',synthetic=True))


def overlay_arm(view,observation,side,age_ms,valid):
    frame=observation.frame
    h,w=frame.image.shape[:2]
    if observation.pose_image is not None and age_ms<=250:
        points=observation.pose_image[:,:2].copy()
        points[:,0]=1-points[:,0]
        points=np.rint(points*[w,h]).astype(int)
        chain=(12,14,16) if side=='right' else (11,13,15)
        color=(255,190,75) if valid else (80,170,240)
        for a,b in [(11,12),(11,23),(12,24),*zip(chain,chain[1:])]:
            cv2.line(view,tuple(points[a]),tuple(points[b]),color,3,cv2.LINE_AA)
        for i in chain:
            cv2.circle(view,tuple(points[i]),7,color,-1,cv2.LINE_AA)
        if observation.roi:
            x1,y1,x2,y2=observation.roi
            cv2.rectangle(view,(w-x2,y1),(w-x1,y2),(120,160,180),1)
    top=max(0,h-94)
    cv2.rectangle(view,(0,top),(w,h),(25,27,31),-1)
    lines=[f"ARM + HAND: {'fresh' if valid else 'targets blocked'} | age {age_ms:.0f} ms",
           f"Elbow: {observation.arm['elbow_flexion_deg']:.0f} deg | visibility {observation.arm['minimum_visibility']:.2f}" if observation.arm else 'Keep your shoulders, hips, elbow, wrist and fingers in view',
           'Simulation only | In the 3D window: drag to orbit, scroll to zoom, F to focus hand']
    for i,line in enumerate(lines):
        cv2.putText(view,line,(14,top+23+i*27),cv2.FONT_HERSHEY_SIMPLEX,.48,(225,230,235),1,cv2.LINE_AA)


def run_live(args,robot,simulation,output):
    camera=worker=None
    clicked=[]
    gate,capture=StreamGate(),CalibrationCapture()
    retarget,generator=Retargeter(robot),SignalGenerator(robot,args.max_age_ms)
    anchor=hand_calibration=opened=human_open=None
    arm_samples,hand_samples=[],[]
    note='Keep torso and whole arm visible. Hold a comfortable OPEN pose and press O.'
    last_result=last_solved=-1
    times=deque()
    metrics=None
    control_reason=''
    active_signal=None
    session=str(uuid.uuid4())
    sequence=0
    started=time.perf_counter()
    current_width=args.width

    def mouse(event,x,y,flags,userdata):
        if event==cv2.EVENT_LBUTTONDOWN:
            for key,_,left,right in BUTTONS:
                if left<=x-current_width<=right and 755<=y<=796:
                    clicked.append(ord(key))

    def open_camera():
        if args.camera is None and not args.headless:
            picked=choose_camera(LatestCamera,args)
            return picked if picked is not None else (None,None)
        choice=CameraChoice(args.camera if args.camera is not None else 0,args.backend,f'Camera {args.camera if args.camera is not None else 0}')
        return choice,LatestCamera(choice.index,args.width,args.height,args.camera_fps,choice.backend)

    try:
        choice,camera=open_camera()
        if camera is None:
            return
        worker=WholeArmVision(camera,args.side,args.swap_handedness,args.min_visibility)
        if not args.headless:
            cv2.namedWindow(WINDOW,cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(WINDOW,mouse)
        started=last_tick=last_publish=last_console=time.perf_counter()
        while simulation.is_running():
            now=time.perf_counter()
            if args.duration is not None and now-started>=args.duration:
                break
            if worker.error or camera.error:
                raise RuntimeError(worker.error or camera.error)
            observation=worker.get()
            if observation is None:
                if now-started>12:
                    raise RuntimeError('No completed body/hand inference; check camera and model setup')
                simulation.step(min(.04,now-last_tick))
                last_tick=now
                if not args.headless:
                    blank=np.full((480,900,3),(28,30,34),np.uint8)
                    cv2.putText(blank,'Starting arm + finger tracking...',(80,220),cv2.FONT_HERSHEY_SIMPLEX,.8,(225,225,225),1)
                    cv2.imshow(WINDOW,blank)
                    if cv2.waitKey(1)&255 in (27,ord('q')):
                        break
                time.sleep(.005)
                continue
            age_ms=(time.perf_counter_ns()-observation.frame.received_ns)/1e6
            new_result=observation.frame.index!=last_result
            valid=observation.arm is not None and observation.hand is not None and age_ms<=args.max_age_ms
            reason=observation.reason
            if valid and anchor is not None:
                try:
                    anchor.validate_measurement(observation.arm)
                except ValueError as error:
                    valid,reason=False,str(error)
            if age_ms>args.max_age_ms:
                reason='Preview only: synchronized arm/hand result is too old for control'
            gate.observe(valid,now)
            sample_hand=observation.hand if valid else None
            count_before=len(capture.samples)
            completed=capture.update(now,sample_hand,new_result,observation.frame.received_ns/1e9,reason)
            if capture.stage=='open' or (completed and completed[0]=='open'):
                if new_result and valid and len(capture.samples)>count_before:
                    arm_samples.append(observation.arm)
                    hand_samples.append(observation.hand)
                elif new_result and valid and len(capture.samples)==1 and count_before>=1:
                    arm_samples[:],hand_samples[:]=[observation.arm],[observation.hand]
            if completed:
                stage,pose=completed
                try:
                    if stage=='open':
                        anchor=ArmAnchor.capture(arm_samples,hand_samples)
                        human_open=hand_samples[-1]
                        opened=pose
                        retarget.calibrate(anchor,human_open)
                        note='Arm + OPEN hand captured. Keep arm visible, close fingers and press C.'
                    else:
                        hand_calibration=make_calibration(args.side.title(),opened,pose)
                        note='Arm + hand calibration complete. Hold still, then press Space to simulate.'
                except ValueError as error:
                    if stage=='open':
                        anchor=opened=human_open=None
                    note=str(error)+f". Press {'O' if stage=='open' else 'C'} to retry."
                    capture.message,capture.phase=note,'error'
            elif capture.phase=='cancelled':
                note=capture.message
            if new_result:
                last_result=observation.frame.index
                if valid:
                    times.append(now)
            while times and times[0]<now-2:
                times.popleft()
            if gate.active and valid and observation.frame.index!=last_solved:
                last_solved=observation.frame.index
                try:
                    desired,metrics=retarget.solve(observation.arm,observation.hand)
                    active_signal=generator.generate(desired,observation.frame.received_ns)
                    simulation.apply(active_signal)
                    reason='Following coordinated arm + LEAP control signal'
                except ValueError as error:
                    control_reason=reason=str(error)
                    active_signal=None
                    simulation.hold(reason)
                    # Infeasible targets hold the pair; stale sensing requires deliberate recovery.
                    if (time.perf_counter_ns()-observation.frame.received_ns)/1e6>args.max_age_ms:
                        gate.observe(False,time.perf_counter())
            if gate.active and valid and active_signal is None:
                reason=control_reason or reason
            if not gate.active or not valid:
                active_signal=None
                simulation.hold(reason if not valid else 'Target stream stopped')
            simulation.step(min(.05,now-last_tick))
            last_tick=now
            state=gate.state(hand_calibration is not None and anchor is not None,capture.active)
            if gate.active and active_signal is None:
                state='LIMIT_ACTIVE' if valid else 'TRACKING_LOST'
            key=-1
            if not args.headless:
                current_width=observation.frame.image.shape[1]
                progress=capture.status(now)
                ui={'title':'QUADRADEX / ARM + HAND','camera_name':choice.name,'hand_label':args.side.title(),
                    'tracking_valid':valid,'calibrated':hand_calibration is not None and anchor is not None,
                    'opened':opened is not None,'calibration_progress':progress,
                    'finger_ranges':finger_ranges(opened,observation.hand) if observation.hand else {}}
                display_metrics={'valid_result_hz':len(times)/min(2,max(.01,now-started)),'result_age_ms':age_ms,
                                 'inference_ms':(observation.completed_ns-observation.frame.received_ns)/1e6}
                view=draw(observation.frame,observation.hand_image if age_ms<=250 else None,observation.hand if age_ms<=250 else None,
                          state,reason,display_metrics,note,True,ui)
                physical_side=('left' if args.side=='right' else 'right') if args.swap_handedness else args.side
                overlay_arm(view,observation,physical_side,age_ms,valid)
                cv2.imshow(WINDOW,view)
                key=cv2.waitKey(1)&255
                if clicked:
                    key=clicked.pop(0)
                if cv2.getWindowProperty(WINDOW,cv2.WND_PROP_VISIBLE)<1:
                    break
            if key in (27,ord('q')):
                break
            if key==ord('o'):
                gate.stop();simulation.hold('Calibrating')
                anchor=hand_calibration=opened=human_open=None
                arm_samples,hand_samples=[],[]
                capture.start('open',time.perf_counter())
                note='Hold your arm still, fingers open. Capturing the arm neutral pose too.'
            elif key==ord('c'):
                if opened is None:
                    note='Capture the OPEN hand + neutral arm first: press O.'
                else:
                    gate.stop();simulation.hold('Calibrating')
                    hand_calibration=None
                    capture.start('closed',time.perf_counter())
            elif key==ord(' ') and not capture.active:
                if gate.active:
                    gate.stop();simulation.hold('Stopped by operator')
                elif gate.start(hand_calibration is not None and anchor is not None,valid,time.perf_counter()):
                    generator.reconcile()
                    last_solved=-1
                    note='Simulation follows generated targets. S stops both arm and hand.'
                else:
                    note='Need O/C calibration and 0.5 seconds of fresh, visible arm + fingers.'
            elif key==ord('s'):
                gate.stop();capture.cancel('Stopped. O/C recalibrates; Space resumes when ready.')
                simulation.hold('Stopped by operator')
            elif key in (ord('r'),ord('h'),ord('m'),ord('v')):
                gate=StreamGate();capture=CalibrationCapture()
                simulation.hold('Reconfiguring - targets inhibited')
                anchor=hand_calibration=opened=human_open=None
                active_signal=None
                if key in (ord('h'),ord('m'),ord('v')):
                    sequence+=1
                    output.emit(envelope(session,sequence,'STOPPED','Camera/side changed'))
                    worker.close();worker=None
                    if key==ord('h'):
                        args.side='left' if args.side=='right' else 'right'
                    if key==ord('m'):
                        args.swap_handedness=not args.swap_handedness
                    if key==ord('v'):
                        camera.close();camera=None
                        picked=choose_camera(LatestCamera,args)
                        if picked is None:
                            break
                        choice,camera=picked
                    worker=WholeArmVision(camera,args.side,args.swap_handedness,args.min_visibility)
                    session,sequence=str(uuid.uuid4()),0
                last_result=last_solved=-1
                times.clear()
                note='Configuration changed. Hold a neutral arm + open hand and press O.'
            # Recheck immediately before publication; the viewer/IK may have consumed time.
            if not gate.active or capture.active:
                active_signal=None
            if active_signal is not None and time.perf_counter_ns()>active_signal['expires_monotonic_ns']:
                active_signal=None
                gate.observe(False,time.perf_counter())
                simulation.hold('Source expired before publication')
                state='TRACKING_LOST'
            elif not gate.active:
                state=gate.state(hand_calibration is not None and anchor is not None,capture.active)
            if time.perf_counter()-last_publish>=1/args.output_hz:
                sequence+=1
                packet=envelope(session,sequence,state,reason,active_signal,observation,metrics)
                packet['calibration_progress']=capture.status(time.perf_counter())
                packet['camera']={'name':choice.name,'index':choice.index,'backend':choice.backend}
                output.emit(packet)
                last_publish=time.perf_counter()
            if output.log_error:
                raise RuntimeError(output.log_error)
            if args.headless and now-last_console>=1:
                print(f'{state}: {reason}; fresh paired results {len(times)/min(2,max(.01,now-started)):.1f}/s',flush=True)
                last_console=now
            time.sleep(.002)
    finally:
        output.emit(envelope(session,sequence+1,'STOPPED','Producer stopped'))
        simulation.hold('Producer stopped')
        if worker is not None:
            worker.close()
        if camera is not None:
            camera.close()
        if not args.headless:
            cv2.destroyAllWindows()


def main(argv=None):
    args=parse_args(argv)
    if args.list_cameras:
        for device in discover_cameras(args.backend):
            print(f'{device.index}: {device.name} ({device.backend})')
        return 0
    if args.download_models:
        print(ensure_models())
        return 0
    robot=RobotModel(args.arm_config)
    output=Output(args.udp,args.log)
    simulation=None
    try:
        simulation=Simulation(robot,args.headless)
        (run_demo if args.demo else run_live)(args,robot,simulation,output)
        if args.snapshot:
            args.snapshot.parent.mkdir(parents=True,exist_ok=True)
            robot.snapshot(args.snapshot)
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as error:
        print(f'Error: {error}',file=sys.stderr,flush=True)
        return 1
    finally:
        if simulation is not None:
            simulation.close()
        output.close()


if __name__=='__main__':
    raise SystemExit(main())
