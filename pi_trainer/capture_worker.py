"""Optional OpenCV camera/video capture worker; never starts without an explicit job."""
import argparse
import json
import time
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--seconds',type=float,required=True);p.add_argument('--fps',type=float,required=True);p.add_argument('--camera',type=int,default=0);p.add_argument('--video');a=p.parse_args()
    try:import cv2
    except ImportError:raise RuntimeError('Install opencv-python in the selected worker Python environment for capture')
    capture=cv2.VideoCapture(a.video if a.video else a.camera)
    if not capture.isOpened():raise RuntimeError('Camera/video could not be opened')
    folder=Path(a.output);folder.mkdir(parents=True,exist_ok=True);start=time.monotonic();next_frame=0.;count=0
    try:
        while count<1000:
            ok,frame=capture.read()
            if not ok:break
            elapsed=capture.get(cv2.CAP_PROP_POS_MSEC)/1000 if a.video else time.monotonic()-start
            if elapsed>a.seconds:break
            if elapsed>=next_frame:
                if not cv2.imwrite(str(folder/f'frame-{count:06d}.png'),frame):raise RuntimeError('Frame write failed')
                count+=1;next_frame+=1/a.fps
            if not a.video:time.sleep(.005)
    finally:capture.release()
    if not count:raise RuntimeError('No frames captured')
    (folder/'capture.json').write_text(json.dumps({'frames':count,'requested_fps':a.fps,'duration_s':a.seconds,'source':a.video or f'camera:{a.camera}','labels':'unreviewed','group':folder.name}))
if __name__=='__main__':main()
