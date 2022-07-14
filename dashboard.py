# Import the Modules
import dash
from dash import html, dcc
from flask import Flask
import dash_bootstrap_components as dbc
from matplotlib.transforms import Bbox
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import cv2
import sys
import time
import torch
import numpy as np
import pandas as pd
from collections import deque
import torch
import cv2
import time
import sys

from flask import Flask,Response
from dash import Dash,html,dcc
import dash_bootstrap_components as dbc
from mainTracker import Tracker,vis_track,draw_lines,lines

server=Flask(__name__)

app=Dash(__name__,server=server,external_stylesheets=[dbc.themes.BOOTSTRAP])

tracker=Tracker(model="yolox-s",ckpt="weights/yolox_s.pth",filter_classes=None)

Main=deque(maxlen=1000)
def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

class VideoCamera(object):
    def __init__(self):
        global res;
        self.video = cv2.VideoCapture(sys.argv[1])
        res = f"{int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))} x {int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))}" 

    def __del__(self):
        self.video.release()
        cv2.destroyAllWindows()

    def get_frame(self):
        global fps;
        success, image = self.video.read()
        if success:
            t1 = time_synchronized()
            #fps  = f"{int((1./(time_synchronized()-t1)))}"
            image=draw_lines(lines,image)
            image,bbox,data=tracker.update(image,logger_=False)
            image=vis_track(image,bbox)
            Main.extend(data)
            ret, jpeg = cv2.imencode('.jpg', image)
            return jpeg.tobytes()
        else:
            return "Video is Completed !!!"

def gen(camera):
    fps = 0.0
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@server.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



app.layout=html.Div([
                         html.Img(src="/video_feed")

])

if __name__=="__main__":
    app.run_server(port=8050,debug=True)

