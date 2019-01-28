import numpy as np
import socket
from struct import unpack

from bokeh.layouts import column, gridplot
from bokeh.models import ColumnDataSource, Button
from bokeh.events import ButtonClick
from bokeh.plotting import curdoc, figure
from bokeh.driving import count


class Visual():
    def __init__(self):
        self.HOST = '127.0.0.1'
        self.PORT = 65430
        self.PACKING = '?ifi'
        self.agents = None
        self.data = []

source = ColumnDataSource(dict(
    time=[], 
    l1=[], l2=[], l3=[],
    l4=[], l5=[], l6=[],
    l7=[], l8=[], l9=[],
    l10=[], l11=[], l12=[]
))

p = figure(plot_height=500, 
           tools="xpan,xwheel_zoom,xbox_zoom,reset", 
           y_axis_location="left", title="Multi Performance")

p.x_range.follow = "end"
p.x_range.follow_interval = 5000
p.x_range.range_padding = 0

p.line(x='time', y='l1', 
       line_width=1, color='green', 
       source=source)
p.line(x='time', y='l2', 
       line_width=1, color='blue', 
       source=source)
p.line(x='time', y='l3', 
       line_width=1, color='orange', 
       source=source)
p.line(x='time', y='l4', 
       line_width=1, color='green', 
       source=source)
p.line(x='time', y='l5', 
       line_width=1, color='blue', 
       source=source)
p.line(x='time', y='l6', 
       line_width=1, color='orange', 
       source=source)
p.line(x='time', y='l7', 
       line_width=1, color='green', 
       source=source)
p.line(x='time', y='l8', 
       line_width=1, color='blue', 
       source=source)
p.line(x='time', y='l9', 
       line_width=1, color='orange', 
       source=source)
p.line(x='time', y='l10', 
       line_width=1, color='green', 
       source=source)
p.line(x='time', y='l11', 
       line_width=1, color='blue', 
       source=source)
p.line(x='time', y='l12', 
       line_width=1, color='orange', 
       source=source)


b = Button()

def callback(event):
    print('Python:Click')
    curdoc().remove_periodic_callback()
    print("Done")

b.on_event(ButtonClick, callback)

visual = Visual()


@count()
def update(t):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((visual.HOST, visual.PORT))
    s.listen()
    conn, addr = s.accept()
    with conn:
        data = conn.recv(1024)
        data = unpack(visual.PACKING, data)
        if data[0]:
            visual.agents = data[1]
            visual.data = [0] * visual.agents
        else:
            target_agent = data[1]
            agent_value = data[2]
            agent_time = data[3]
            visual.data[target_agent] = agent_value
            if target_agent == visual.agents - 1:

                new_data = dict(
                    time=[agent_time],
                    l1=[visual.data[0]], 
                    l2=[visual.data[1]], 
                    l3=[visual.data[2]],
                    l4=[visual.data[3]], 
                    l5=[visual.data[4]], 
                    l6=[visual.data[5]],
                    l7=[visual.data[6]], 
                    l8=[visual.data[7]], 
                    l9=[visual.data[8]],
                    l10=[visual.data[9]], 
                    l11=[visual.data[10]], 
                    l12=[visual.data[11]]
                )
                source.stream(new_data, 300)
        s.close()
                

curdoc().add_root(column(gridplot([[p],[b]], toolbar_location="left")))
curdoc().add_periodic_callback(update, 1000)
curdoc().title = "Multi Visual"


