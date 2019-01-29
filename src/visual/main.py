import numpy as np
import socket
from struct import unpack

from bokeh.layouts import column, gridplot
from bokeh.models import ColumnDataSource, Button
from bokeh.events import ButtonClick
from bokeh.plotting import curdoc, figure
from bokeh.driving import count

import time

class Visual():
    def __init__(self):
        self.HOST = '127.0.0.1'
        self.PORT = 65430
        self.PACKING = '?ifi'
        self.agents = None
        self.current_data = None
        self.previous_time = 1

visual = Visual()

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((visual.HOST, visual.PORT))
s.listen()
conn, addr = s.accept()
with conn:
    data = conn.recv(1024)
    data = unpack(visual.PACKING, data)
    print("Data 1:",data)
    if data[0]:
        visual.agents = data[1]
        visual.current_data = [0] * visual.agents
s.close()


source_data = dict(time=[])
print("Vis A:",visual.agents)
for i in range(visual.agents):
    source_data[f'l{i}'] = []

source = ColumnDataSource(data=source_data)

p = figure(plot_height=500, 
           tools="xpan,xwheel_zoom,xbox_zoom,reset", 
           y_axis_location="left", title="Multi Performance")

p.x_range.follow = "end"
p.x_range.follow_interval = 5000
p.x_range.range_padding = 0

for i in range(visual.agents):
    p.line(x='time', y=f'l{i}', source=source)


@count()
def update(t):
    global source
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((visual.HOST, visual.PORT))
    s.listen()
    conn, addr = s.accept()
    with conn:
        data = conn.recv(1024)
        data = unpack(visual.PACKING, data)
        print("Data 2:",data)
        if data[0]:
            print("Warning")
            pass
        else:
            target_agent = data[1]
            agent_value = data[2]
            agent_time = data[3]
#             print(agent_time, target_agent, agent_value)
            if visual.previous_time != agent_time:
                ## send data, previous time step completed
#                 print("Caught")
                new_data = dict(
                    time = [visual.previous_time]
                )
                for idx,i in enumerate(visual.current_data):
                    new_data[f'l{idx}'] = [i]
#                 print(new_data)
                source.stream(new_data, 300)
                visual.previous_time = agent_time
#                 print(visual.previous_time, agent_time)
            visual.current_data[target_agent] = agent_value

        s.close()
#     print("----")

curdoc().add_root(column(gridplot([[p]], toolbar_location="left")))
curdoc().add_periodic_callback(update, 10)
curdoc().title = "Multi Visual"























# import numpy as np
# import socket
# from struct import unpack

# from bokeh.layouts import column, gridplot
# from bokeh.models import ColumnDataSource, Button
# from bokeh.events import ButtonClick
# from bokeh.plotting import curdoc, figure
# from bokeh.driving import count


# class Visual():
#     def __init__(self):
#         self.HOST = '127.0.0.1'
#         self.PORT = 65430
#         self.PACKING = '?ifi'
#         self.agents = None
#         self.current_data = None

# source = ColumnDataSource(dict(
#                                xs=[], 
#                                ys=[]
#                               ))
        
        

# def callback(event):
#     print('Python:Click')
#     curdoc().remove_periodic_callback()
#     print("Done")


# visual = Visual()

# p = figure(plot_height=500, 
#            tools="xpan,xwheel_zoom,xbox_zoom,reset", 
#            y_axis_location="left", title="Multi Performance")

# p.x_range.follow = "end"
# p.x_range.follow_interval = 5000
# p.x_range.range_padding = 0

# p.multi_line(xs='xs', ys='ys', source=source)

# b = Button()
# b.on_event(ButtonClick, callback)


# @count()
# def update(t):
#     global source
#     s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     s.bind((visual.HOST, visual.PORT))
#     s.listen()
#     conn, addr = s.accept()
#     with conn:
#         data = conn.recv(1024)
#         data = unpack(visual.PACKING, data)
#         if data[0]:
#             visual.agents = data[1]
#             visual.current_data = [0] * visual.agents
#             source.data = dict(
#                 xs=[[] for i in range(visual.agents)], 
#                 ys=[[] for i in range(visual.agents)]
#             )
#         else:
#             target_agent = data[1]
#             agent_value = data[2]
#             agent_time = data[3]
#             visual.current_data[target_agent] = agent_value
#             if target_agent == visual.agents - 1:
#                 new_data = dict()
                
#                 new_data['xs'] = [agent_time] * visual.agents
#                 new_data['ys'] = visual.current_data
#                 print(new_data)
#                 source.stream(new_data, 300)
#                 print(source.data)
#                 print("====")
#         s.close()
                

# curdoc().add_root(column(gridplot([[p],[b]], toolbar_location="left")))
# curdoc().add_periodic_callback(update, 1000)
# curdoc().title = "Multi Visual"


