import numpy as np
import socket
from struct import unpack

HOST = '127.0.0.1'
PORT = 65430

from bokeh.layouts import row, column, gridplot
from bokeh.models import ColumnDataSource, Slider, Select
from bokeh.themes import built_in_themes
from bokeh.models.widgets import PreText
from bokeh.plotting import curdoc, figure
from bokeh.driving import count

#curdoc().theme = 'dark_minimal'

source = ColumnDataSource(dict(
    time=[], inst_val=[], cum_val=[], 
    count=[], pnl_color=[], rnd_color=[]
))

p = figure(plot_height=500, 
           tools="xpan,xwheel_zoom,xbox_zoom,reset", 
           y_axis_location="left", title="Cumulative Performance")
p.x_range.follow = "end"
p.x_range.follow_interval = 5000
p.x_range.range_padding = 0
p.line(x='time', y='cum_val', 
       line_width=1, color='grey', 
       source=source)
p.circle(x='time', y='cum_val', 
         color='pnl_color', source=source, size=5)


p2 = figure(plot_height=200, x_range=p.x_range,
            tools="xpan,xwheel_zoom,xbox_zoom,reset", 
            y_axis_location="left", 
            title="Instantaneous Performance")
p2.line(x='time', y='inst_val', 
        line_width=1, color='grey', 
        source=source)
p2.circle(x='time', y='inst_val', 
          color='rnd_color', size=5, 
          source=source)


p3 = figure(plot_height=200, x_range=p.x_range, 
            tools="xpan,xwheel_zoom,xbox_zoom,reset", 
            y_axis_location="left", 
            title="Order Length")
p3.line(x='time', y='count', 
        color='green', source=source)

count_msg = PreText(text="",width=500, height=100)
backtest_msg = PreText(text="",width=500, height=100)

@count()
def update(t):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        conn, addr = s.accept()
        with conn:
            data = conn.recv(1024).decode("utf-8") 
            vals = data.split(',')
    
    count = float(vals[3])
    count_msg.text = f"Current order length: {str(int(count))}"
    
    if (vals[0] != 'NA') and (vals[1] != 'NA') and (vals[2] != 'NA'):
        order_num, instaneous, cumulative = int(float(vals[0])), float(vals[1]), float(vals[2])
        rnd_order = float(vals[4])
        
        if rnd_order == 0.0:
            rnd_col = 'blue'
        else:
            rnd_col = 'orange'

        if instaneous < 0:
            pnl_col = 'red'
        elif instaneous == 0:
            pnl_col = 'blue'
        else:
            pnl_col = 'green'

        new_data = dict(
            time=[order_num],
            inst_val=[instaneous], 
            cum_val=[cumulative],
            count = [count],
            pnl_color=[pnl_col],
            rnd_color=[rnd_col]
        )
        source.stream(new_data, 300)

curdoc().add_root(column(gridplot([[p], [p2], [p3], [count_msg]], toolbar_location="left", plot_width=1800)))
curdoc().add_periodic_callback(update, 25)
curdoc().title = "Performance"