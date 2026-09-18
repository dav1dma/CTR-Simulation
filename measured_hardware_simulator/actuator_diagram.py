"""Camera-independent screen-space side view of measured actuator positions."""
import numpy as np
from vispy import scene

class ActuatorDiagram:
    def __init__(self, viewer):
        self.viewer=viewer
        self.panel=scene.widgets.Widget(parent=viewer.canvas.scene,pos=(12,12),size=(620,185),
             bgcolor=(.97,.98,1,.97),border_color='#b8c5d5',border_width=1)
        self.panel.order=10000
        self.title=scene.visuals.Text('',parent=self.panel,font_size=9,color='#17334e',anchor_x='left',pos=(12,18))
        self.axis=scene.visuals.Line(parent=self.panel,color='#9caabd',width=1)
        self.plate=scene.visuals.Line(parent=self.panel,color='#334155',width=4)
        self.plate_label=scene.visuals.Text('Plate',parent=self.panel,font_size=8,color='#334155')
        self.boxes=[];self.labels=[]
        for color in ('#2166d1','#22966c','#ed8936'):
            self.boxes.append(scene.visuals.Rectangle(center=(0,0),width=60,height=34,color=color,
                    border_color='#334155',border_width=2,parent=self.panel))
            self.labels.append(scene.visuals.Text('',parent=self.panel,font_size=8,color='white'))
        self.gaps=[scene.visuals.Text('',parent=self.panel,font_size=8,color='#334155') for _ in range(2)]
        self.info=scene.visuals.Text('',parent=self.panel,font_size=8,color='#334155',anchor_x='left',pos=(12,151))
        self.legend=scene.visuals.Text('Black: selected | Purple: follower | Forward: right | C: control mode',
                parent=self.panel,font_size=8,color='#334155',anchor_x='left',pos=(12,172))
        viewer.canvas.events.resize.connect(self.update)
        self.update()

    def update(self, event=None):
        v=self.viewer
        width=max(410,min(690,v.canvas.size[0]-(384 if v.ui_visible else 24)))
        self.panel.size=(width,185)
        scale=(width-60)/360
        x=lambda coordinate: 20+(coordinate+360)*scale
        d=v.profile.total_length_mm-v.deployment*1000
        self.axis.set_data(pos=np.array([[x(-360),88],[x(0),88]]))
        self.plate.set_data(pos=np.array([[x(0),55],[x(0),113]]))
        self.plate_label.pos=(x(0)-3,43)
        followers=getattr(v,'following_actuators',())
        for i,(box,label) in enumerate(zip(self.boxes,self.labels)):
            mid=x(-d[i]-71.5/2)
            box.center=(mid,88);box.width=71.5*scale
            box.border_color='#9c27b0' if i in followers else ('#111827' if i==v.selected_tube else '#94a3b8')
            label.pos=(mid,88);label.text=('REAR / INNER','MIDDLE','FRONT / OUTER')[i]
        gap=d[:-1]-d[1:]-71.5
        for i,label in enumerate(self.gaps):
            label.pos=(x((-d[i]-d[i+1]-71.5)/2),123)
            label.text=f'{gap[i]:.1f} mm'
            label.color='#c05b00' if min(abs(gap[i]-8.5),abs(gap[i]-v.limits.max_gap_mm))<.1 else '#334155'
        cartesian=getattr(v,'control_mode','JOINT')=='tip'
        self.title.text='ACTUATOR POSITIONS — schematic | '+('Cartesian IK' if cartesian else v.joint_motion_mode.title())
        travel=v.profile.carriage_displacement_mm(v.deployment)
        self.info.text='Travel from rear stop:  '+ '  |  '.join(f'{n} {a:.1f}/100 mm' for n,a in zip(('I','M','O'),travel))
