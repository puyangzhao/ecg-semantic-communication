"""Render a schematic method GIF (requires Pillow). No patient signals are used."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math
HERE=Path(__file__).resolve().parent
FONT=HERE.parent/'code/fonts/LiberationSans-Regular.ttf'
BOLD=HERE.parent/'code/fonts/LiberationSans-Bold.ttf'
navy='#213c59';purple='#76629c';amber='#c58b36';gray='#8392a1'
def font(n,b=False):return ImageFont.truetype(str(BOLD if b else FONT),n)
def label(d,xy,s,n=18,color=navy,b=False):d.text(xy,s,font=font(n,b),fill=color,anchor='mm')
def box(d,rect,title,sub,color):
 d.rounded_rectangle(rect,14,fill='white',outline=color,width=2)
 x=(rect[0]+rect[2])/2;y=(rect[1]+rect[3])/2
 label(d,(x,y-10),title,19,color,True);label(d,(x,y+18),sub,14,gray)
def line(d,pts,color):
 d.line(pts,fill=color,width=2);x,y=pts[-1];d.polygon([(x,y),(x-8,y-4),(x-8,y+4)],fill=color)
frames=[]
for f in range(48):
 im=Image.new('RGB',(1120,410),'#f8fafc');d=ImageDraw.Draw(im)
 label(d,(560,34),'Selecting and transmitting ECG information',28,navy,True)
 label(d,(560,66),'All 12 leads acquired · Shared encoder · Five diagnostic outputs',16,gray)
 for k in range(12):
  y=130+k*15
  pts=[]
  for x in range(35,150):
   u=(x-35)/115;v=math.exp(-((u-.43)/.025)**2)*11-math.exp(-((u-.47)/.027)**2)*5
   pts.append((x,y-v))
  d.line(pts,fill=navy,width=1)
 label(d,(92,330),'12-lead ECG',17,navy,True)
 box(d,(185,180,365,270),'Shared 1-D ResNet','12 × 32 embeddings',navy)
 box(d,(440,115,665,190),'Lead selection','K = 1, 3 or 6',purple)
 box(d,(735,115,1065,190),'Pool + classify','Clean link',purple)
 box(d,(440,265,665,340),'Alternative codes','Repetition · SSCC · JSCC',amber)
 box(d,(715,265,855,340),'Channel','Noise / fading',amber)
 box(d,(905,265,1065,340),'Receiver','Decode + classify',amber)
 paths=[([(150,225),(185,225)],navy), ([(365,225),(400,225),(400,152),(440,152)],purple), ([(665,152),(735,152)],purple), ([(365,225),(400,225),(400,302),(440,302)],amber), ([(665,302),(715,302)],amber), ([(855,302),(905,302)],amber)]
 for pts,c in paths:
  line(d,pts,c)
  lengths=[math.dist(a,b) for a,b in zip(pts,pts[1:])];v=(f/48)*sum(lengths)
  for a,b,l in zip(pts,pts[1:],lengths):
   if v<=l:
    t=v/l;x=a[0]+(b[0]-a[0])*t;y=a[1]+(b[1]-a[1])*t;d.ellipse((x-4,y-4,x+4,y+4),fill=c);break
   v-=l
 label(d,(560,382),'Coding compares one selected embedding with joint coding of all 12 embeddings at the same budget.',15,gray)
 frames.append(im)
frames[0].save(HERE/'overview.gif',save_all=True,append_images=frames[1:],duration=85,loop=0,optimize=True)
frames[0].save(HERE/'overview.png')
print('Rendered',HERE/'overview.gif')
