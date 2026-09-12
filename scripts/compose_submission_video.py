"""Assemble labeled physical rollouts and saved synthetic speech evidence.

Uses original recordings only. Faster playback is labeled, simulation timestamps
stay visible, and the audio is explicitly identified as synthetic test speech.
"""
import argparse,json,os,sys,wave
from fractions import Fraction
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import av,numpy as np
from PIL import Image,ImageDraw,ImageFont
from simulation_lab.storage import require_space,GIB

def run(a):
    out=Path(a.output);source=Path(a.source)
    if out.exists():raise FileExistsError(out)
    require_space(out,GIB);out.parent.mkdir(parents=True,exist_ok=True)
    fontfile=Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts/arial.ttf'
    font=ImageFont.truetype(str(fontfile),32);large=ImageFont.truetype(str(fontfile),50)
    container=av.open(str(out),'w',options={'movflags':'+faststart'});stream=container.add_stream('libx264',rate=20)
    stream.width=1280;stream.height=720;stream.pix_fmt='yuv420p';stream.options={'crf':'20','preset':'fast'}
    audio_stream=container.add_stream('aac',rate=44100);audio_stream.layout='mono'
    frame_count=0;chapters=[];audio_cues=[]
    def emit(image):
        nonlocal frame_count
        if frame_count%400==0:require_space(out,128*1024**2)
        frame=av.VideoFrame.from_ndarray(np.asarray(image),format='rgb24');frame.pts=frame_count;frame.time_base=Fraction(1,20)
        for packet in stream.encode(frame):container.mux(packet)
        frame_count+=1
    def card(title,lines,seconds=5,audio=None):
        chapters.append({'title':title,'start_s':frame_count/20,'duration_s':seconds,'type':'explanatory_card'})
        if audio:audio_cues.append((frame_count/20+.8,source/audio))
        im=Image.new('RGB',(1280,720),(13,28,43));draw=ImageDraw.Draw(im)
        draw.text((70,95),title,font=large,fill=(102,228,213))
        for i,line in enumerate(lines):draw.text((70,240+i*72),line,font=font,fill=(232,241,246))
        for _ in range(int(seconds*20)):emit(im)
    def clip(name,speed,label):
        first=frame_count;chapters.append({'title':label,'start_s':first/20,'playback_speed':speed,'source':name,'type':'recorded_physics'})
        with av.open(str(source/name)) as recording:
            for i,frame in enumerate(recording.decode(video=0)):
                if i%speed:continue
                im=frame.to_image().convert('RGB');draw=ImageDraw.Draw(im)
                draw.rectangle((0,0,1280,42),fill=(13,28,43));draw.text((15,2),label,font=font,fill=(232,241,246))
                small=ImageFont.truetype(str(fontfile),18)
                for x,camera in [(8,'Opposite side'),(648,'Overhead')]:
                    draw.rectangle((x,44,x+140,69),fill=(13,28,43));draw.text((x+4,46),camera,font=small,fill='white')
                emit(im)
        chapters[-1]['duration_s']=(frame_count-first)/20
        print(json.dumps(chapters[-1]),flush=True)
    try:
        card('Talos: dinner-table robotics',['Two SO-101 arms in MuJoCo','Speechmatics voice input and typed instructions','Programmed skills + an OpenVINO learned bottle baseline'],6)
        card('Speechmatics test: “Set the table”',['Audio: synthetic English test speech','Saved API transcript: “Set the table”','This real transcript starts six physical skills.'],5,'synthetic-table-english.wav')
        clip('voice-set-table-720p.mp4',2,'Programmed physical skills | 2x playback | exact-state IK')
        card('“Pass the bottle to the right arm”',['Typed command; programmed table-supported relay','Left arm releases and parks before the right arm regrips.','Both lifts and the final placement pass contact checks.'],6)
        clip('relay-bottle-720p.mp4',2,'Programmed relay | 2x playback | table release and regrasp')
        card('Speechmatics test: “Place the bottle”',['Audio: synthetic English test speech','Saved API transcript: “Place. The bottle.”','Normalized instruction starts the learned controller.'],5,'synthetic-bottle-english.wav')
        clip('visual-bottle-720p.mp4',1,'Learned upright bottle | 1x | OpenVINO + initial RGB localization')
        card('Intel OpenVINO FP32',['Network-only median inference: Intel CPU 0.157 ms','Intel UHD iGPU 0.688 ms; numerical parity checked','Camera rendering currently uses NVIDIA.'],8)
        card('Measured scope',['Revised upright model: 10/10 new task-preset scene seeds','Small bottle jitter; wider-position development: 1/3','Original model: 0/10 broader seeds; failures retained','github.com/jannissio/talos-ai-infra'],10)
        for packet in stream.encode():container.mux(packet)
        pcm=np.zeros(round(frame_count/20*44100),dtype='float32')
        for start,file in audio_cues:
            with wave.open(str(file),'rb') as w:
                if w.getnchannels()!=1 or w.getsampwidth()!=2:raise ValueError('Expected mono PCM16 source audio')
                rate=w.getframerate();x=np.frombuffer(w.readframes(w.getnframes()),dtype='<i2').astype('float32')/32768
            samples=np.interp(np.arange(round(len(x)/rate*44100))*rate/44100,np.arange(len(x)),x).astype('float32')
            begin=round(start*44100);pcm[begin:begin+len(samples)]=samples
        for offset in range(0,len(pcm),1024):
            f=av.AudioFrame.from_ndarray(pcm[None,offset:offset+1024],format='fltp',layout='mono');f.sample_rate=44100;f.pts=offset;f.time_base=Fraction(1,44100)
            for packet in audio_stream.encode(f):container.mux(packet)
        for packet in audio_stream.encode():container.mux(packet)
    finally:container.close()
    report={'resolution':[1280,720],'fps':20,'duration_s':frame_count/20,'bytes':out.stat().st_size,'chapters':chapters,
            'synthetic_speech_only':True,'physical_footage':'Recorded authoritative MuJoCo states, rendered offline; speeds labeled; no invented movements'}
    if report['duration_s']>300 or report['bytes']>300_000_000:raise ValueError('Submission video exceeds upload limits')
    out.with_suffix('.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',default='.run/submission-work');p.add_argument('--output',required=True);run(p.parse_args())
