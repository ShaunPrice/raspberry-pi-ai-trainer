"""Build normalized display captions from actual ElevenLabs word timestamps.

Writes a reviewable candidate, never the released SRT by default. All chapter
alignments and an ElevenLabs-marked manifest are required before any write.
"""
from __future__ import annotations
import argparse
import difflib
import hashlib
import json
import math
from pathlib import Path
import re
import textwrap

BASE=Path(__file__).resolve().parent
# Each display phrase maps to the exact spoken phrase's measured word span.
# These are typography transformations, not phonetic timing estimates.
PHRASES={
 'Hailo eight L':'Hailo-8L','Hailo eight':'Hailo-8','Hailo ten H':'Hailo-10H',
 'ten H':'Hailo-10H','x eighty six sixty four':'x86-64',
 'The tested eight':'The tested Hailo-8','eight L':'Hailo-8L',
 'Dataflow Compiler three point thirty four':'DFC 3.34','five point four':'DFC 5.4',
 'Raspberry Pi five with eight gigabytes of memory':'Raspberry Pi 5 (8GB)',
 'the thirteen TOPS eight L':'the 13 TOPS Hailo-8L',
 'and twenty six TOPS eight AI HAT Plus boards':'and 26 TOPS Hailo-8 AI HAT+ boards',
 'Transformers P E F T':'Transformers/PEFT','Gen AI':'GenAI','M C P':'MCP',
}
PHRASE_PATTERN=re.compile(r'\b(?:'+ '|'.join(re.escape(s) for s in sorted(PHRASES,key=len,reverse=True))+r')\b[.,;:!?]?')

def canonical(value):return re.sub(r'[^a-z0-9]','',value.lower())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def join(parts):return re.sub(r'\s+([.,;:!?])',r'\1',' '.join(parts))

def atoms(word):
    key=canonical(word)
    aliases={'halo':'hailo','colour':'color','organise':'organize','optimise':'optimize','customised':'customized','optimised':'optimized','laura':'lora','halos':'hailos','pie':'pi'}
    if word.startswith('.34'):return ['point','thirty','four']
    if word.startswith('.4'):return ['point','four']
    expansions={'3':['three'],'5':['five'],'dataset':['data','set'],'filesystem':['file','system'],'preprocessing':['pre','processing'],
      '8':['eight'],'8l':['eight','l'],'10':['ten'],'10h':['ten','h'],
      '13':['thirteen'],'26':['twenty','six'],'64':['sixty','four'],'x86':['x','eighty','six'],
      '334':['three','point','thirty','four'],'54':['five','point','four'],
      'mcp':['m','c','p'],'peft':['p','e','f','t'],'genai':['gen','ai'],'8lhef':['eight','l','hef']}
    return expansions.get(key,[aliases.get(key,key)])

def map_source_words(source,words,review):
    tokens=list(re.finditer(r'\S+',source))
    if not words:raise ValueError('Empty forced-alignment words')
    previous=-1.0
    for index,word in enumerate(words):
        if not isinstance(word.get('word'),str):raise ValueError('Missing aligned word text')
        start,end=word.get('start'),word.get('end')
        if any(type(v) not in (int,float) or not math.isfinite(v) for v in (start,end)) or start<0 or end<start or start<previous:
            raise ValueError('Invalid or out-of-order word timestamp at '+str(index))
        previous=start
    source_atoms=[(atom,i) for i,t in enumerate(tokens) for atom in atoms(t.group())]
    audio_atoms=[(atom,i) for i,w in enumerate(words) for atom in atoms(w['word'])]
    matcher=difflib.SequenceMatcher(None,[a[0] for a in source_atoms],[a[0] for a in audio_atoms],autojunk=False)
    mapped=[[] for _ in tokens]
    for tag,a,b,c,d in matcher.get_opcodes():
        if tag=='equal':
            for i,j in zip(range(a,b),range(c,d)):
                audio=words[audio_atoms[j][1]];mapped[source_atoms[i][1]].append((audio['start'],audio['end']))
        elif a<b and c<d:
            si=sorted({i for _,i in source_atoms[a:b]});ai=sorted({i for _,i in audio_atoms[c:d]})
            review.append({'kind':'source_alignment_replacement','source':join([tokens[i].group() for i in si]),'aligned':join([words[i]['word'] for i in ai]),'audio_start':words[ai[0]]['start'],'audio_end':words[ai[-1]]['end']})
            for i in si:mapped[i].append((words[ai[0]]['start'],words[ai[-1]]['end']))
        else:review.append({'kind':'source_alignment_gap','opcode':tag,'source_atoms':[a,b],'aligned_atoms':[c,d]})
    if any(not value for value in mapped):raise ValueError('Source words missing from forced alignment; review transcript before captioning')
    return tokens,[(min(v[0] for v in values),max(v[1] for v in values)) for values in mapped]

def display_units(source,display,words,review):
    source_tokens,times=map_source_words(source,words,review)
    segments=[];last=0
    for match in PHRASE_PATTERN.finditer(source):
        if match.start()>last:
            for word in re.finditer(r'\S+',source[last:match.start()]):segments.append((word.group(),last+word.start(),last+word.end(),False))
        raw=match.group();phrase=raw.rstrip('.,;:!?');suffix=raw[len(phrase):]
        segments.append((PHRASES[phrase]+suffix,match.start(),match.end(),True));last=match.end()
    for word in re.finditer(r'\S+',source[last:]):segments.append((word.group(),last+word.start(),last+word.end(),False))
    rendered=join([s[0] for s in segments])
    if rendered!=display:raise ValueError('Normalization mapping does not reproduce exact caption display text')
    units=[]
    for text,start,end,technical in segments:
        indices=[i for i,t in enumerate(source_tokens) if t.start()<end and t.end()>start]
        if not indices:raise ValueError('Display phrase has no source words')
        units.append({'text':text,'start':times[indices[0]][0],'end':times[indices[-1]][1],'technical':technical})
    return units

def make_cues(units,review,width=42,max_duration=5.5):
    groups=[];current=[]
    for unit in units:
        candidate=current+[unit];text=join([u['text'] for u in candidate])
        wrapped=textwrap.wrap(text,width=width,break_long_words=False,break_on_hyphens=False)
        if current and (len(wrapped)>2 or unit['end']-current[0]['start']>max_duration or unit['start']-current[-1]['end']>0.65):
            groups.append(current);current=[]
        current.append(unit)
        if unit['text'].endswith(('.', '?', '!')):
            groups.append(current);current=[]
    if current:groups.append(current)
    # Avoid a fleeting one-word tail when a line-length split lands just before
    # sentence end. Move measured word units, never invent proportional timing.
    for i in range(1,len(groups)):
        previous=groups[i-1]
        if len(groups[i])<3 and len(previous)>4 and not previous[-1]['text'].endswith(('.', '?', '!')):
            groups[i]=previous[-3:]+groups[i];groups[i-1]=previous[:-3]
    cues=[]
    for index,group in enumerate(groups):
        text=join([u['text'] for u in group]);lines=textwrap.wrap(text,width=width,break_long_words=False,break_on_hyphens=False)
        start=group[0]['start'];end=group[-1]['end']+0.12
        if index+1<len(groups):end=min(end,groups[index+1][0]['start']-0.02)
        if end<=start:raise ValueError('Caption units have overlapping or zero-duration timing; inspect alignment')
        cps=len(text)/(end-start)
        if len(lines)>2 or any(len(line)>width for line in lines) or cps>24 or end-start<0.65:
            review.append({'kind':'readability','text':text,'seconds':end-start,'characters_per_second':round(cps,2),'lines':len(lines)})
        cues.append({'start':start,'end':end,'text':'\n'.join(lines)})
    return cues

def timestamp(seconds):
    ms=round(seconds*1000);hours,ms=divmod(ms,3600000);minutes,ms=divmod(ms,60000);secs,ms=divmod(ms,1000)
    return f'{hours:02}:{minutes:02}:{secs:02},{ms:03}'

def build(args):
    source=json.loads(args.narration.read_text());display=json.loads(args.script.read_text());manifest=json.loads(args.manifest.read_text())
    ids=[c['id'] for c in source]
    if len(ids)!=11 or [c['id'] for c in display]!=ids or [c['id'] for c in manifest['chapters']]!=ids:raise ValueError('Expected matching ordered11chapters')
    marked='elevenlabs' in str(manifest.get('voice_provider','')).lower()
    if not marked and not args.allow_unmarked_manifest:raise ValueError('Manifest is not marked voice_provider ElevenLabs; wait for new video timing')
    delay=manifest.get('voice_delay_seconds',args.voice_delay)
    if type(delay) not in (int,float) or not 0<=delay<=3:raise ValueError('Invalid voice delay')
    # Read every alignment before writing anything. Missing chapter is fatal.
    alignments=[]
    for chapter in source:
        path=args.alignments/(chapter['id']+'-alignment.json')
        data=json.loads(path.read_text())
        if not data.get('ok'):raise ValueError('Alignment not successful: '+chapter['id'])
        alignments.append((path,data['words']))
    all_cues=[];chapter_reports=[];previous_end=0
    for src,dst,timing,(path,words) in zip(source,display,manifest['chapters'],alignments):
        review=[];units=display_units(src['text'],dst['text'],words,review);cues=make_cues(units,review)
        start=timing['start'];voice=timing['voice_seconds'];duration=timing['duration']
        if start<previous_end-0.05 or voice<=0 or duration<voice+delay-0.05:raise ValueError('Invalid new chapter timing: '+src['id'])
        if words[-1]['end']>voice+0.2:raise ValueError('Alignment exceeds voice duration; stale manifest: '+src['id'])
        for cue in cues:
            cue['start']+=start+delay;cue['end']=min(cue['end']+start+delay,start+duration)
            all_cues.append(cue)
        previous_end=start+duration
        for item in review:
            if 'audio_start' in item:
                item['video_start']=round(start+delay+item['audio_start'],3)
                item['video_end']=round(start+delay+item['audio_end'],3)
                item['disposition']='Correct source/display wording retained; actual measured replacement span used. Spot-listen if exact pronunciation is material.'
        chapter_reports.append({'id':src['id'],'title':dst['title'],'cues':len(cues),'aligned_word_count':len(words),'alignment_sha256':sha(path),'technical_phrase_units':sum(u['technical'] for u in units),'review_items':review})
    if any(a['end']>b['start'] for a,b in zip(all_cues,all_cues[1:])):raise ValueError('Overlapping SRT cues')
    srt='\n\n'.join(f"{i}\n{timestamp(c['start'])} --> {timestamp(c['end'])}\n{c['text']}" for i,c in enumerate(all_cues,1))+'\n'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(srt)
    report={'schema':'pi-trainer/word-aligned-captions/v1','voice_delay_seconds':delay,'cue_count':len(all_cues),'timing':'Actual forced-alignment word intervals; normalized phrases retain their source spoken span','script_sha256':sha(args.script),'manifest_sha256':sha(args.manifest),'srt_sha256':sha(args.output),'unmarked_manifest_override':not marked,'review_item_count':sum(len(c['review_items']) for c in chapter_reports),'chapters':chapter_reports}
    args.output.with_suffix('.review.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'output':str(args.output),'cues':len(all_cues),'review_items':report['review_item_count']},indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--script',type=Path,default=BASE/'caption-script.json');p.add_argument('--narration',type=Path,default=BASE/'narration.json')
    p.add_argument('--manifest',type=Path,default=BASE/'video-manifest.json');p.add_argument('--alignments',type=Path,default=BASE/'audio/elevenlabs')
    p.add_argument('--output',type=Path,default=BASE/'revision-elevenlabs/captions.srt');p.add_argument('--voice-delay',type=float,default=0.7)
    p.add_argument('--allow-unmarked-manifest',action='store_true',help='Explicit review override for a newly timed manifest lacking provider metadata')
    args=p.parse_args()
    if args.output.resolve()==(BASE/'output/Pi-Trainer-Walkthrough.srt').resolve():p.error('Write a candidate first; final release SRT publication belongs to the release step')
    build(args)
if __name__=='__main__':main()
