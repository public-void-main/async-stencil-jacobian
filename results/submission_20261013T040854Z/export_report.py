"""Small offline renderer for this report's Markdown subset; no network assets."""
import html
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[2]
source=ROOT/'report.md'
target=ROOT/'report_print.html'

def inline(value):
    value=html.escape(value)
    value=re.sub(r'`([^`]+)`',r'<code>\1</code>',value)
    value=re.sub(r'\*\*([^*]+)\*\*',r'<strong>\1</strong>',value)
    value=re.sub(r'\*([^*]+)\*',r'<em>\1</em>',value)
    value=re.sub(r'\[([^\]]+)\]\((.*?)\)(?=\s|$|[.,;])',r'<a href="\2">\1</a>',value)
    return value

def table_value(value):
    # Compact printed decimals only; Markdown/CSV retain their original precision.
    if len(value)>8 and re.fullmatch(r'[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?',value):
        value=format(float(value),'.4g')
    return inline(value)

lines=source.read_text().splitlines()
body=[];headings=[];i=0
while i<len(lines):
    line=lines[i]
    if not line.strip():i+=1;continue
    if line.startswith('```'):
        code=[];i+=1
        while i<len(lines) and not lines[i].startswith('```'):
            code.append(lines[i]);i+=1
        body.append('<pre>'+html.escape('\n'.join(code))+'</pre>');i+=1;continue
    heading=re.match(r'^(#{1,6}) (.*)',line)
    if heading:
        level=len(heading[1]);title=heading[2];anchor='section-'+str(len(headings))
        headings.append((level,title,anchor))
        body.append(f'<h{level} id="{anchor}">'+inline(title)+f'</h{level}>');i+=1;continue
    image=re.match(r'^!\[(.*?)\]\((.*?)\)$',line)
    if image:
        path=ROOT/image[2]
        if not path.is_file():raise FileNotFoundError(path)
        body.append('<figure><img src="'+path.resolve().as_uri()+'" alt="'+html.escape(image[1])+'"></figure>')
        i+=1;continue
    if line.startswith('|'):
        rows=[]
        while i<len(lines) and lines[i].startswith('|'):
            cells=[x.strip() for x in lines[i].strip('|').split('|')]
            if not all(re.fullmatch(r':?-+:?',x) for x in cells):rows.append(cells)
            i+=1
        # Writer's HTML importer does not reliably inherit table font sizes.
        # Explicit cell fonts and nowrap keep numeric measurements readable.
        cell_width=str(100/len(rows[0]))+'%'
        body.append('<table width="100%" border="1" cellpadding="3" cellspacing="0"><thead><tr>'+''.join('<th width="'+cell_width+'"><font face="Arial" size="1">'+inline(x)+'</font></th>' for x in rows[0])+'</tr></thead><tbody>')
        body.extend('<tr>'+''.join('<td width="'+cell_width+'" nowrap><font face="Arial" size="1">'+table_value(x)+'</font></td>' for x in row)+'</tr>' for row in rows[1:])
        body.append('</tbody></table>');continue
    item=re.match(r'^(?:- |\d+\. )(.*)',line)
    if item:
        numbered=bool(re.match(r'^\d+\.',line));tag='ol' if numbered else 'ul'
        body.append('<'+tag+'>')
        while i<len(lines):
            match=re.match(r'^(?:- |\d+\. )(.*)',lines[i])
            if not match:break
            body.append('<li>'+inline(match[1])+'</li>');i+=1
        body.append('</'+tag+'>');continue
    paragraph=[]
    while i<len(lines) and lines[i].strip() and not re.match(r'^(#|```|\||!\[)',lines[i]):
        paragraph.append(lines[i]);i+=1
    body.append('<p>'+inline(' '.join(paragraph))+'</p>')

css='''
@page { size: A4; margin: 18mm 16mm; }
body { font-family: "DejaVu Serif", serif; font-size: 10.5pt; line-height: 1.35; color: #17202a; }
h1 { font-family: "DejaVu Sans", sans-serif; font-size: 23pt; color: #17365d; }
h2 { font-family: "DejaVu Sans", sans-serif; font-size: 15pt; color: #17365d; margin-top: 20pt; page-break-after: avoid; }
h3 { font-size: 12pt; color: #17365d; page-break-after: avoid; }
p { margin: 7pt 0; }
table { border-collapse: collapse; width: 100%; margin: 10pt 0; font-family: "DejaVu Sans", sans-serif; font-size: 8pt; }
th { background: #e7edf5; text-align: left; }
th,td { border: 1px solid #b8c6d6; padding: 4px; }
tr { page-break-inside: avoid; }
pre { font-family: "DejaVu Sans Mono", monospace; font-size: 8pt; background: #f4f6f8; padding: 9px; white-space: pre-wrap; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 9pt; }
figure { margin: 12pt 0; page-break-inside: avoid; }
img { width: 100%; height: auto; }
a { color: #174f80; }
li { margin-bottom: 5pt; }
'''
document='<!DOCTYPE html><html><head><meta charset="utf-8"><title>CUDA Jacobi PCAP Report</title><style>'+css+'</style></head><body>'+''.join(body)+'</body></html>'
with target.open('w') as handle:handle.write(document)
print('HTML report:',target)
