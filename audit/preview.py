"""Render medium_post.md to a standalone HTML preview with local images inlined (for review only)."""
import base64, os, re, subprocess, sys
H = os.path.dirname(os.path.abspath(__file__))
body = subprocess.run([sys.executable, '/home/claude/tools/md2medium.py', os.path.join(H, 'medium_post.md')],
                      capture_output=True, text=True, check=True).stdout
def inline_img(m):
    name = m.group(1).split('/')[-1]
    data = base64.b64encode(open(os.path.join(H, 'figures', name), 'rb').read()).decode()
    return f'src="data:image/png;base64,{data}"'
body = re.sub(r'src="(https://raw\.githubusercontent\.com[^"]+)"', inline_img, body)
html = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Draft preview</title><style>
:root{{--bg:#fff;--fg:#242424;--mut:#6b6b6b}}
@media (prefers-color-scheme:dark){{:root{{--bg:#121212;--fg:#e6e6e6;--mut:#a0a0a0}}}}
body{{background:var(--bg);color:var(--fg);font:20px/1.6 Georgia,serif;max-width:700px;margin:40px auto;padding:0 16px}}
h1{{font:700 40px/1.2 -apple-system,Helvetica,sans-serif}} h4{{color:var(--mut);font:400 22px/1.4 -apple-system,Helvetica,sans-serif;margin-top:-10px}}
h3{{font:700 26px/1.3 -apple-system,Helvetica,sans-serif;margin-top:40px}} img{{max-width:100%;background:#fff}}
figcaption{{color:var(--mut);font-size:15px;text-align:center}} a{{color:inherit}}</style></head><body>{body}</body></html>"""
open(os.path.join(H, 'preview.html'), 'w').write(html)
print('preview written')
