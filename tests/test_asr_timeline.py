import ast
from pathlib import Path

# Importing core pulls optional audio packages, so execute only the pure helper AST.
source=Path("youtube_auto_dub/core.py").read_text(encoding="utf-8")
tree=ast.parse(source)
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="_asr_timeline_metrics")
mod=ast.Module(body=[node],type_ignores=[]); ns={}; exec(compile(mod,"core.py","exec"),ns)
metrics=ns["_asr_timeline_metrics"]

def test_asr_timeline_finds_middle_hole():
    coverage,gap=metrics([{"start":0,"end":4},{"start":9,"end":12}],12)
    assert round(coverage,3)==0.583
    assert gap==5

def test_asr_timeline_merges_overlap():
    coverage,gap=metrics([{"start":0,"end":6},{"start":5,"end":10}],10)
    assert coverage==1 and gap==0
