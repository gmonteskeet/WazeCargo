"""Internal helper: build .ipynb files from (cell_type, source) lists."""
import nbformat as nbf
from pathlib import Path

def build_notebook(out_path: str, cells: list):
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.13",
            "mimetype": "text/x-python",
            "file_extension": ".py",
        },
    }
    nb_cells = []
    for kind, src in cells:
        if kind == "md":
            nb_cells.append(nbf.v4.new_markdown_cell(src))
        elif kind == "code":
            nb_cells.append(nbf.v4.new_code_cell(src))
        else:
            raise ValueError(kind)
    nb.cells = nb_cells
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(p))
    print(f"  ✓ {p.name}  ({len(cells)} cells)")
