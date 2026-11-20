# ml-annotator

A lightweight, dependency-light desktop tool (Python + Tkinter) for labeling images for
**object detection** (bounding boxes) and **segmentation** (polygons).

## Features

- Bounding boxes and polygons, with a dedicated **Select** tool to move shapes and drag
  corner/vertex handles to resize or reshape them
- Classes are auto-detected from your annotations; double-click a class to rename/merge it everywhere
- **Auto-save / auto-load**: work is stored in `annotations_project.json` inside the image folder
- Undo / redo (`Ctrl+Z` / `Ctrl+Y`)
- Zoom to cursor, pan, EXIF-aware image orientation, fast rendering on large images
- Export to **YOLO** (detection or segmentation), **COCO JSON**, and **Pascal VOC XML**

## Install & run

Requires Python 3.9+ with Tkinter (bundled with the python.org installers; on Debian/Ubuntu: `sudo apt install python3-tk`).

```bash
pip install -r requirements.txt
python ml_annotator.py                # or: python ml_annotator.py /path/to/images
```

## Controls

| Action | How |
|---|---|
| Tools | `V` select · `B` box · `P` polygon |
| Draw box | drag on the image, then pick/type a class |
| Draw polygon | click points; click the first point, press `Enter`, or double-click to finish |
| Undo last polygon point / cancel polygon | `Backspace` / `Esc` |
| Move / resize (Select tool) | drag a shape; drag a handle |
| Delete selected | `Delete` |
| Previous / next image | `←` / `→` |
| Zoom | mouse wheel (zooms around the cursor) |
| Pan | right- or middle-button drag |
| Deselect | right-click |
| Open folder / save | `Ctrl+O` / `Ctrl+S` (`Cmd` on macOS) |

## Export formats

| Format | Output |
|---|---|
| YOLO – detection | `labels/*.txt` (`class cx cy w h`, normalized), `classes.txt`, `data.yaml`. Polygons are converted to their bounding box. |
| YOLO – segmentation | `labels/*.txt` (`class x1 y1 x2 y2 …`, normalized). Boxes are converted to 4-point polygons. |
| COCO | `annotations.json` (bbox, polygon segmentation, true polygon area) |
| Pascal VOC | one `.xml` per image; polygons also get their enclosing `bndbox` plus a non-standard `<polygon>` element |

Only images that have at least one annotation are exported.

## License

MIT. See [LICENSE](LICENSE).
