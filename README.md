4D Visualizer

Interactive 4D → 2D Oblique Projection Viewer

This repository provides an interactive Python-based visualizer for projecting 4-dimensional vectors into a 2-dimensional plane using oblique projection.
It is designed as a research tool for exploring 4D geometry, rotation planes, basis decomposition, and higher-dimensional intuition.
This tool provides an intuitive way to understand 4D rotations and projections through real-time interactive visualization.

<img width="1145" height="957" alt="image" src="https://github.com/user-attachments/assets/0fdf28ab-3af4-4f06-8f26-3962075ce9ab" />


Features

Projection of the four coordinate axes (x, y, z, w) into 2D.
Adjustable projection angles (θx, θy, θz, θw).
Full 4D rotation capability in the planes: xy, xz, xw, yz, yw, zw.
Two modes:
AXES mode — rotate the projection basis
OBJECT mode — rotate the 4D object
Component-wise decomposition of 4D vectors.
Visualization of 3D slices (“boxes”) in 2D projection.
Interactive controls using Matplotlib widgets.

Requirements

Python 3.8+
Install dependencies:
pip install numpy matplotlib

Usage

Run the viewer:
python 4d_visualizer.py

Controls:
Drag → rotate in the selected 4D planes
Radio buttons → choose rotation planes
Sliders → adjust projection angles and vector components
Mode switch → AXES / OBJECT

File Structure

4d_visualizer.py — main viewer script
README.md — description and documentation

Citation

Kohei Abe, “4D Visualizer: Interactive 4D → 2D Oblique Projection Viewer”,
GitHub repository, https://github.com/KoheiAbeLab/4d-visualizer

BibTeX:
@misc{abe2025_4dvisualizer,
  author       = {Kohei Abe},
  title        = {4D Visualizer: Interactive 4D → 2D Oblique Projection Viewer},
  year         = {2025},
  publisher    = {GitHub},
  howpublished = {https://github.com/KoheiAbeLab/4d-visualizer},
}


License

MIT License

Contact

Kohei Abe
ORCID: https://orcid.org/0009-0001-1126-3282
GitHub: https://github.com/KoheiAbeLab
