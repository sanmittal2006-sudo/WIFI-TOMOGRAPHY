#!/usr/bin/env python3
"""
Generate ESP32 Stand STL — Matching Reference Photo
Total height: 178mm, ESP32 at 6cm, hollow rod, 4mm base
"""
import numpy as np
from stl import mesh
import os

OUTPUT_DIR = "3d_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_box(x, y, z, w, d, h):
    """Create box at (x,y,z) with dimensions (w,d,h). Returns triangle faces."""
    v = [
        [x,y,z],[x+w,y,z],[x+w,y+d,z],[x,y+d,z],
        [x,y,z+h],[x+w,y,z+h],[x+w,y+d,z+h],[x,y+d,z+h]
    ]
    return [
        [v[0],v[2],v[1]], [v[0],v[3],v[2]],  # bottom
        [v[4],v[5],v[6]], [v[4],v[6],v[7]],  # top
        [v[0],v[1],v[5]], [v[0],v[5],v[4]],  # front
        [v[2],v[3],v[7]], [v[2],v[7],v[6]],  # back
        [v[0],v[4],v[7]], [v[0],v[7],v[3]],  # left
        [v[1],v[2],v[6]], [v[1],v[6],v[5]],  # right
    ]

def save_stl(faces, filename):
    n = len(faces)
    m = mesh.Mesh(np.zeros(n, dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            m.vectors[i][j] = np.array(f[j])
    filepath = os.path.join(OUTPUT_DIR, filename)
    m.save(filepath)
    print(f"  Saved: {filepath} ({n} triangles)")

def generate_esp32_stand():
    """
    Matches reference photo exactly:
    - Total height: 178mm
    - Base: 50×50×4mm
    - Hollow square rod: 14mm outer, 10mm inner
    - ESP32 C-bracket at 6cm from base
    - Rod continues to top
    """
    print("  Generating ESP32 Stand (photo-matched)...", flush=True)
    
    # === DIMENSIONS ===
    total_h = 178.0
    base_w, base_d, base_h = 50.0, 50.0, 4.0
    rod_outer = 14.0
    rod_wall = 2.0
    rod_inner = rod_outer - 2 * rod_wall  # 10mm
    
    # ESP32 bracket
    esp_center_z = 60.0    # 6cm from base bottom
    pcb_l = 64.5           # length with tolerance
    pcb_w = 27.0           # width with tolerance  
    board_d = 12.0         # total depth with components
    bw = 2.5               # bracket wall thickness
    
    bracket_h = pcb_w + 2 * bw   # ~32mm
    bracket_z = esp_center_z - bracket_h / 2
    bracket_total_w = pcb_l + 2 * bw   # ~69.5mm
    bracket_depth = board_d + bw        # ~14.5mm
    
    faces = []
    
    # === 1. BASE PLATE ===
    faces += create_box(-base_w/2, -base_d/2, 0, base_w, base_d, base_h)
    
    # === 2. HOLLOW SQUARE ROD (4 walls) ===
    rod_h = total_h - base_h  # 174mm
    rz = base_h
    ro = rod_outer / 2
    ri = rod_inner / 2
    
    # Front wall of rod
    faces += create_box(-ro, -ro, rz, rod_outer, rod_wall, rod_h)
    # Back wall of rod
    faces += create_box(-ro, ro - rod_wall, rz, rod_outer, rod_wall, rod_h)
    # Left wall of rod
    faces += create_box(-ro, -ri, rz, rod_wall, rod_inner, rod_h)
    # Right wall of rod
    faces += create_box(ro - rod_wall, -ri, rz, rod_wall, rod_inner, rod_h)
    # Top cap of rod
    faces += create_box(-ro, -ro, rz + rod_h, rod_outer, rod_outer, rod_wall)
    
    # === 3. ESP32 BRACKET (C-shape, attached to front of rod) ===
    # Bracket extends forward (toward -Y direction = toward phantom)
    by = -ro - bracket_depth  # front edge Y
    
    # Bottom shelf
    faces += create_box(-bracket_total_w/2, by, bracket_z,
                        bracket_total_w, bracket_depth + ro, bw)
    
    # Top shelf
    faces += create_box(-bracket_total_w/2, by, bracket_z + bracket_h - bw,
                        bracket_total_w, bracket_depth + ro, bw)
    
    # Left side wall
    faces += create_box(-bracket_total_w/2, by, bracket_z,
                        bw, bracket_depth, bracket_h)
    
    # Right side wall
    faces += create_box(bracket_total_w/2 - bw, by, bracket_z,
                        bw, bracket_depth, bracket_h)
    
    # Back wall (reinforcement against rod)
    faces += create_box(-bracket_total_w/2, -ro - bw, bracket_z,
                        bracket_total_w, bw, bracket_h)
    
    # Front lips (partial — leave center open for antenna)
    lip_w = 12.0  # each lip width
    # Left front lip
    faces += create_box(-bracket_total_w/2, by - bw, bracket_z,
                        lip_w, bw, bracket_h)
    # Right front lip
    faces += create_box(bracket_total_w/2 - lip_w, by - bw, bracket_z,
                        lip_w, bw, bracket_h)
    
    # Bottom PCB rail (thin ledge for PCB to rest on)
    rail_h = 1.5
    rail_d = 2.0
    faces += create_box(-pcb_l/2, by + bw, bracket_z + bw,
                        pcb_l, rail_d, rail_h)
    
    save_stl(faces, "esp32_stand.stl")

# === ALSO REGENERATE CYLINDER PARTS ===
def create_cylinder_faces(outer_r, inner_r, height, bottom_thick, segs=64, z0=0):
    faces = []
    for i in range(segs):
        a1 = 2*np.pi*i/segs
        a2 = 2*np.pi*(i+1)/segs
        c1,s1 = np.cos(a1),np.sin(a1)
        c2,s2 = np.cos(a2),np.sin(a2)
        
        ob1=[outer_r*c1,outer_r*s1,z0]; ob2=[outer_r*c2,outer_r*s2,z0]
        ot1=[outer_r*c1,outer_r*s1,z0+height]; ot2=[outer_r*c2,outer_r*s2,z0+height]
        
        # Outer wall
        faces.append([ob1,ot1,ob2]); faces.append([ob2,ot1,ot2])
        # Bottom (solid)
        faces.append([ob2,ob1,[0,0,z0]])
        
        if inner_r > 0:
            ib1=[inner_r*c1,inner_r*s1,z0+bottom_thick]
            ib2=[inner_r*c2,inner_r*s2,z0+bottom_thick]
            it1=[inner_r*c1,inner_r*s1,z0+height]
            it2=[inner_r*c2,inner_r*s2,z0+height]
            # Inner wall
            faces.append([ib1,ib2,it1]); faces.append([ib2,it2,it1])
            # Top rim
            faces.append([ot1,it1,ot2]); faces.append([ot2,it1,it2])
            # Inner bottom
            faces.append([[0,0,z0+bottom_thick],ib1,ib2])
            # Top of bottom slab
            obt1=[outer_r*c1,outer_r*s1,z0+bottom_thick]
            obt2=[outer_r*c2,outer_r*s2,z0+bottom_thick]
            faces.append([obt2,obt1,[0,0,z0+bottom_thick]])
        else:
            faces.append([[0,0,z0+height],ot1,ot2])
    return faces

def main():
    print("="*55)
    print("  3D Part Generator — Wi-Fi Tomography")
    print("="*55)
    
    # ESP32 Stand (matches photo)
    generate_esp32_stand()
    
    # Phantom outer shell: 200mm dia, 120mm tall, 2.5mm wall
    print("\n  Generating Phantom Outer Shell...")
    save_stl(create_cylinder_faces(100, 97.5, 120, 3, segs=80), "phantom_outer_shell.stl")
    
    # Lung cylinders: 50mm dia, 105mm tall, 2mm wall
    print("\n  Generating Lung Left...")
    save_stl(create_cylinder_faces(25, 23, 105, 2.5, segs=60), "lung_left.stl")
    print("\n  Generating Lung Right...")
    save_stl(create_cylinder_faces(25, 23, 105, 2.5, segs=60), "lung_right.stl")
    
    # Heart: 30mm dia, 105mm tall, 1.5mm wall
    print("\n  Generating Heart Cylinder...")
    save_stl(create_cylinder_faces(15, 13.5, 105, 2, segs=48), "heart_cylinder.stl")
    
    print(f"\n{'='*55}")
    print("  ALL PARTS GENERATED!")
    print(f"{'='*55}")
    print(f"\n  Files in 3d_models/:")
    print(f"    esp32_stand.stl        — Print ×2 (TX + RX)")
    print(f"    phantom_outer_shell.stl — Print ×1 (PETG, 100% walls)")
    print(f"    lung_left.stl          — Print ×1")
    print(f"    lung_right.stl         — Print ×1")
    print(f"    heart_cylinder.stl     — Print ×1 (optional)")
    print(f"\n  ESP32 STAND SPECS:")
    print(f"    Total height: 178mm")
    print(f"    Base: 50×50×4mm")
    print(f"    Rod: 14mm square, HOLLOW (2mm wall)")
    print(f"    ESP32 bracket: center at 60mm (6cm) from base")
    print(f"    Bracket fits: ESP32-S3-DevKitC-1 (62.74×25.40mm)")

if __name__ == '__main__':
    main()
