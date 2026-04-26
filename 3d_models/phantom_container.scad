/*
 * Phantom Container for Wi-Fi Tomography
 * =======================================
 * 3D-printed chest model with:
 *   - Outer cylindrical shell (chest cavity)
 *   - Left lung cylinder (hollow, air-filled)
 *   - Right lung cylinder (hollow, air-filled)
 *   - Heart cylinder (optional, center)
 *
 * HOW TO USE:
 *   1. Open in OpenSCAD (free: openscad.org)
 *   2. F5 to preview, F6 to render
 *   3. Export as STL
 *   4. Print EACH piece separately!
 *
 * Print settings: PETG (waterproof), 0.2mm layer, 100% infill walls,
 *                 4+ wall lines, NO top layers (open top)
 */

// === PARAMETERS ===
// Outer shell (chest cavity)
outer_diameter = 200;     // 20cm
outer_height = 120;       // 12cm
outer_wall = 2.5;         // Wall thickness for waterproofing
outer_bottom = 3;         // Bottom plate thickness

// Lung cylinders
lung_diameter = 50;       // 5cm each
lung_height = 105;        // 10.5cm (slightly shorter than outer)
lung_wall = 2;            // Wall thickness
lung_bottom = 2.5;        // Bottom plate thickness
lung_offset_x = 45;       // 4.5cm from center (each side)

// Heart cylinder (optional)
heart_diameter = 30;      // 3cm
heart_height = 105;       // Same as lungs
heart_wall = 1.5;         // Wall thickness
heart_bottom = 2;         // Bottom plate thickness

// Alignment pegs (to position lung cylinders correctly)
peg_diameter = 5;
peg_height = 3;

// Which piece to show/export (uncomment one at a time for export)
show_piece = "all";  // "outer", "lung_left", "lung_right", "heart", "all"

// === MODULES ===

// Outer Shell (chest cavity)
module outer_shell() {
    color("LightBlue", 0.5)
    difference() {
        // Outer cylinder
        cylinder(d=outer_diameter, h=outer_height, $fn=120);
        // Inner cavity (leave bottom solid)
        translate([0, 0, outer_bottom])
            cylinder(d=outer_diameter - 2*outer_wall, 
                    h=outer_height - outer_bottom + 1, $fn=120);
    }
    
    // Alignment pegs on bottom (to position lung cylinders)
    // Left lung peg
    translate([-lung_offset_x, 0, outer_bottom])
        cylinder(d=peg_diameter, h=peg_height, $fn=20);
    // Right lung peg
    translate([lung_offset_x, 0, outer_bottom])
        cylinder(d=peg_diameter, h=peg_height, $fn=20);
    // Heart peg
    translate([0, 0, outer_bottom])
        cylinder(d=peg_diameter, h=peg_height, $fn=20);
}

// Lung Cylinder (one lung)
module lung_cylinder() {
    color("PaleGreen", 0.7)
    difference() {
        // Outer wall
        cylinder(d=lung_diameter, h=lung_height, $fn=80);
        // Inner cavity (hollow)
        translate([0, 0, lung_bottom])
            cylinder(d=lung_diameter - 2*lung_wall, 
                    h=lung_height - lung_bottom + 1, $fn=80);
        // Alignment peg hole in bottom
        translate([0, 0, -0.1])
            cylinder(d=peg_diameter + 0.5, h=lung_bottom + 0.2, $fn=20);
    }
}

// Heart Cylinder
module heart_cylinder() {
    color("LightCoral", 0.7)
    difference() {
        cylinder(d=heart_diameter, h=heart_height, $fn=60);
        translate([0, 0, heart_bottom])
            cylinder(d=heart_diameter - 2*heart_wall, 
                    h=heart_height - heart_bottom + 1, $fn=60);
        // Alignment peg hole
        translate([0, 0, -0.1])
            cylinder(d=peg_diameter + 0.5, h=heart_bottom + 0.2, $fn=20);
    }
}

// === RENDER ===

if (show_piece == "outer" || show_piece == "all") {
    outer_shell();
}

if (show_piece == "lung_left" || show_piece == "all") {
    translate([-lung_offset_x, 0, outer_bottom + peg_height])
        lung_cylinder();
}

if (show_piece == "lung_right" || show_piece == "all") {
    translate([lung_offset_x, 0, outer_bottom + peg_height])
        lung_cylinder();
}

if (show_piece == "heart" || show_piece == "all") {
    translate([0, 0, outer_bottom + peg_height])
        heart_cylinder();
}

// === CROSS-SECTION VIEW (uncomment to see inside) ===
/*
difference() {
    union() {
        outer_shell();
        translate([-lung_offset_x, 0, outer_bottom + peg_height]) lung_cylinder();
        translate([lung_offset_x, 0, outer_bottom + peg_height]) lung_cylinder();
        translate([0, 0, outer_bottom + peg_height]) heart_cylinder();
    }
    // Cut away front half
    translate([-150, 0, -1]) cube([300, 150, 200]);
}
*/

// Debug info
echo(str("Outer shell: ", outer_diameter, "mm dia × ", outer_height, "mm tall"));
echo(str("Lung cylinders: ", lung_diameter, "mm dia × ", lung_height, "mm tall"));
echo(str("Heart: ", heart_diameter, "mm dia × ", heart_height, "mm tall"));
echo(str("Lung offset: ±", lung_offset_x, "mm from center"));
echo(str("Volume between walls ≈ ", 
    (PI*pow(outer_diameter/2-outer_wall,2)*outer_height - 
     2*PI*pow(lung_diameter/2,2)*lung_height - 
     PI*pow(heart_diameter/2,2)*heart_height) / 1000, " ml"));
