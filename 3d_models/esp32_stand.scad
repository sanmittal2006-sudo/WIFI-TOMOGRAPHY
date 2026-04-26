/*
 * ESP32-S3 Mounting Stand — Matches Reference Photo
 * ===================================================
 * - Total height: 178mm (17.8cm)
 * - Base: 50×50×4mm
 * - Hollow square rod (saves material)  
 * - ESP32 bracket at 60mm from base (6cm)
 * - Rod continues above ESP32 to top
 * - ESP32-S3-DevKitC-1: 62.74mm × 25.40mm
 *
 * OpenSCAD: F5=preview, F6=render, Export as STL
 */

// === DIMENSIONS (all in mm) ===
total_height = 178;
base_w = 50;
base_d = 50;
base_h = 4;

// Rod (hollow square tube)
rod_outer = 14;        // 14mm square rod
rod_wall = 2;          // 2mm wall → 10mm hollow core
rod_inner = rod_outer - 2*rod_wall;

// ESP32 bracket position
bracket_center_z = 60; // ESP32 center at 6cm from base bottom

// ESP32-S3-DevKitC-1 with print tolerance
pcb_length = 64.5;    // 62.74 + 1.76mm tolerance
pcb_width = 27;        // 25.40 + 1.6mm tolerance
pcb_thick = 1.8;       // PCB thickness
board_thick = 10;      // Total thickness with components below

// Bracket walls
bwall = 2.5;           // Bracket wall thickness

// === COMPUTED ===
bracket_h = pcb_width + 2*bwall;  // ~32mm tall
bracket_z = bracket_center_z - bracket_h/2;  // Z start of bracket

// === BASE PLATE ===
module base() {
    // Simple flat base
    translate([-base_w/2, -base_d/2, 0])
        cube([base_w, base_d, base_h]);
}

// === HOLLOW SQUARE ROD ===
module hollow_rod() {
    rod_h = total_height - base_h;
    translate([0, 0, base_h]) {
        difference() {
            // Outer square
            translate([-rod_outer/2, -rod_outer/2, 0])
                cube([rod_outer, rod_outer, rod_h]);
            // Inner hollow
            translate([-rod_inner/2, -rod_inner/2, -0.1])
                cube([rod_inner, rod_inner, rod_h + 0.2]);
        }
    }
}

// === ESP32 BRACKET (C-shaped cradle) ===
module esp32_bracket() {
    /*
     * Side view (looking from left):
     *        ┌─────────────┐ ← top wall
     *        │  ESP32 PCB  │
     *        │  sits here  │
     *        │             │
     * ROD ──►│             │◄── open front (antenna)
     *        │             │
     *        └─────────────┘ ← bottom shelf
     *
     * The bracket extends from the rod toward the phantom.
     * Front is OPEN so antenna radiates toward phantom.
     */
    
    bracket_w = pcb_length + 2*bwall;  // total width ~69.5mm
    bracket_d = board_thick + bwall;   // depth from rod face ~12.5mm
    
    translate([0, 0, bracket_z]) {
        // Bottom shelf (ESP32 rests on this)
        translate([-bracket_w/2, -rod_outer/2 - bracket_d, 0])
            cube([bracket_w, bracket_d + rod_outer/2, bwall]);
        
        // Left side wall
        translate([-bracket_w/2, -rod_outer/2 - bracket_d, 0])
            cube([bwall, bracket_d, bracket_h]);
        
        // Right side wall
        translate([bracket_w/2 - bwall, -rod_outer/2 - bracket_d, 0])
            cube([bwall, bracket_d, bracket_h]);
        
        // Back wall (connects to rod, small reinforcement)
        translate([-bracket_w/2, -rod_outer/2 - bwall, 0])
            cube([bracket_w, bwall, bracket_h]);
        
        // Top wall (holds ESP32 from above)
        translate([-bracket_w/2, -rod_outer/2 - bracket_d, bracket_h - bwall])
            cube([bracket_w, bracket_d + rod_outer/2, bwall]);
        
        // Front lip (small, leaves antenna area open)
        // Left lip
        translate([-bracket_w/2, -rod_outer/2 - bracket_d - bwall, 0])
            cube([15, bwall, bracket_h]);
        // Right lip  
        translate([bracket_w/2 - 15, -rod_outer/2 - bracket_d - bwall, 0])
            cube([15, bwall, bracket_h]);
        
        // PCB guide rails (two thin rails for PCB to slide into)
        // Bottom rail (PCB rests on these)
        translate([-pcb_length/2, -rod_outer/2 - bracket_d + bwall, bwall])
            cube([pcb_length, 1.5, pcb_thick]);
        // Top rail (holds PCB from above)
        translate([-pcb_length/2, -rod_outer/2 - bracket_d + bwall, bracket_h - bwall - pcb_thick])
            cube([pcb_length, 1.5, pcb_thick]);
    }
}

// === USB CABLE CHANNEL ===
module usb_channel() {
    // Small channel through the bracket for USB cable
    translate([-5, -rod_outer/2 - 15, bracket_z + bwall + 2])
        cube([10, 5, 12]);
}

// === COMPLETE STAND ===
module esp32_stand() {
    difference() {
        union() {
            base();
            hollow_rod();
            esp32_bracket();
        }
        usb_channel();
    }
}

// === RENDER ===
color("LightGray")
    esp32_stand();

// === DEBUG INFO ===
echo(str("Total height: ", total_height, " mm"));
echo(str("Base: ", base_w, "×", base_d, "×", base_h, " mm"));
echo(str("Rod: ", rod_outer, "mm square, ", rod_wall, "mm wall (hollow)"));
echo(str("ESP32 bracket center at Z=", bracket_center_z, " mm (", bracket_center_z/10, " cm)"));
echo(str("Bracket size: ", pcb_length + 2*bwall, "×", bracket_h, " mm"));
