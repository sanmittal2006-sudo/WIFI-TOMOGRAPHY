#Author: Wi-Fi Tomography Project
#Description: ESP32-S3 mounting stand — NO SHELL, just extrude+cut

import adsk.core, adsk.fusion, traceback

def run(context):
    ui = None
    try:
        app = adsk.core.Application.get()
        ui = app.userInterface
        design = app.activeProduct
        rootComp = design.rootComponent

        # DIMENSIONS in cm (Fusion internal unit)
        TOTAL_H  = 17.8
        BASE_W   = 5.0
        BASE_H   = 0.4
        ROD      = 1.4
        ROD_WALL = 0.2
        ESP_Z    = 15.0

        PCB_L    = 6.45
        PCB_W    = 2.70
        PCB_D    = 1.10
        HW       = 0.30

        H_W      = PCB_L + 2*HW
        H_H      = PCB_W + 2*HW
        DEPTH    = PCB_D + HW
        H_BOT    = ESP_Z - H_H/2

        USB_W    = 1.2
        USB_H    = 1.0
        CR       = 0.20

        sketches = rootComp.sketches
        extrudes = rootComp.features.extrudeFeatures
        xyPlane = rootComp.xYConstructionPlane

        # ===== 1. BASE =====
        sk = sketches.add(xyPlane)
        sk.sketchCurves.sketchLines.addTwoPointRectangle(
            adsk.core.Point3D.create(-BASE_W/2, -BASE_W/2, 0),
            adsk.core.Point3D.create(BASE_W/2, BASE_W/2, 0))
        ext = extrudes.addSimple(sk.profiles.item(0),
            adsk.core.ValueInput.createByReal(BASE_H),
            adsk.fusion.FeatureOperations.NewBodyFeatureOperation)

        # ===== 2. SOLID ROD =====
        planes = rootComp.constructionPlanes
        pInput = planes.createInput()
        pInput.setByOffset(xyPlane, adsk.core.ValueInput.createByReal(BASE_H))
        rodPlane = planes.add(pInput)

        sk2 = sketches.add(rodPlane)
        sk2.sketchCurves.sketchLines.addTwoPointRectangle(
            adsk.core.Point3D.create(-ROD/2, -ROD/2, 0),
            adsk.core.Point3D.create(ROD/2, ROD/2, 0))
        extrudes.addSimple(sk2.profiles.item(0),
            adsk.core.ValueInput.createByReal(TOTAL_H - BASE_H),
            adsk.fusion.FeatureOperations.JoinFeatureOperation)

        # ===== 3. HOLLOW ROD (cut inner rectangle) =====
        sk3 = sketches.add(rodPlane)
        inner = ROD - 2*ROD_WALL
        sk3.sketchCurves.sketchLines.addTwoPointRectangle(
            adsk.core.Point3D.create(-inner/2, -inner/2, 0),
            adsk.core.Point3D.create(inner/2, inner/2, 0))
        # Cut upward through the rod, leaving top cap
        cutInput = extrudes.createInput(sk3.profiles.item(0),
            adsk.fusion.FeatureOperations.CutFeatureOperation)
        cutDist = adsk.fusion.DistanceExtentDefinition.create(
            adsk.core.ValueInput.createByReal(TOTAL_H - BASE_H - ROD_WALL))
        cutInput.setOneSideExtent(cutDist, adsk.fusion.ExtentDirections.PositiveExtentDirection)
        extrudes.add(cutInput)

        # ===== 4. HOLDER SOLID BOX =====
        pInput2 = planes.createInput()
        pInput2.setByOffset(xyPlane, adsk.core.ValueInput.createByReal(H_BOT))
        holderPlane = planes.add(pInput2)

        sk4 = sketches.add(holderPlane)
        sk4.sketchCurves.sketchLines.addTwoPointRectangle(
            adsk.core.Point3D.create(-H_W/2, -ROD/2 - DEPTH, 0),
            adsk.core.Point3D.create(H_W/2, -ROD/2, 0))
        extrudes.addSimple(sk4.profiles.item(0),
            adsk.core.ValueInput.createByReal(H_H),
            adsk.fusion.FeatureOperations.JoinFeatureOperation)

        # ===== 5. CUT ESP32 POCKET (from front, open front) =====
        # Pocket: starts at front face, goes inward, stops before back wall
        pInput3 = planes.createInput()
        pInput3.setByOffset(xyPlane, adsk.core.ValueInput.createByReal(H_BOT + HW))
        pocketPlane = planes.add(pInput3)

        sk5 = sketches.add(pocketPlane)
        sk5.sketchCurves.sketchLines.addTwoPointRectangle(
            adsk.core.Point3D.create(-PCB_L/2, -ROD/2 - DEPTH, 0),
            adsk.core.Point3D.create(PCB_L/2, -ROD/2 - HW, 0))
        cutInput2 = extrudes.createInput(sk5.profiles.item(0),
            adsk.fusion.FeatureOperations.CutFeatureOperation)
        cutDist2 = adsk.fusion.DistanceExtentDefinition.create(
            adsk.core.ValueInput.createByReal(PCB_W))
        cutInput2.setOneSideExtent(cutDist2, adsk.fusion.ExtentDirections.PositiveExtentDirection)
        extrudes.add(cutInput2)

        # ===== 6. CUT USB SLOT (left side) =====
        # Find or create a plane on the left face of holder
        pInput4 = planes.createInput()
        # Create XZ plane offset to the left side of holder
        xzPlane = rootComp.xZConstructionPlane
        pInput4.setByOffset(xzPlane, adsk.core.ValueInput.createByReal(-ROD/2 - DEPTH/2))
        usbPlane = planes.add(pInput4)

        sk6 = sketches.add(usbPlane)
        usb_z_center = ESP_Z
        sk6.sketchCurves.sketchLines.addTwoPointRectangle(
            adsk.core.Point3D.create(-H_W/2 - 0.1, usb_z_center - USB_H/2, 0),
            adsk.core.Point3D.create(-H_W/2 + HW + 0.1, usb_z_center + USB_H/2, 0))
        cutInput3 = extrudes.createInput(sk6.profiles.item(0),
            adsk.fusion.FeatureOperations.CutFeatureOperation)
        cutInput3.setAllExtent(adsk.fusion.ExtentDirections.SymmetricExtentDirection)
        extrudes.add(cutInput3)

        # ===== 7. FILLETS =====
        fillets = rootComp.features.filletFeatures
        body = rootComp.bRepBodies.item(0)

        # Fillet holder outer vertical edges
        holderEdges = adsk.core.ObjectCollection.create()
        for edge in body.edges:
            sp = edge.startVertex.geometry
            ep = edge.endVertex.geometry
            mz = (sp.z + ep.z) / 2
            # Vertical edges in holder Z range
            if mz > H_BOT + 0.01 and mz < (H_BOT + H_H - 0.01):
                if abs(sp.x - ep.x) < 0.001 and abs(sp.y - ep.y) < 0.001:
                    # Check if it's an outer edge (not inner pocket edge)
                    if abs(abs(sp.x) - H_W/2) < 0.01:
                        holderEdges.add(edge)

        if holderEdges.count > 0:
            try:
                fInput = fillets.createInput()
                fInput.addConstantRadiusEdgeSet(holderEdges,
                    adsk.core.ValueInput.createByReal(CR), True)
                fillets.add(fInput)
            except:
                pass

        # Fillet base vertical edges
        baseEdges = adsk.core.ObjectCollection.create()
        for edge in body.edges:
            sp = edge.startVertex.geometry
            ep = edge.endVertex.geometry
            mz = (sp.z + ep.z) / 2
            if mz > 0 and mz < BASE_H:
                if abs(sp.x - ep.x) < 0.001 and abs(sp.y - ep.y) < 0.001:
                    if abs(abs(sp.x) - BASE_W/2) < 0.02 or abs(abs(sp.y) - BASE_W/2) < 0.02:
                        baseEdges.add(edge)

        if baseEdges.count > 0:
            try:
                fInput2 = fillets.createInput()
                fInput2.addConstantRadiusEdgeSet(baseEdges,
                    adsk.core.ValueInput.createByReal(CR), True)
                fillets.add(fInput2)
            except:
                pass

        ui.messageBox('ESP32 Stand created!\n\n' +
                      'Height: 178mm\n' +
                      'Base: 50x50x4mm\n' +
                      'ESP32 at 150mm (15cm)\n' +
                      'Hollow rod, open-front holder\n' +
                      'USB slot on left side\n\n' +
                      'Add fillets manually if needed:\n' +
                      'Select edges -> Modify -> Fillet -> 2mm')

    except:
        if ui:
            ui.messageBox('Error:\n{}'.format(traceback.format_exc()))
