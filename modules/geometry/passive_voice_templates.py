import random
from typing import Dict
from random_constr import Element
from translate_utils import invert_pi_expression

# These are written in the passive voice, i.e, they describe the action that has been performed.
# This is how AIME problems are worded.
def format_regular_command(command, idents: Dict[Element, str]):
    output_labels = [e.label for e in command.output_elements]
    input_labels = [idents[e] if e in idents else e.label for e in command.input_elements]
    if command.name == "point_c":
        return random.choice([
            f"Point {output_labels[0]} lies on circle {input_labels[0]}.",
            f"Point {output_labels[0]} is on circle {input_labels[0]}.",
        ])
    if command.name == "point_l":
        return random.choice([
            f"Point {output_labels[0]} lies on line {input_labels[0]}.",
            f"Point {output_labels[0]} is on line {input_labels[0]}.",
        ])
    if command.name == "point_s":
        return random.choice([
            f"Point {output_labels[0]} is on segment {input_labels[0]}.",
            f"Point {output_labels[0]} lies on segment {input_labels[0]}.",
        ])
    if command.name == "point_pm":
        existing_point = command.input_elements[0]
        if existing_point.command.name == "point_":
            return random.choice([
                f"Let {input_labels[0]}{output_labels[0]} = {input_labels[1]}.",
                f"{input_labels[0]}{output_labels[0]} = {input_labels[1]}.",
            ])
        else: 
            return random.choice([
                f"{output_labels[0]} is constructed so that {input_labels[0]}{output_labels[0]} = {input_labels[1]}.",
                f"{output_labels[0]} is placed {input_labels[1]} units away from {input_labels[0]}.",
            ])
    if command.name == "point_at_distance":
        return random.choice([
            f"Point {output_labels[0]} is at distance {input_labels[1]} from point {input_labels[0]}.",
            f"The distance from point {input_labels[0]} to point {output_labels[0]} is {input_labels[1]}.",
        ])
    if command.name == "point_at_distance_along_line":
        return random.choice([
            f"Point {output_labels[0]} lies on line {input_labels[0]} at distance {input_labels[2]} from point {input_labels[1]}.",
            f"Point {output_labels[0]} is on line {input_labels[0]} with |{output_labels[0]}{input_labels[1]}| = {input_labels[2]}.",
        ])
    
    # Line creation
    if command.name == "line_pp":
        return random.choice([
            f"Line {output_labels[0]} passes through points {input_labels[0]} and {input_labels[1]}.",
            f"Let {output_labels[0]} be the line through points {input_labels[0]} and {input_labels[1]}.",
        ])
    if command.name == "line_pl":
        return random.choice([
            f"Line {output_labels[0]} passes through point {input_labels[0]} parallel to line {input_labels[1]}.",
            f"Let {output_labels[0]} be the line through point {input_labels[0]} parallel to line {input_labels[1]}.",
        ])
    if command.name == "line_pr":
        return random.choice([
            f"Line {output_labels[0]} passes through point {input_labels[0]} parallel to ray {input_labels[1]}.",
            f"Let {output_labels[0]} be the line through point {input_labels[0]} parallel to ray {input_labels[1]}.",
        ])
    if command.name == "line_ps":
        return random.choice([
            f"Line {output_labels[0]} passes through point {input_labels[0]} parallel to segment {input_labels[1]}.",
            f"Let {output_labels[0]} be the line through point {input_labels[0]} parallel to segment {input_labels[1]}.",
        ])
    if command.name == "line_bisector_pp":
        return random.choice([
            f"Line {output_labels[0]} is the perpendicular bisector of segment {input_labels[0]}{input_labels[1]}.",
            f"Let {output_labels[0]} be the perpendicular bisector of segment {input_labels[0]}{input_labels[1]}.",
        ])
    if command.name == "line_bisector_s":
        return random.choice([
            f"Line {output_labels[0]} is the perpendicular bisector of segment {input_labels[0]}.",
            f"Let {output_labels[0]} be the perpendicular bisector of segment {input_labels[0]}.",
        ])
    
    # Orthogonal lines
    if command.name == "orthogonal_line_pl":
        return random.choice([
            f"Line {output_labels[0]} passes through point {input_labels[0]} perpendicular to line {input_labels[1]}.",
            f"Let {output_labels[0]} be the line through point {input_labels[0]} perpendicular to line {input_labels[1]}.",
        ])
    if command.name == "orthogonal_line_pr":
        return random.choice([
            f"Line {output_labels[0]} passes through point {input_labels[0]} perpendicular to ray {input_labels[1]}.",
            f"Let {output_labels[0]} be the line through point {input_labels[0]} perpendicular to ray {input_labels[1]}.",
        ])
    if command.name == "orthogonal_line_ps":
        return random.choice([
            f"Line {output_labels[0]} passes through point {input_labels[0]} perpendicular to segment {input_labels[1]}.",
            f"Let {output_labels[0]} be the line through point {input_labels[0]} perpendicular to segment {input_labels[1]}.",
        ])
    
    # Angular bisectors
    if command.name == "angular_bisector_ll":
        return random.choice([
            f"Lines {output_labels[0]} and {output_labels[1]} are the angle bisectors of lines {input_labels[0]} and {input_labels[1]}.",
            f"Let {output_labels[0]} and {output_labels[1]} be the angle bisectors of lines {input_labels[0]} and {input_labels[1]}.",
        ])
    if command.name == "angular_bisector_ppp":
        return random.choice([
            f"Line {output_labels[0]} is the angle bisector of angle {input_labels[0]}{input_labels[1]}{input_labels[2]}.",
            f"Let {output_labels[0]} be the angle bisector of angle {input_labels[0]}{input_labels[1]}{input_labels[2]}.",
        ])
    if command.name == "angular_bisector_ss":
        return random.choice([
            f"Lines {output_labels[0]} and {output_labels[1]} are the angle bisectors of segments {input_labels[0]} and {input_labels[1]}.",
            f"Let {output_labels[0]} and {output_labels[1]} be the angle bisectors of segments {input_labels[0]} and {input_labels[1]}.",
        ])
    
    # Intersections
    if 'intersect' in command.name:
        if len(command.output_elements) > 1:
            output_str = 's {output_labels[0]} and {output_labels[1]} are'
        else:
            output_str = f' {output_labels[0]} is'
        if command.name == "intersect_lc":
            return random.choice([
                f"Point{output_str} the intersection of line {input_labels[0]} and circle {input_labels[1]}.",
                f"Point{output_str} the intersection of line {input_labels[0]} and circle {input_labels[1]}.",
            ])
        if command.name == "intersect_cs":
            return random.choice([
                f"Point{output_str} the intersection(s) of circle {input_labels[0]} and segment {input_labels[1]}.",
                f"Circle {input_labels[0]} and segment {input_labels[1]} intersect at point{output_str}.",
            ])
        if command.name == "intersect_cc":
            return random.choice([
                f"Point{output_str} the intersection(s) of circles {input_labels[0]} and {input_labels[1]}.",
                f"Circles {input_labels[0]} and {input_labels[1]} intersect at point{output_str}.",
            ])
        if command.name == "intersect_cl":
            return random.choice([
                f"Point{output_str} the intersection(s) of circle {input_labels[0]} and line {input_labels[1]}.",
                f"Circle {input_labels[0]} and line {input_labels[1]} intersect at point{output_str}.",
            ])
    if command.name == "intersect_ll":
        return random.choice([
            f"Point {output_labels[0]} is the intersection of lines {input_labels[0]} and {input_labels[1]}.",
            f"Lines {input_labels[0]} and {input_labels[1]} intersect at point {output_labels[0]}.",
        ])
    if command.name == "intersect_lr":
        return random.choice([
            f"Point {output_labels[0]} is the intersection of line {input_labels[0]} and ray {input_labels[1]}.",
            f"Line {input_labels[0]} and ray {input_labels[1]} intersect at point {output_labels[0]}.",
        ])
    if command.name == "intersect_ls":
        return random.choice([
            f"Point {output_labels[0]} is the intersection of line {input_labels[0]} and segment {input_labels[1]}.",
            f"Line {input_labels[0]} and segment {input_labels[1]} intersect at point {output_labels[0]}.",
        ])
    if command.name == "intersect_rl":
        return random.choice([
            f"Point {output_labels[0]} is the intersection of ray {input_labels[0]} and line {input_labels[1]}.",
            f"Ray {input_labels[0]} and line {input_labels[1]} intersect at point {output_labels[0]}.",
        ])
    if command.name == "intersect_rr":
        return random.choice([
            f"Point {output_labels[0]} is the intersection of rays {input_labels[0]} and {input_labels[1]}.",
            f"Rays {input_labels[0]} and {input_labels[1]} intersect at point {output_labels[0]}.",
        ])
    if command.name == "intersect_rs":
        return random.choice([
            f"Point {output_labels[0]} is the intersection of ray {input_labels[0]} and segment {input_labels[1]}.",
            f"Ray {input_labels[0]} and segment {input_labels[1]} intersect at point {output_labels[0]}.",
        ])
    if command.name == "intersect_sl":
        return random.choice([
            f"Point {output_labels[0]} is the intersection of segment {input_labels[0]} and line {input_labels[1]}.",
            f"Segment {input_labels[0]} and line {input_labels[1]} intersect at point {output_labels[0]}.",
        ])
    if command.name == "intersect_sr":
        return random.choice([
            f"Point {output_labels[0]} is the intersection of segment {input_labels[0]} and ray {input_labels[1]}.",
            f"Segment {input_labels[0]} and ray {input_labels[1]} intersect at point {output_labels[0]}.",
        ])
    if command.name == "intersect_ss":
        return random.choice([
            f"Point {output_labels[0]} is the intersection of segments {input_labels[0]} and {input_labels[1]}.",
            f"Segments {input_labels[0]} and {input_labels[1]} intersect at point {output_labels[0]}.",
        ])
    
    # Circles
    if command.name == "circle_pp":
        return random.choice([
            f"Circle {output_labels[0]} has center {input_labels[0]} and passes through point {input_labels[1]}.",
            f"Let {output_labels[0]} be the circle with center {input_labels[0]} passing through point {input_labels[1]}.",
        ])
    if command.name == "circle_ppp":
        return random.choice([
            f"Circle {output_labels[0]} passes through points {input_labels[0]}, {input_labels[1]}, and {input_labels[2]}.",
            f"Let {output_labels[0]} be the circle through points {input_labels[0]}, {input_labels[1]}, and {input_labels[2]}.",
        ])
    if command.name == "circle_pm":
        return random.choice([
            f"Circle {output_labels[0]} has center {input_labels[0]} and radius {input_labels[1]}.",
            f"Let {output_labels[0]} be the circle with center {input_labels[0]} and radius {input_labels[1]}.",
        ])
    if command.name == "circle_ps":
        return random.choice([
            f"Circle {output_labels[0]} has center {input_labels[0]} and radius equal to length of segment {input_labels[1]}.",
            f"Let {output_labels[0]} be the circle with center {input_labels[0]} and radius |{input_labels[1]}|.",
        ])
    if command.name == "center_c":
        return random.choice([
            f"Point {output_labels[0]} is the center of circle {input_labels[0]}.",
            f"Let {output_labels[0]} be the center of circle {input_labels[0]}.",
        ])
    
    # Segments
    if command.name == "segment_pp":
        return random.choice([
            f"Segment {output_labels[0]} connects points {input_labels[0]} and {input_labels[1]}.",
            f"Let {output_labels[0]} be the segment from point {input_labels[0]} to point {input_labels[1]}.",
        ])
    
    # Midpoints
    if command.name == "midpoint_pp":
        return random.choice([
            f"Point {output_labels[0]} is the midpoint of segment {input_labels[0]}{input_labels[1]}.",
            f"Let {output_labels[0]} be the midpoint of segment {input_labels[0]}{input_labels[1]}.",
        ])
    
    # Midpoints
    if command.name == "midpoint_s":
        return random.choice([
            f"Point {output_labels[0]} is the midpoint of segment {input_labels[0]}.",
            f"Let {output_labels[0]} be the midpoint of segment {input_labels[0]}.",
        ])
    
    # Reflections/Mirrors
    if command.name == "mirror_pp":
        return random.choice([
            f"Point {output_labels[0]} is the reflection of point {input_labels[0]} across point {input_labels[1]}.",
            f"Let {output_labels[0]} be the reflection of point {input_labels[0]} across point {input_labels[1]}.",
        ])
    if command.name == "mirror_pl":
        return random.choice([
            f"Point {output_labels[0]} is the reflection of point {input_labels[0]} across line {input_labels[1]}.",
        ])
    if command.name == "mirror_ps":
        return random.choice([
            f"Point {output_labels[0]} is the reflection of point {input_labels[0]} across segment {input_labels[1]}.",
        ])
    if command.name == "mirror_pc":
        return random.choice([
            f"Point {output_labels[0]} is the inversion of point {input_labels[0]} with respect to circle {input_labels[1]}.",
            f"Let {output_labels[0]} be the inversion of point {input_labels[0]} in circle {input_labels[1]}.",
        ])
    if command.name == "mirror_lp":
        return random.choice([
            f"Line {output_labels[0]} is the reflection of line {input_labels[0]} across point {input_labels[1]}.",
            f"Let {output_labels[0]} be the reflection of line {input_labels[0]} through point {input_labels[1]}.",
        ])
    if command.name == "mirror_ll":
        return random.choice([
            f"Line {output_labels[0]} is the reflection of line {input_labels[0]} across line {input_labels[1]}.",
            f"Let {output_labels[0]} be the reflection of line {input_labels[0]} across line {input_labels[1]}.",
        ])
    if command.name == "mirror_cl":
        return random.choice([
            f"Circle {output_labels[0]} is the reflection of circle {input_labels[0]} across line {input_labels[1]}.",
            f"Let {output_labels[0]} be the reflection of circle {input_labels[0]} across line {input_labels[1]}.",
        ])
    if command.name == "mirror_cp":
        return random.choice([
            f"Circle {output_labels[0]} is the reflection of circle {input_labels[0]} across point {input_labels[1]}.",
            f"Let {output_labels[0]} be the reflection of circle {input_labels[0]} through point {input_labels[1]}.",
        ])
    
    # Angles
    if command.name == "angle_ppp":
        return random.choice([
            f"Let {output_labels[0]} be the measure of angle {input_labels[0]}{input_labels[1]}{input_labels[2]}.",
            f"Angle {input_labels[0]}{input_labels[1]}{input_labels[2]} has measure {output_labels[0]}.",
        ])
    
    # these actually should not matter; since these compute a Measure the only thing that can follow them is a "measure" command, and that will handle the actual translation of these commands
    if command.name == "distance_pp":
        return random.choice([
            f"|{input_labels[0]}{input_labels[1]}| = {output_labels[0]}.",
            f"Let {output_labels[0]} be the distance between points {input_labels[0]} and {input_labels[1]}.",
        ])
    if command.name == "radius_c":
        return random.choice([
            f"The radius of circle {input_labels[0]} is {output_labels[0]}.",
            f"Let {output_labels[0]} be the radius of circle {input_labels[0]}.",
        ])
    if command.name == "area_P":
        return random.choice([
            f"The area of polygon {input_labels[0]} is {output_labels[0]}.",
            f"Let {output_labels[0]} be the area of polygon {input_labels[0]}.",
        ])
    
    # Ray
    if command.name == "ray_pp":
        return random.choice([
            f"Ray {output_labels[0]} starts at point {input_labels[0]} and passes through point {input_labels[1]}.",
            f"Let {output_labels[0]} be the ray from point {input_labels[0]} through point {input_labels[1]}.",
        ])
    
    # Rotations
    if command.name == "rotate_pap":
        return random.choice([
            f"Point {output_labels[0]} is obtained by rotating point {input_labels[0]} by {invert_pi_expression(input_labels[1])} radians about point {input_labels[2]}.",
            f"Let {output_labels[0]} be the rotation of point {input_labels[0]} by {invert_pi_expression(input_labels[1])} radians around point {input_labels[2]}.",
        ])
    if command.name == "rotate_pAp":
        return random.choice([
            f"Point {output_labels[0]} is obtained by rotating point {input_labels[0]} by angle {invert_pi_expression(input_labels[1])} about point {input_labels[2]}.",
            f"Let {output_labels[0]} be the rotation of point {input_labels[0]} by angle {invert_pi_expression(input_labels[1])} around point {input_labels[2]}.",
        ])
    
    # Vectors
    if command.name == "vector_pp":
        return random.choice([
            f"Let {output_labels[0]} be the vector from point {input_labels[0]} to point {input_labels[1]}.",
        ])
    if command.name == "translate_pv":
        return random.choice([
            f"Point {output_labels[0]} is the translation of point {input_labels[0]} by vector {input_labels[1]}.",
            f"Let {output_labels[0]} be point {input_labels[0]} translated by vector {input_labels[1]}.",
        ])
    
    # Tangents
    if command.name == "tangent_pc":
        return random.choice([
            f"Line {output_labels[0]} is the tangent from point {input_labels[0]} to circle {input_labels[1]}.",
            f"Let {output_labels[0]} be the tangent line from point {input_labels[0]} to circle {input_labels[1]}.",
        ])
    if command.name == "polar_pc":
        return random.choice([
            f"Line {output_labels[0]} is the polar of point {input_labels[0]} with respect to circle {input_labels[1]}.",
            f"Let {output_labels[0]} be the polar line of point {input_labels[0]} relative to circle {input_labels[1]}.",
        ])
    
    # Triangle-related commands
    if command.name == "triangle_ppp":
        return random.choice([
            f"Triangle {output_labels[0]} has vertices {input_labels[0]}, {input_labels[1]}, and {input_labels[2]}.",
            f"Let {output_labels[0]} be triangle {input_labels[0]}{input_labels[1]}{input_labels[2]}.",
        ])
    if command.name == "circumcircle_t":
        return random.choice([
            f"Circle {output_labels[0]} is the circumcircle of triangle {input_labels[0]}.",
            f"Let {output_labels[0]} be the circumcircle of triangle {input_labels[0]}.",
        ])
    if command.name == "circumcenter_t":
        return random.choice([
            f"Point {output_labels[0]} is the circumcenter of triangle {input_labels[0]}.",
            f"Let {output_labels[0]} be the circumcenter of triangle {input_labels[0]}.",
        ])
    if command.name == "circumradius_t":
        return random.choice([
            f"The circumradius of triangle {input_labels[0]} is {output_labels[0]}.",
            f"Let {output_labels[0]} be the circumradius of triangle {input_labels[0]}.",
        ])
    if command.name == "centroid_t":
        return random.choice([
            f"Point {output_labels[0]} is the centroid of triangle {input_labels[0]}.",
            f"Let {output_labels[0]} be the centroid of triangle {input_labels[0]}.",
        ])
    if command.name == "orthocenter_t":
        return random.choice([
            f"Point {output_labels[0]} is the orthocenter of triangle {input_labels[0]}.",
            f"Let {output_labels[0]} be the orthocenter of triangle {input_labels[0]}.",
        ])
    if command.name == "incircle_t":
        return random.choice([
            f"Circle {output_labels[0]} is the incircle of triangle {input_labels[0]}.",
            f"Let {output_labels[0]} be the incircle of triangle {input_labels[0]}.",
        ])
    if command.name == "incenter_t":
        return random.choice([
            f"Point {output_labels[0]} is the incenter of triangle {input_labels[0]}.",
            f"Let {output_labels[0]} be the incenter of triangle {input_labels[0]}.",
        ])
    if command.name == "inradius_t":
        return random.choice([
            f"The inradius of triangle {input_labels[0]} is {output_labels[0]}.",
            f"Let {output_labels[0]} be the inradius of triangle {input_labels[0]}.",
        ])
    if command.name == "circumcircle_p":
        return random.choice([
            f"Let {output_labels[0]} be the circumcircle of polygon {input_labels[0]}.",
        ])
    if command.name == "externally_tangent_c":
        return random.choice([
            f"{output_labels[1]} has center {output_labels[0]} and radius {input_labels[0]} and is externally tangent to circle {input_labels[1]}.",
        ])
    if command.name == "internally_tangent_c":
        return random.choice([
            f"{output_labels[1]} has center {output_labels[0]} and radius {input_labels[0]} and is internally tangent to circle {input_labels[1]}.",
        ])
    if command.name == "externally_tangent_cc":
        return random.choice([
            f"{output_labels[1]} has center {output_labels[0]} and radius {input_labels[0]} and is externally tangent to circles {input_labels[1]} and {input_labels[2]}.",
        ])
    if command.name == "chord_c":
        idents[command.output_elements[2]] = f"{output_labels[0]}{output_labels[1]}"
        return random.choice([
            f"Chord {output_labels[0]}{output_labels[1]} of circle {input_labels[1]} has length {input_labels[0]}.",
        ])
    raise Exception(f"Unknown command: {command.name}")
    

def format_polygon_command(command, idents: Dict[Element, str]):
    output_labels = [e.label for e in command.output_elements]
    input_labels = [idents[e] if e in idents else e.label for e in command.input_elements]
    if command.name == "polygon_from_center_and_circumradius":
        num_sides = input_labels[0]
        center = input_labels[1]
        circumradius = input_labels[2]
        polygon_out = output_labels[-1]
        vertices = output_labels[:-1]
        vertices_str = ''.join(vertices)
        polygon_templates = [
                f"Let {vertices_str} be a regular {num_sides}-gon with center {center} and circumradius {circumradius}.",
                f"{vertices_str} is a regular {num_sides}-gon with center {center} and circumradius {circumradius}.",
        ]
    if command.name == "rotate_polygon_about_center_by_equivalent_angle":
        polygon_in = input_labels[0]
        angle = input_labels[1]
        polygon_out = output_labels[-1]
        vertices = output_labels[:-1]
        vertices_str = ''.join(vertices)
        polygon_templates = [
            f"{vertices_str} is {polygon_in} rotated counterclockwise about its center by angle {angle}.",
            f"{vertices_str} is obtained by rotating {polygon_in} counterclockwise about its center by angle {angle}.",
        ]
    if command.name == "rotate_polygon_about_center":
        polygon_in = input_labels[0]
        angle = invert_pi_expression(input_labels[1])
        vertices = output_labels[:-1]
        
        polygon_templates = [
            f"{vertices_str} is {polygon_in} rotated counterclockwise about its center by {angle} radians.",
            f"{vertices_str} is obtained by rotating {polygon_in} counterclockwise about its center by {angle} radians.",
        ]
    template = random.choice(polygon_templates)
    # yes, it's extremely weird to describe a square in terms of its circumradius, but at least it is logically unambiguous.
    if num_sides == 4:
        template = template.replace("regular 4-gon", "square")
    if num_sides == 6:
        template = template.replace("6-gon", "hexagon")
    if num_sides == 8:
        template = template.replace("8-gon", "octagon")
    if num_sides == 12:
        template = template.replace("12-gon", "dodecagon")
    idents[command.output_elements[-1]] = vertices_str
    return template
        
        
