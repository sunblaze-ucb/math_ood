import pdb
import random
from typing import Optional, Dict, Any
from random_constr import Construction, Element
from passive_voice_templates import format_polygon_command, format_regular_command

def translate_problem(contents: str, answer: Optional[str] = None) -> Optional[str]:
    """
    Mechanically translate a geometric construction in command language to natural language.
    
    Args:
        contents: String containing the commands in the formal language
    
    Returns:
        A string with the natural language translation of the construction
    """
    construction = Construction()
    construction.load(file_contents=contents)
    stats = {
        "num_commands": len(construction.nc_commands),
        "measure_type": "other",
        "num_raw_points_constructed": 0,
        "num_lines_constructed": 0,
        "num_circles_constructed": 0,
        "num_angles_constructed": 0,
        "num_segments_constructed": 0,
        "num_triangles_constructed": 0,
        "num_polygons_constructed": 0,
        "num_midpoints_constructed": 0,
        "num_rotations": 0,
        "num_reflections": 0,
        "num_intersections": 0, 
        "num_angle_bisections": 0,
        "num_special_triangle_ops": 0,
    }
    # Initialize the result list
    result_lines = []
    # Some things, such as consts, phrases like "angle ABC" or "circle ABC" or "segment AB" should be referred to as such and not given a new identifier in natural language.
    idents: Dict[Element, Any] = {}

    for command in construction.const_commands:
        idents[command.element] = command.element.data

    for command in construction.nc_commands:
        # gather construction stats here, so it's all in one place
        if "point" in command.name:
            stats["num_raw_points_constructed"] += 1 # including point_at_distance*, not including intersection points
        if "circle" in command.name or command.name.endswith("_c") or command.name.endswith("_cc"): # including incircle, circumcircle
            stats["num_circles_constructed"] += 1
        if "polygon" in command.name: # including rotate_polygon_about_center
            stats["num_polygons_constructed"] += 1
        if ("line" in command.name and "bisector" not in command.name) or "polar_pc" in command.name or "tangent_pc" in command.name: # this includes orthogonal lines, but not angle_bisector
            stats["num_lines_constructed"] += 1
        if "segment" in command.name or "diagonal_p" in command.name:
            stats["num_segments_constructed"] += 1
        if "angle" in command.name:
            stats["num_angles_constructed"] += 1
        if "intersect" in command.name:
            stats["num_intersections"] += 1
        if "mirror" in command.name:
            stats["num_reflections"] += 1
        if "midpoint" in command.name or "line_bisector" in command.name:
            stats["num_midpoints_constructed"] += 1
        if "rotate" in command.name: # including rotate_polygon_about_center
            stats["num_rotations"] += 1
        if "angular_bisector" in command.name:
            stats["num_angle_bisections"] += 1
        if "triangle" in command.name:
            stats["num_triangles_constructed"] += 1
        if command.name.endswith("_t"):
            stats["num_special_triangle_ops"] += 1
            
        # don't actually do the translation of these commands; all they do is define something that can be referred to by a new name, later
        # instead just handle the mapping from their "internal" name to what they should be called, e.g, "segment AB", "angle ABC", etc.
        if command.name == "point_":
            continue # try literally doing nothing and see if it's incoherent grammatically
        if command.name == "segment_pp":
            idents[command.output_elements[0]] = command.input_elements[0].label + command.input_elements[1].label
            continue
        if command.name == "angle_ppp":
            idents[command.output_elements[0]] = command.input_elements[0].label + command.input_elements[1].label + command.input_elements[2].label
            continue
        if command.name == "triangle_ppp":
            idents[command.output_elements[0]] = command.input_elements[0].label + command.input_elements[1].label + command.input_elements[2].label
            idents[command.output_elements[1]] = command.input_elements[0].label + command.input_elements[1].label
            idents[command.output_elements[2]] = command.input_elements[1].label + command.input_elements[2].label
            idents[command.output_elements[3]] = command.input_elements[2].label + command.input_elements[0].label
            continue
        if command.name == "equilateral_triangle":
            idents[command.output_elements[0]] = command.output_elements[1].label
            idents[command.output_elements[2]] = command.output_elements[3].label
            idents[command.output_elements[4]] = command.output_elements[0].label + command.output_elements[1].label
            idents[command.output_elements[5]] = command.output_elements[1].label + command.output_elements[2].label
            idents[command.output_elements[6]] = command.output_elements[2].label + command.output_elements[0].label
            continue
        if command.name == "diagonal_p":
            idents[command.output_elements[0]] = command.input_elements[0].label + command.input_elements[1].label
            continue

        if "polygon" in command.name:
            new_line = format_polygon_command(command, idents) # also names the polygon, modifying idents
            result_lines.append(new_line)
            continue

        if len(command.output_elements) > 1:
            if 'intersect' in command.name:
                # fuck it, just remove the extra word here
                result_lines.append(format_regular_command(command, idents).replace(' are.', '.'))
                continue
            elif 'chord_c' in command.name or 'tangent' in command.name:
                result_lines.append(format_regular_command(command, idents))
                continue
            else:
                # give up entirely
                # for now, all non-polygon commands that have multiple outputs are not supported
                # the reason for this is basically that the generator has a consistent way of producing them in a fixed order, but there is no natural translation for this
                # TODO: handle this, ideally by just randomizing the output order of the command earlier in the pipeline, so that we at least get to save problems that actually are solvable despite the ambiguity.
                print("Threw out a problem due to multiple outputs")
                return None, None
        # if no special casing happened up until now, just format the template according to "common" logic
        result_lines.append(format_regular_command(command, idents))


    generating_command = construction.to_measure.command
    # this is the command that generated the object of measure in the first place
    measured_object = idents[construction.to_measure] if construction.to_measure in idents else construction.to_measure.label
    if "polygon" in generating_command.name:
        measure_templates = [
            f"What is the area of polygon {measured_object}?",
            f"Find the area of polygon {measured_object}."
        ]
        stats["measure_type"] = "area"
    elif "angle_ppp" in generating_command.name:
        measure_templates = [
            f"What is the measure of angle {measured_object}, in radians?",
            f"Find the measure of angle {measured_object}, in radians."
        ]
        stats["measure_type"] = "angle"
    elif "diagonal" in generating_command.name:
        measure_templates = [
            f"What is the length of diagonal {measured_object}?",
            f"Find the length of diagonal {measured_object}."
        ]
        stats["measure_type"] = "distance"
    elif "segment" in generating_command.name:
        measure_templates = [
            f"What is the length of segment {measured_object}?",
            f"Find the length of segment {measured_object}."
        ]
        stats["measure_type"] = "distance"
    elif "triangle" in generating_command.name:
        measure_templates = [
            f"What is the area of triangle {measured_object}?",
            f"Find the area of triangle {measured_object}."
        ]
        stats["measure_type"] = "area"
    # in the following cases, we are actually measuring a Measure that was constructed by the previous command
    elif generating_command.name in ("distance_pp", "radius_c", "circumradius_t", "area_P", "inradius_t"):
        last_line_inputs = [idents[x] if x in idents else x.label for x in generating_command.input_elements]
        # remove the last line, which is the command that generated the Measure.
        # Our question will semantically construct the Measure implicitly, so we don't need to mention its construction.
        result_lines.pop()
        if "distance_pp" in generating_command.name:
            measure_templates = [
                f"What is the distance between points {last_line_inputs[0]} and {last_line_inputs[1]}?",
                f"Find the distance between points {last_line_inputs[0]} and {last_line_inputs[1]}."
            ]
            stats["measure_type"] = "distance"
        elif "radius_c" in generating_command.name:
            measure_templates = [
                f"What is the radius of circle {last_line_inputs[0]}?",
                f"Find the radius of circle {last_line_inputs[0]}."
            ]
            stats["measure_type"] = "distance"
        elif "inradius_t" in generating_command.name:
            measure_templates = [
                f"What is the inradius of triangle {last_line_inputs[0]}?",
                f"Find the inradius of triangle {last_line_inputs[0]}."
            ]
            stats["measure_type"] = "triangle_special_property"
        elif "circumradius_t" in generating_command.name:
            measure_templates = [
                f"What is the circumradius of triangle {last_line_inputs[0]}?",
                f"Find the circumradius of triangle {last_line_inputs[0]}."
            ]
            stats["measure_type"] = "triangle_special_property"
        elif "area_P" in generating_command.name:
            measure_templates = [
                f"What is the area of polygon {last_line_inputs[0]}?",
                f"Find the area of polygon {last_line_inputs[0]}."
            ]
            stats["measure_type"] = "area"
    else:
        raise Exception(f"Unknown measure command: {generating_command.name}")
    translated_line = random.choice(measure_templates)
    result_lines.append(translated_line)
    # Combine all translated lines into a coherent problem statement
    if not result_lines:
        return None, None
    result_lines.append("Express your answer as a decimal number rounded to 2 decimal places.")
    return result_lines, stats