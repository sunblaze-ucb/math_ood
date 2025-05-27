import random
from typing import Optional
from translate_utils import Command, invert_pi_expression, format_vertex_list
from passive_voice_templates import format_polygon_command, format_regular_command

def translate_problem(contents: str, answer: str) -> Optional[str]:
    """
    Mechanically translate a geometric construction in command language to natural language.
    
    Args:
        contents: String containing the commands in the formal language
    
    Returns:
        A string with the natural language translation of the construction
    """
    already_obscured_angle = False
    new_answer = None
    stats = {
        "num_commands": 0,
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
    num_commands = 0
    # Initialize the result list
    result_lines = []
    command_constructed_by = {}
    # Some things, such as consts, phrases like "angle ABC" or "circle ABC" or "segment AB" should be referred to as such and not given a new identifier in natural language.
    idents = {}
    # Process each line
    for line in contents.strip().split('\n'):
        # Skip empty lines and comments
        if not line.strip() or line.strip().startswith('#'):
            continue

        # this has to go at the very beginning because const command syntax is different from every other: contains no ':'
        if "const" in line:
            output = line.strip().split(' ')[-1]
            value = line.strip().split(' ')[2]
            idents[output] = value
        
        num_commands += 1

        # Split the line into command and arguments
        parts = line.split(':')
        if len(parts) < 2:
            continue
        cmd = parts[0].strip()
        # Split arguments into inputs and outputs
        args_parts = parts[1].split('->')
        if len(args_parts) < 2:
            continue
        
        # gather construction stats here, so it's all in one place
        if "point" in cmd:
            stats["num_raw_points_constructed"] += 1 # including point_at_distance*, not including intersection points
        if "circle" in cmd: # including incircle, circumcircle
            stats["num_circles_constructed"] += 1
        if "polygon" in cmd: # including rotate_polygon_about_center
            stats["num_polygons_constructed"] += 1
        if ("line" in cmd and "bisector" not in cmd) or "polar_pc" in cmd or "tangent_pc" in cmd: # this includes orthogonal lines, but not angle_bisector
            stats["num_lines_constructed"] += 1
        if "segment" in cmd or "diagonal_p" in cmd:
            stats["num_segments_constructed"] += 1
        if "angle" in cmd:
            stats["num_angles_constructed"] += 1
        if "intersect" in cmd:
            stats["num_intersections"] += 1
        if "mirror" in cmd:
            stats["num_reflections"] += 1
        if "midpoint" in cmd or "line_bisector" in cmd:
            stats["num_midpoints_constructed"] += 1
        if "rotate" in cmd: # including rotate_polygon_about_center
            stats["num_rotations"] += 1
        if "angular_bisector" in cmd:
            stats["num_angle_bisections"] += 1
        if "triangle" in cmd:
            stats["num_triangles_constructed"] += 1
        if cmd.endswith("_t"):
            stats["num_special_triangle_ops"] += 1
            

        raw_inputs = [arg.strip() for arg in args_parts[0].split() if arg.strip()]
        outputs = [arg.strip() for arg in args_parts[1].split() if arg.strip()]
        inputs = [idents[arg] if arg in idents else arg for arg in raw_inputs]
        for output in outputs:
            command_constructed_by[output] = Command(cmd, inputs)
        

        if cmd == "segment_pp":
            idents[outputs[0]] = inputs[0] + inputs[1]
            continue
        if cmd == "angle_ppp":
            idents[outputs[0]] = inputs[0] + inputs[1] + inputs[2]
            continue
        if cmd == "triangle_ppp":
            idents[outputs[0]] = inputs[0] + inputs[1] + inputs[2]
            continue
        if cmd == "diagonal_p":
            idents[outputs[0]] = "diagonal " + inputs[0] + inputs[1]
            continue

        if cmd == "polygon_from_center_and_circumradius":
            
            translated_line = random.choice(polygon_templates)
            result_lines.append(translated_line)
            continue

        if cmd == "rotate_polygon_about_center_by_equivalent_angle":
            polygon_templates = [
                f"The polygon {inputs[0]} is rotated about its center counterclockwise by an angle equivalent to the measure of angle {inputs[1]}. The resulting polygon is called {outputs[-1]}, and the corresponding vertices {format_vertex_list(outputs[:-1])}.",
                f"The polygon {outputs[-1]} is created by rotating the polygon {inputs[0]} counterclockwise about its center by the same angle as the measure of angle {inputs[1]}. The vertices of the resulting polygon {format_vertex_list(outputs[:-1])} are labeled, in correspondence to the vertices of {inputs[0]}.",
                f"The polygon {outputs[-1]} is drawn by rotating {inputs[0]} counterclockwise about its center by an angle equal to the measure of angle {inputs[1]}. The vertices of the resulting polygon {format_vertex_list(outputs[:-1])} are labeled, in correspondence to the vertices of {inputs[0]}."
            ]
            translated_line = random.choice(polygon_templates)
            result_lines.append(translated_line)
            continue

        if cmd == "rotate_polygon_about_center":
            if not already_obscured_angle:
                new_answer = inputs[1]
                inputs[1] = "\\alpha"
                already_obscured_angle = True
            else: 
                inputs[1] = invert_pi_expression(inputs[1])
            polygon_templates = [
                f"The polygon {inputs[0]} is rotated about its center counterclockwise by an angle of {inputs[1]} radians. The resulting polygon is called {outputs[-1]}, and the corresponding vertices {format_vertex_list(outputs[:-1])}.",
                f"The polygon {outputs[-1]} is created by rotating the polygon {inputs[0]} counterclockwise about its center by an angle of {inputs[1]} radians. The vertices of the resulting polygon {format_vertex_list(outputs[:-1])} are labeled, in correspondence to the vertices of {inputs[0]}.",
                f"The polygon {outputs[-1]} is drawn by rotating {inputs[0]} counterclockwise about its center by an angle of {inputs[1]} radians. The vertices of the resulting polygon {format_vertex_list(outputs[:-1])} are labeled, in correspondence to the vertices of {inputs[0]}."
            ]
            translated_line = random.choice(polygon_templates)
            result_lines.append(translated_line)
            continue
        
        if cmd == "measure":
            if "polygon" in command_constructed_by[raw_inputs[0]].name:
                measure_templates = [
                    f"Finally, the area of polygon {inputs[0]}, to four decimal places, is found to be {answer}.",
                    f"Afterwards, the area of polygon {inputs[0]}, to four decimal places, equals {answer}."
                ]
                translated_line = random.choice(measure_templates)
                result_lines.append(translated_line)
                stats["measure_type"] = "area"
                continue

            if "angle_ppp" in command_constructed_by[raw_inputs[0]].name:
                measure_templates = [
                    f"Finally, the measure of angle {inputs[0]}, in radians, is found to be {answer}, to four decimal places.",
                    f"Afterwards, the measure of angle {inputs[0]}, in radians, equals {answer}, to four decimal places."
                ]
                translated_line = random.choice(measure_templates)
                result_lines.append(translated_line)
                stats["measure_type"] = "angle"
                continue
            if "segment" in command_constructed_by[raw_inputs[0]].name:
                measure_templates = [
                    f"Finally, the length of segment {inputs[0]} is found to be {answer}, to four decimal places.",
                    f"Afterwards, the length of segment {inputs[0]} equals {answer}, to four decimal places."
                ]
                translated_line = random.choice(measure_templates)
                result_lines.append(translated_line)
                stats["measure_type"] = "segment_length"
                continue
            # in the following cases, we are actually measuring a Measure that was constructed by the previous command
            if "distance_pp" in command_constructed_by[raw_inputs[0]].name:
                result_lines.pop()
                last_line_inputs = command_constructed_by[raw_inputs[0]].inputs
                measure_templates = [
                    f"Finally, the distance between points {last_line_inputs[0]} and {last_line_inputs[1]} is found to be {answer}, to four decimal places.",
                    f"Afterwards, the distance between points {last_line_inputs[0]} and {last_line_inputs[1]} equals {answer}, to four decimal places."
                ]
                translated_line = random.choice(measure_templates)
                result_lines.append(translated_line)
                stats["measure_type"] = "two_points_distance"
                continue
            if "radius_c" in command_constructed_by[raw_inputs[0]].name:
                result_lines.pop()
                last_line_inputs = command_constructed_by[raw_inputs[0]].inputs
                measure_templates = [
                    f"Finally, the radius of circle {last_line_inputs[0]} is found to be {answer}, to four decimal places.",
                    f"Afterwards, the radius of circle {last_line_inputs[0]} equals {answer}, to four decimal places."
                ]
                translated_line = random.choice(measure_templates)
                result_lines.append(translated_line)
                stats["measure_type"] = "circle_radius"
                continue
            if "area_P" in command_constructed_by[raw_inputs[0]].name:
                result_lines.pop()
                last_line_inputs = command_constructed_by[raw_inputs[0]].inputs
                measure_templates = [
                    f"Finally, the area of polygon {last_line_inputs[0]} is found to be {answer}, to four decimal places.",
                    f"Afterwards, the area of polygon {last_line_inputs[0]} equals {answer}, to four decimal places."
                ]
                translated_line = random.choice(measure_templates)
                result_lines.append(translated_line)
                stats["measure_type"] = "area"
                continue

            measure_templates = [
                f"Finally, the value of {inputs[0]} is found to be {answer}, to four decimal places.",
                f"Afterwards, the value of {inputs[0]} equals {answer}, to four decimal places."
            ]

            translated_line = random.choice(measure_templates)
            result_lines.append(translated_line) 
            continue
        # Get the template for this command
        template_options = command_templates.get(cmd)
        if not template_options:
            # Skip unknown commands
            # continue
            print(f"Unknown command: {cmd}")
            raise # for now

        # Randomly select one of the template options
        template = random.choice(template_options)

        # Format the template differently based on the number of outputs
        try:
            # Commands with variable outputs (like intersections)
            if len(outputs) > 1 and cmd.startswith(("intersect_", "tangent_")):
                # Handle multiple output points
                if len(outputs) == 2:
                    output_str = f"{outputs[0]} and {outputs[1]}"
                else:
                    output_str = ", ".join(outputs[:-1]) + ", and " + outputs[-1]
                
                # Modify template format for multiple outputs
                if "point" in template:
                    template = template.replace("point {0}", "points " + output_str)
                elif "line" in template:
                    template = template.replace("line {0}", "lines " + output_str)
                elif "bisector" in template:
                    template = template.replace("bisector(s) {0}", "bisectors " + output_str)
                template = template.replace("{1}", "{0}").replace("{2}", "{1}")
                # Apply the template with inputs only
                translated_line = template.format(*inputs)
            else:
                # Standard case - single output or non-intersection commands
                format_args = outputs + inputs
                translated_line = template.format(*format_args)
                
            result_lines.append(translated_line)
        except Exception as e:
            # For debugging
            print(f"Error formatting line: {line}")
            print(f"Command: {cmd}")
            print(f"Template: {template}")
            print(f"Format args: {outputs + inputs}")
            print(f"Exception: {e}")
            # continue
            raise # for now
    
    # Due to passive voice, some transitory words:
    for idx, line in enumerate(result_lines):
        if idx == 0:
            continue # don't mess with the first statement, since it doesn't follow any other statements
        if idx == len(result_lines) - 1:
            continue # don't mess with the last statement, since those are built into the measure commands.
        roll = random.random()
        if roll < 0.2:
            # Uncapitalize the first letter of the line after "Then, "
            first_char = result_lines[idx][0].lower()
            result_lines[idx] = "Then, " + first_char + result_lines[idx][1:]
        elif roll < 0.3:
            first_char = result_lines[idx][0].lower()
            result_lines[idx] = "After that, " + first_char + result_lines[idx][1:]
        elif roll < 0.45:
            first_char = result_lines[idx][0].lower()
            result_lines[idx] = "Next, " + first_char + result_lines[idx][1:]
        elif roll < 0.5:
            first_char = result_lines[idx][0].lower()
            result_lines[idx] = "Afterwards, " + first_char + result_lines[idx][1:]
    
    if not result_lines:
        return None, None
    
    result_lines.append("What was the measure of angle \\alpha?")

    stats["num_commands"] = num_commands    
    if new_answer is None:
        raise ValueError("Didn't substitute for missing angle.")

    return result_lines, stats, new_answer