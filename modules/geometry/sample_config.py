basic_commands = [
    'distance_pp',
    'intersect_ll',
    'intersect_ls',
    'intersect_sl',
    'intersect_ss',
    'line_bisector_pp',
    'line_bisector_s',
    'line_pl',
    'line_pp',
    'line_ps',
    'measure',
    'midpoint_pp',
    'midpoint_s',
    'mirror_cl',
    'mirror_cp',
    'mirror_ll',
    'mirror_lp',
    'mirror_pc',
    'mirror_pl',
    'mirror_pp',
    'mirror_ps',
    'orthogonal_line_pl',
    'orthogonal_line_ps',
    'point_',
    'point_at_distance',
    'point_at_distance_along_line',
    'point_l',
    'point_pm',
    'point_s',
    'prove_b',
    'ratio_mm',
    'segment_pp',
    'translate_pv',
    'vector_pp',
    'angle_ppp',
    'angular_bisector_ll',
    'angular_bisector_ppp',
    'angular_bisector_ss',
    'rotate_pAp',
    'rotate_pap',
]

rotate_polygon_commands = [
    'rotate_polygon_about_center',
    'rotate_polygon_about_center_by_equivalent_angle',
]
polygon_commands = [
    'polygon_from_center_and_circumradius',
    'area_P',
    'circumcircle_p',
]

circle_commands = [
    'circle_pm',
    'circle_pp',
    'circle_ppp',
    'circle_pm',
    'center_c',
    'chord_c',
    'intersect_Cl',
    'intersect_cc',
    'intersect_cl',
    'intersect_cs',
    'intersect_lc',
    'point_c',
    'polar_pc',
    'radius_c',
    'tangent_pc',
    'externally_tangent_c',
    'internally_tangent_c',
    'externally_tangent_cc',
]

triangle_commands = [
    'triangle_ppp',
    'centroid_t',
    'circumcenter_t',
    'circumcircle_t',
    'circumradius_t',
    'incenter_t',
    'incircle_t',
    'inradius_t',
    'orthocenter_t',
    'equilateral_triangle',
    
]

def get_commands(command_types):
    commands = basic_commands.copy()
    if "all" in command_types:
        commands.extend(triangle_commands)
        commands.extend(circle_commands)
        commands.extend(angle_commands)
        commands.extend(polygon_commands)
        commands.extend(rotate_polygon_commands)
        return commands
    if "triangle" in command_types:
        commands.extend(triangle_commands)
    if "circle" in command_types:
        commands.extend(circle_commands)
    if "angle" in command_types:
        commands.extend(angle_commands)
    if "polygon" in command_types:
        commands.extend(polygon_commands)
    if "angle" in command_types and "polygon" in command_types:
        commands.extend(rotate_polygon_commands)
    return commands