#version 450

layout (location = 0) out vec3 outColor;

void main()
{
    // position of triangle vertices
    const vec3 positions[3] = vec3[3](
        vec3( 0.5,  0.5,  0.0),
        vec3(-0.5,  0.5,  0.0),
        vec3( 0.0, -0.5,  0.0)
    );

    // color array for the vertices
    const vec3 colors[3] = vec3[3](
        vec3( 1.0,  0.0,  0.0), // red
        vec3( 0.0,  1.0,  0.0), // green
        vec3( 0.0,  0.0,  1.0)  // blue
    );

    // save output
    outColor= colors[gl_VertexIndex];
    gl_Position = vec4(positions[gl_VertexIndex], 1.0);
}