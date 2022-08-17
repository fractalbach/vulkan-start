#version 450

#define PI 3.1415926538

layout (location = 0) in vec3 vPosition;
layout (location = 1) in vec3 vNormal;
layout (location = 2) in vec3 vColor;

layout (location = 0) out vec3 outColor;
layout (location = 1) out int time;

layout(push_constant) uniform constants {
    vec4 data;
    mat4 render_matrix;
    int time;
} PushConstants;

void main() {

    const float t = PushConstants.data[0]; // the current time (in ms)
    const int T = 2000; // the period of a cycle
    const float a = mod(t, T);
    const float c = int(a > (T/2)) * (T-a)  + int(a <= (T/2)) * a;
    const float x = c / (T/2);

    const float r = vColor[0];
    const float g = vColor[1];
    const float b = vColor[2];

    const vec3 colors = vec3(
        r*x + g*(1-x),
        g*x + b*(1-x),
        b*x + r*(1-x)
    );

    gl_Position = PushConstants.render_matrix * vec4(vPosition, 1.0);
    outColor = colors;
    time = PushConstants.time;
}