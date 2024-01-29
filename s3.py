import torch
import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *

vertex_shader_code = """
#version 330 core
layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in vec3 position;
void main()
{
    gl_Position = vec4(position, 1.0);
}
"""

fragment_shader_code = """
#version 330 core
out vec4 color;
void main()
{
    color = vec4(0.0, 0.0, 0.0, 1.0);
}
"""


def create_shader(shader_type, source_code):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source_code)
    glCompileShader(shader)
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        raise Exception("Error compiling shader: " + glGetShaderInfoLog(shader).decode())
    return shader


def create_program(vertex_shader_code, fragment_shader_code):
    program = glCreateProgram()
    vertex_shader = create_shader(GL_VERTEX_SHADER, vertex_shader_code)
    fragment_shader = create_shader(GL_FRAGMENT_SHADER, fragment_shader_code)

    glAttachShader(program, vertex_shader)
    glAttachShader(program, fragment_shader)
    glLinkProgram(program)

    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)

    return program


def display():
    glClearColor(0.0, 0.0, 0.0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glUseProgram(program)

    glBegin(GL_TRIANGLES)
    glVertex3f(-0.5, -0.5, 0.0)
    glVertex3f(0.5, -0.5, 0.0)
    glVertex3f(0.0, 0.5, 0.0)
    glEnd()

    glUseProgram(0)

    glutSwapBuffers()


glutInit()
glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
glutCreateWindow(b"OpenGL and Torch Example")

program = create_program(vertex_shader_code, fragment_shader_code)

glutDisplayFunc(display)
glutMainLoop()
