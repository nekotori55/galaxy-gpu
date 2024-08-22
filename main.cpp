#include "vector2.h"
#include <GL/glut.h>
#include <iostream>
#include <vector>

using namespace std;

static GLfloat windowSizeX = 1800.0f;
static GLfloat windowSizeY = 1600.0f;

GLfloat windowWidth;
GLfloat windowHeight;

struct Point
{
    Vector2 pos;      // position
    Vector2 velocity; // velocity

    long int mass = 1e7;

    unsigned char r, g, b, a;
};

vector<Point> points;

const float G = 6.6743e-11;

void ChangeSize(int w, int h)
{
    // glViewport(0, 0, w, h);

    GLfloat aspectRatio;
    if (h == 0)
        h = 1;

    double zoomh = 100;

    glViewport(0 - zoomh, 0 - zoomh, w + zoomh, h + zoomh);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    aspectRatio = (GLfloat)w / (GLfloat)h;
    if (w <= h)
    {
        windowWidth = 100;
        windowHeight = 100 / aspectRatio;
        glOrtho(-100.0, 100.0, -windowHeight, windowHeight, 1.0, -1.0);
    }
    else
    {
        windowWidth = 100 * aspectRatio;
        windowHeight = 100;
        glOrtho(-windowWidth, windowWidth, -100.0, 100.0, 1.0, -1.0);
    }
}

void SetupRC()
{
    glClearColor(0.01f, 0.01f, 0.02f, 1.0f);
    for (size_t i = 0; i < 4000; ++i)
    {
        Point pt;
        pt.pos.x = -50 + (rand() % 100);
        pt.pos.y = -50 + (rand() % 100);

        pt.velocity.x = -5 + (rand() % 10);
        pt.velocity.y = -5 + (rand() % 10);

        pt.velocity = pt.velocity * 0.01;
        // pt.r = rand() % 255;
        // pt.g = rand() % 255;
        // pt.b = rand() % 255;
        pt.r = 255;
        pt.g = 255;
        pt.b = 255;
        pt.a = 255;
        points.push_back(pt);
    }
}

void Tick(float delta = 1)
{
    for (Point &point : points)
    {
        for (Point &point2 : points)
        {
            Vector2 R = point2.pos - point.pos;
            float Rlen = R.len();
            if (Rlen <= 0.01)
            {
                // point.velocity = point2.velocity;
                Rlen = 0.001;
                // continue;
            }
            float acc = point2.mass * G / (Rlen * Rlen);
            point.velocity = point.velocity + R.normalize() * acc;
            // cout << point.velocity.len();
        }
        point.pos = point.pos + (point.velocity * delta);
    }

    glutPostRedisplay();
}

void RenderScene()
{
    glClear(GL_COLOR_BUFFER_BIT);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-50, 50, -50, 50, -1, 1);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // draw
    glColor3ub(255, 255, 255);
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    glVertexPointer(2, GL_FLOAT, sizeof(Point), &points[0].pos.x);
    glColorPointer(4, GL_UNSIGNED_BYTE, sizeof(Point), &points[0].r);
    glPointSize(1.0);
    glDrawArrays(GL_POINTS, 0, points.size());
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);

    glFlush();
    glutSwapBuffers();
}

int oldTimeSinceStart = 0;

void timer(int)
{
    int timeSinceStart = glutGet(GLUT_ELAPSED_TIME);
    int deltaTime = timeSinceStart - oldTimeSinceStart;
    oldTimeSinceStart = timeSinceStart;

    Tick((float)deltaTime / 1000.0);

    glutPostRedisplay();
    glutTimerFunc(1000 / 60, timer, 0);
}

int main(int argc, char *argv[])
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutCreateWindow("galaxy-gpu");
    // glutReshapeWindow(windowSizeX, windowSizeY);
    // glutPositionWindow(100, 80);
    // glutIdleFunc(Tick);
    glutTimerFunc(33, timer, 0);
    // glutMotionFunc(mouseMove);
    // glutPassiveMotionFunc(mousePassive);
    // glutMouseFunc(mouseButton);
    glutReshapeFunc(ChangeSize);
    // glutSpecialFunc(SpecialKeys);
    // glutIdleFunc(timer);
    glutDisplayFunc(RenderScene);
    SetupRC();
    glutMainLoop();
    return 0;
}
