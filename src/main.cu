#include "vector2.h"
#include <GL/glut.h>
#include <iostream>
#include <vector>

#define POINTS_COUNT 10000

#define gpuErrchk(ans)                    \
  {                                       \
    gpuAssert((ans), __FILE__, __LINE__); \
  }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort)
      exit(code);
  }
}

using namespace std;

static GLdouble windowSizeX = 1800.0f;
static GLdouble windowSizeY = 1600.0f;

GLdouble z = 2000000;

GLdouble windowWidth;
GLdouble windowHeight;

struct Point
{
  double2 pos;      // position
  double2 velocity; // velocity

  double mass = 2e32;

  unsigned char r, g, b, a;
};

Point *points;

#define G 6.6743e-11

// Kernel definition
__global__ void add(int n, Point *points)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (size_t i = index; i < n; i += stride)
  {
    for (size_t j = 0; j < n; j++)
    {
      double2 R = points[j].pos - points[i].pos;
      double Rlen = len(R);
      // if (Rlen <= 1)
      // {
      //   points[i].velocity = points[j].velocity;
      //   Rlen = 1;
      //   // continue;
      // }
      Rlen += 10;
      Rlen *= 1000;
      double acc = G / (Rlen * Rlen);
      double2 Rnorm = normalize(R) * acc;
      points[i].velocity = points[i].velocity + Rnorm * points[j].mass;
    }
  }
}

void initCUDA()
{
  gpuErrchk(cudaMallocManaged(&points, POINTS_COUNT * sizeof(Point)));

  for (size_t i = 0; i < POINTS_COUNT; ++i)
  {
    points[i] = Point();
    Point &pt = points[i];

    pt.pos.x = -50 * z + (rand() % 100 * z);
    pt.pos.y = -50 * z + (rand() % 100 * z);

    pt.pos = pt.pos + double2(100 * z / 2, 100 * z / 2);

    // pt.velocity = pt.pos * -1 + double2(1e5 * ((rand() % 10) - 10), 1e4 * ((rand() % 10) - 10));

    pt.velocity.x = -50 + (rand() % 100);
    pt.velocity.y = -50 + (rand() % 100);

    pt.velocity = pt.velocity * (rand() % 10000);

    int massc = rand() % 3;
    pt.mass = 1 * pow(10, massc + 31);
    //  pt.r = rand() % 255;
    //  pt.g = rand() % 255;
    //  pt.b = rand() % 255;
    pt.b = 255 - (int)(255.0 * (3.0 / (float)massc));
    // pt.g = (int)(255.0 * (10.0 / (float)massc));
    pt.g = 100;
    pt.r = (int)(255.0 * (3.0 / (float)massc));
    pt.a = 255;
  }
}

void ChangeSize(int w, int h)
{
  // glViewport(0, 0, w, h);

  GLdouble aspectRatio;
  if (h == 0)
    h = 1;

  double zoomh = 1000;

  glViewport(0 - zoomh, 0 - zoomh, w + zoomh, h + zoomh);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();

  aspectRatio = (GLdouble)w / (GLdouble)h;
  if (w <= h)
  {
    windowWidth = 100;
    windowHeight = 100 / aspectRatio;
    glOrtho(-100.0 * z, 100.0 * z, -windowHeight * z, windowHeight * z, 1.0, -1.0);
  }
  else
  {
    windowWidth = 100 * aspectRatio;
    windowHeight = 100;
    glOrtho(-windowWidth * z, windowWidth * z, -100.0 * z, 100.0 * z, 1.0, -1.0);
  }
}

void SetupRC()
{
  glClearColor(0.01f, 0.01f, 0.02f, 1.0f);
  cudaDeviceSynchronize();
  initCUDA();
}

size_t N = POINTS_COUNT;
int blockSize = 256;
int numBlocks = (N + blockSize - 1) / blockSize;

void Tick(double delta = 1)
{

  add<<<numBlocks, blockSize>>>(N, points);

  cudaDeviceSynchronize();

  for (size_t i = 0; i < POINTS_COUNT; i++)
  {
    points[i].velocity = points[i].velocity * 0.995;
    points[i].pos = points[i].pos + (points[i].velocity * delta);
  }

  glutPostRedisplay();
}

void RenderScene()
{
  glClear(GL_COLOR_BUFFER_BIT);

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // glMatrixMode(GL_PROJECTION);
  // glLoadIdentity();
  //  glOrtho(-10000, 15000, -15000, 15000, -1, 1);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  // draw
  glColor3ub(255, 255, 255);
  glEnableClientState(GL_VERTEX_ARRAY);
  glEnableClientState(GL_COLOR_ARRAY);
  glVertexPointer(2, GL_DOUBLE, sizeof(Point), &points[0].pos.x);
  glColorPointer(4, GL_UNSIGNED_BYTE, sizeof(Point), &points[0].r);
  glPointSize(1.0);
  glDrawArrays(GL_POINTS, 0, POINTS_COUNT);
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

  Tick((double)deltaTime / 1000.0);

  glutPostRedisplay();
  glutTimerFunc(1000 / 60, timer, 0);
}

int main(int argc, char *argv[])
{
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
  glutCreateWindow("galaxy-gpu");
  glutFullScreen();
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
