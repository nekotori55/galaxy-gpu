#include <cmath>
#include <cstdlib>

struct Vector2
{
    float x = 0;
    float y = 0;

    Vector2 operator+(const Vector2 &b)
    {
        return Vector2(x + b.x, y + b.y);
    };

    Vector2 operator-(const Vector2 &b)
    {
        return Vector2(x - b.x, y - b.y);
    };

    Vector2 &operator+=(const Vector2 &b)
    {
        x += b.x;
        y += b.y;
        return *this;
    };

    Vector2 &operator-=(const Vector2 &b)
    {
        x -= b.x;
        y -= b.y;
        return *this;
    };

    Vector2 operator*(const float &b)
    {
        return Vector2(x * b, y * b);
    };

    Vector2 operator/(float b)
    {
        return Vector2(x / b, y / b);
    };

    float len()
    {
        return sqrt(x * x + y * y);
    }

    Vector2 &normalize()
    {
        float l = len();

        if (l == 0)
        {
            return *this;
        }

        x /= l;
        y /= l;

        return *this;
    }

public:
    Vector2() : Vector2(0, 0){};
    Vector2(float x, float y) : x(x), y(y){};
};
