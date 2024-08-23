
// struct Vector2
// {
//     double x = 0;
//     double y = 0;

//     Vector2 operator+(const Vector2 &b)
//     {
//         return Vector2(x + b.x, y + b.y);
//     };

//     Vector2 operator-(const Vector2 &b)
//     {
//         return Vector2(x - b.x, y - b.y);
//     };

//     Vector2 &operator+=(const Vector2 &b)
//     {
//         x += b.x;
//         y += b.y;
//         return *this;
//     };

//     Vector2 &operator-=(const Vector2 &b)
//     {
//         x -= b.x;
//         y -= b.y;
//         return *this;
//     };

//     Vector2 operator*(const double &b)
//     {
//         return Vector2(x * b, y * b);
//     };

//     Vector2 operator/(double b)
//     {
//         return Vector2(x / b, y / b);
//     };

//     double len()
//     {
//         return sqrt(x * x + y * y);
//     }

//     Vector2 &normalize()
//     {
//         double l = len();

//         if (l == 0)
//         {
//             return *this;
//         }

//         x /= l;
//         y /= l;

//         return *this;
//     }

// public:
//     Vector2() : Vector2(0, 0){};
//     Vector2(double x, double y) : x(x), y(y){};
// };

__device__ __host__ inline double len(double2 vector)
{
    return sqrtf(vector.x * vector.x + vector.y * vector.y);
}

__device__ __host__ inline double2 &normalize(double2 &vector)
{
    double l = len(vector);

    if (l == 0)
    {
        return vector;
    }

    vector.x /= l;
    vector.y /= l;

    return vector;
}

__device__ __host__ inline double2 operator-(double2 &v1, double2 &v2)
{
    return double2(v1.x - v2.x, v1.y - v2.y);
}

__device__ __host__ inline double2 operator+(double2 v1, double2 v2)
{
    return double2(v1.x + v2.x, v1.y + v2.y);
}

__device__ __host__ inline double2 operator*(double2 v, double n)
{
    return double2(v.x * n, v.y * n);
}

__device__ __host__ inline double2 operator/(double2 &v, double n)
{
    return double2(v.x / n, v.y / n);
}