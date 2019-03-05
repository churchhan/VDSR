__kernel
void matrixMultiplication(__global float* A, __global float* B, __global float* C,  int widthA, int widthB )
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    float value=0;
    for ( int k = 0; k < widthA; k++)
    {
        value = value + A[k + j * widthA] * B[k*widthB + i];
    }
    C[i + widthA * j] = value;
}
