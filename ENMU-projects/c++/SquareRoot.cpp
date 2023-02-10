#include <iostream>
#include <cmath>

using namespace std;

int main()
{
    //a program to print the value of y, where y = square root of (b^2 + c^2)
    //b = 10, c = 5
    int b = 10;
    int c = 5;
    double y;

    y = sqrt(b*b + c*c);

    cout<<"The value of y is: "<<y<<endl;
    return 0;
}
