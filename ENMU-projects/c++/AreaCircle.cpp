#include <iostream>

using namespace std;

int main()
{
    //a program to print the area of a circle where the radius is 3
    //area= pi r^2
    int r = 3;
    const double PI = 3.14;
    double areaC;

    areaC = PI*r*r;

    cout<<"The area is: "<<areaC<<endl;
    return 0;
}
