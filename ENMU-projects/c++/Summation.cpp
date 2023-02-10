#include <iostream>

using namespace std;

int main()
{
    //a program to calculate summation of first 3 numbers
    int sum = 0;
    int counter = 1;
    while (counter <=3) {
        sum = sum + counter;
        counter++;
    }
    cout<<"The summation of 1+2+3 is: "<<sum<<endl;
    return 0;
}
