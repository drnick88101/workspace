#include <iostream>

using namespace std;

int main()
{
    //a program to calculate the factorial of 3
    int counter;
    int fact = 1;
    for (counter=1; counter <=3; counter++) {
        fact = fact * counter;
    }
    cout<<"The factorial of 3 is: "<<fact<<endl;
    return 0;
}
