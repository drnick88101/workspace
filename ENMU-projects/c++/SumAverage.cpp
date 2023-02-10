#include <iostream>

using namespace std;

int main()
{
    //a program to calculate the sum and average from 5 integer numbers entered by user
    int i;
    int n = 5;
    int input;
    int sum = 0;
    int average;
    cout<<"Enter 5 integer numbers: ";

    for (i=1; i<=n; i++) {
            cin>>input;
            sum = sum + input;
    }
    average = sum/n;
    cout<<"Sum: "<<sum<<endl;
    cout<<"Average: "<<average;
    return 0;
}
