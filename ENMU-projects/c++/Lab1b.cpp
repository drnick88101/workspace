#include <iostream>

using namespace std;

int main()
{
    int total_num = 5;
    int i = 0;
    int sum = 0;
    int element;
    int average;

    cout<<"Enter 5 integer numbers and press enter"<<endl;

    for (i=0; i<total_num; i++) {
        cin>>element;
        sum = sum + element;
    }

    average = sum/total_num;
    cout<<"The average value is: "<<average<<endl;
    return 0;
}
