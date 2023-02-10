#include <iostream>

using namespace std;

int main()
{
    int large;
    int input;
    int n=4;
    cout<<"Enter 4 integers: ";
    cin>>large;

    for (int i=1; i<=n-1; i++) {
            cin>>input;
            if (input>large) {
                    large=input;
            }
    }
    cout<<"The largest number is: "<<large<<endl;
    return 0;
}
