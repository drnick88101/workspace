#include <iostream>

using namespace std;

int main()
{
    //a program to print the total price of an item
    //the sale price of the item is $545.00 and the sales tax is 5%
    double sale_price = 545.00;
    double sale_tax = 0.05;
    double tax_amount;
    double total_price;

    tax_amount = sale_price * sale_tax;
    total_price = sale_price + tax_amount;

    cout<<"The total price is: "<<total_price<<endl;
    return 0;
}
