/**
 * Larry Maes
 * 09/03/2021
 * CS301 Assignment 1
 *
 * a simple FOR loop that adds the first 100 positive integers (from 1 to 100, included).
 */
package CS301;
public class Assignment1 {
	public static void main(String[] args) {
		//set the sum to 0
		int sum = 0;

		//for loop to add i to sum 
		for (int i=1; i<=100; i++) {
			sum = sum + i;
		}

		//The output of sum
		System.out.print("The sum of the numbers is: " + sum);
		System.out.println();
	}		
}