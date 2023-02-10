/**
 * Larry Maes
 * 09/11/2021
 * CS301 Assignment 2
 *
 * A program to multiply two square matrices of random numbers.
 * The sizes of these matrices are: 250, 500, 1000, 1500 and 2000 elements.
 */
package CS301;

import java.util.Arrays;

public class Assignment2 {
	public static void main(String[] args) {
        int matrixsize = 5;
		int[][] matrix1 = new int[matrixsize][matrixsize];
        int[][] matrix2 = new int[matrixsize][matrixsize];
        int[][] matrix3 = new int[matrixsize][matrixsize];
        int total;
   
        for (int i=0; i<matrixsize; i++) {
            for (int j=0; j<matrixsize; j++) {
                matrix1[i][j] = (int)(Math.random()*10);
                matrix2[i][j] = (int)(Math.random()*10);
            }
        }

        for (int i=0; i<matrixsize; i++) {
            for (int j=0; j<matrixsize; j++) {
                for (int k=0; k<matrixsize; k++) {
                    total = matrix1[i][k] * matrix2[k][j];
                    matrix3[i][j] = matrix3[i][j] + total;
                }
            }
        }

        System.out.println();
        System.out.println("Matrix 1");
        for (int row = 0; row < matrix1.length; row++) {
            for (int col = 0; col < matrix1[row].length; col++) {
                System.out.printf("%d ", matrix1[row][col]);
            }
            System.out.println();
        }

        System.out.println();
        System.out.println("Matrix 2");
        for (int row = 0; row < matrix2.length; row++) {
            for (int col = 0; col < matrix2[row].length; col++) {
                System.out.printf("%d ", matrix2[row][col]);
            }
            System.out.println();
        }

        System.out.println();
        System.out.println("Matrix 3");
        for (int row = 0; row < matrix3.length; row++) {
            for (int col = 0; col < matrix3[row].length; col++) {
                System.out.printf("%3d ", matrix3[row][col]);
            }
            System.out.println();
        }
    }
}