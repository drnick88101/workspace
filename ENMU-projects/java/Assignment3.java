public class Assignment3 {
    //recusive inverse sort
    public static void insertionSortRecursive(int[] arr, int n) {
        if (n <= 1) {
            return;
        }
        insertionSortRecursive(arr, n - 1);
        int key = arr[n - 1];
        int j = n - 2;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }

    public static void main (String[] args) {
        int[] arr = {5, 2, 4, 6, 1, 3};
        insertionSortRecursive(arr, arr.length);
        for (int i = 0; i < arr.length; i++) {
            System.out.print(arr[i] + " ");
        }
    }
}