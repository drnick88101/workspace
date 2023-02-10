#include <stdio.h>
#include <stdlib.h>

int main()
{
    int x;
    pid_t pid;

    while (x <= 0) {
        printf("Enter a number greater than 0.\n");
        scanf("%d", &x);
    }
    pid = fork();

    if (pid == 0) {
        printf("Child process.\n");
        while (x != 1) {
            if (x%2 == 0) {
                x = x/2;
            }
            else {
                x = (3 * x) + 1;
            }
            printf("%d ", x);
        }
    }
    else {
        printf("Parent process.\n");
        wait();
    }
    return 0;
}
