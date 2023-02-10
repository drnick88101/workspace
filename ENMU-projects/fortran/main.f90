! multiply 2 matrices of random numbers together

program main
    implicit none
    
    INTEGER :: i,j,k,size
    REAL :: x,y,total
    REAL, ALLOCATABLE :: matrix1(:,:), matrix2(:,:), matrix3(:,:)
    character (len = 4) :: input
    
    CALL GET_COMMAND_ARGUMENT(1,input)
    READ(input,*)size

    ALLOCATE (matrix1(size,size), matrix2(size,size), matrix3(size,size))

    do i=1,size
        do j=1,size
            call RANDOM_NUMBER(x)
            y = x
            matrix1(i,j) = y
            call RANDOM_NUMBER(x)
            y = x
            matrix2(i,j) = y
        end do
    end do

    do i=1,size
        do j=1,size
            do k=1,size
                total = matrix1(i,k) * matrix2(k,j)
                matrix3(i,j) = matrix3(i,j) + total
            end do
        end do
    end do

end program main