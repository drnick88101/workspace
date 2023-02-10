! multiply 2 matrices of random numbers together


program main
    implicit none
    
    INTEGER :: i,j,k,z,total,size
    REAL :: x,y
    INTEGER, ALLOCATABLE :: matrix1(:,:), matrix2(:,:), matrix3(:,:)
    character (len = 1) :: input
    
    CALL GET_COMMAND_ARGUMENT(1,input)
    READ(input,*)size

    ALLOCATE (matrix1(size,size), matrix2(size,size), matrix3(size,size))

    do i=1,size
        do j=1,size
            call RANDOM_NUMBER(x)
            y = x * 10
            z = int(y)
            matrix1(i,j) = z
            call RANDOM_NUMBER(x)
            y = x * 10
            z = int(y)
            matrix2(i,j) = z
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
    
    do i=1,size
        write(*,*) ((matrix1(i,j)),j=1,size)
    end do

    print '(A)'

    do i=1,size
        write(*,*) ((matrix2(i,j)),j=1,size)
    end do

    print '(A)'

    do i=1,size
        write(*,*) ((matrix3(i,j)),j=1,size)
    end do

end program main