! simple command-line argument parsing example

program ReadTerminalInput
    implicit none
  
    character (len = 1) :: input
    integer (kind = 2) :: x
  
    CALL GET_COMMAND_ARGUMENT(1,input)
    READ(input,*)x
    print *, "The input was: ", x

  end program ReadTerminalInput