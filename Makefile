all: compile link

compile:
	g++ -Isrc/include -c main.cpp

link:
	g++ main.o -o maze_solver -Lsrc/lib -lsfml-graphics -lsfml-window -lsfml-system
	
clean:
	rm -f *.o maze_solver.exe