OBJECTS = svm.o data.o test.o grad_desc.o
PARAMS = -c -fPIC -std=c++14
SO_PARAMS = -std=c++14 -shared -o

lib: svmlib.so clean
all:
	g++ -std=c++14 new_svm.cpp test.cpp data.cpp grad_desc.cpp -o test
svmlib.so: $(OBJECTS)
	g++ $(OBJECTS) $(SO_PARAMS) svmlib.so
svm.o: svm.cpp
	g++ $(PARAMS) svm.cpp
data.o: data.cpp
	g++ $(PARAMS) data.cpp
test.o: test.cpp
	g++ $(PARAMS) test.cpp
grad_desc.o: grad_desc.cpp
	g++ $(PARAMS) grad_desc.cpp
clean:
	rm -f *.o svm.o data.o test.o
