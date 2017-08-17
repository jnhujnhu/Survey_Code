OBJECTS = svm.o data.o test.o grad_desc.o ridge.o blackbox.o
PARAMS = -c -fPIC -std=c++14
SO_PARAMS = -std=c++14 -shared -o

run:
	g++ -std=c++14 svm.cpp logistic.cpp ridge.cpp blackbox.cpp test.cpp data.cpp grad_desc.cpp -o test
lib: svmlib.so clean
svmlib.so: $(OBJECTS)
	g++ $(OBJECTS) $(SO_PARAMS) svmlib.so
svm.o: svm.cpp
	g++ $(PARAMS) svm.cpp
data.o: data.cpp
	g++ $(PARAMS) data.cpp
test.o: test.cpp
	g++ $(PARAMS) test.cpp
ridge.o: ridge.cpp
	g++ $(PARAMS) ridge.cpp
blackbox.o: blackbox.cpp
	g++ $(PARAMS) blackbox.cpp
grad_desc.o: grad_desc.cpp
	g++ $(PARAMS) grad_desc.cpp
clean:
	rm -f *.o svm.o data.o test.o
