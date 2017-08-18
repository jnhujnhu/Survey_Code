OBJECTS = svm.o data.o test.o grad_desc.o least_square.o blackbox.o logistic.o regularizer.o
PARAMS = -c -fPIC -std=c++14
SO_PARAMS = -std=c++14 -shared -o

run:
	g++ -std=c++14 svm.cpp logistic.cpp least_square.cpp blackbox.cpp regularizer.o test.cpp data.cpp grad_desc.cpp -o test
lib: svmlib.so clean
svmlib.so: $(OBJECTS)
	g++ $(OBJECTS) $(SO_PARAMS) svmlib.so
svm.o: svm.cpp
	g++ $(PARAMS) svm.cpp
data.o: data.cpp
	g++ $(PARAMS) data.cpp
test.o: test.cpp
	g++ $(PARAMS) test.cpp
least_square.o: least_square.cpp
	g++ $(PARAMS) least_square.cpp
logistic.o: logistic.cpp
	g++ $(PARAMS) logistic.cpp
blackbox.o: blackbox.cpp
	g++ $(PARAMS) blackbox.cpp
regularizer.o: regularizer.cpp
	g++ $(PARAMS) regularizer.cpp
grad_desc.o: grad_desc.cpp
	g++ $(PARAMS) grad_desc.cpp
clean:
	rm -f *.o svm.o data.o test.o
