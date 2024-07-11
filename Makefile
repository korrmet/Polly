all: clean
	g++ main.cpp -o main
	./main --train -i train_db -d db -v 2 -q "Hi! How do you do?" | tee log

clean:
	rm -rf main
