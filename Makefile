VOL = 2

all: clean
	g++ main.cpp -o main
	./main --train -i train_db -d db -v $(VOL)
	./main -d db -v $(VOL) -q "Hi! How do you do?"
	./main -d db -v $(VOL) -q "What's your name?"
	./main -d db -v $(VOL) -q "Do you want a cracker?"

clean:
	rm -rf main
