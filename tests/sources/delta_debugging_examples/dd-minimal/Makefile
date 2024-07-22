CC=gcc
PROG=dd-minimal

all: $(PROG)

dd-minimal: main.c
	$(CC) $< -g --coverage -o $@

clean:
	rm -rf $(PROG) *.gc*
