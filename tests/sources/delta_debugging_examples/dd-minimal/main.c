#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

#define MAGIC_NUMBER 100

void magicLoop() {
    for (int i = 0; i < MAGIC_NUMBER; ++i) {
        usleep(500000);
    }
}

void checkInputString(const char* inputString) {
    int count = 0;
    for (int i = 0; inputString[i] != '\0'; ++i) {
        char character = inputString[i];
        if (character == '-') {
            ++count;
            if (count == 3) {
                magicLoop();
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        return 1;
    }

    checkInputString(argv[1]);
    return 0;
}
